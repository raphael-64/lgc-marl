"""Progressive depth evolution for graph refinement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..environments.rware_wrapper import RWAREGraphWrapper
from ..graph_generation.graph_types import GraphCandidate, TaskGraph
from ..graph_generation.llm_planner import LLMGraphPlanner
from ..lgc_marl.graph_policy import GraphConditionedPolicy
from ..lgc_marl.marl_trainer import MARLTrainer
from .policy_transfer import PolicyTransferManager, create_child_policy
from .selection import ElitistSelection, SelectionStrategy

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    """Configuration for an evolution stage."""

    candidates: int  # Number of candidates in this stage
    episodes: int  # Training episodes per candidate


@dataclass
class StageResult:
    """Results from a single evolution stage."""

    stage_idx: int
    candidates: List[GraphCandidate]
    policies: List[GraphConditionedPolicy]
    performances: List[Dict[str, Any]]
    trajectories: List[List[Dict[str, Any]]]
    best_idx: int
    best_performance: Dict[str, Any]
    insights: Dict[str, Any]


class ProgressiveDepthEvolution:
    """
    Progressive depth evolution for graph refinement.

    Key idea:
    - Start with many candidates, few training episodes
    - Progressively reduce candidates, increase training
    - Evolve graphs between stages based on performance
    - Transfer policies to avoid wasting training

    Stages example:
    - Stage 1: 8 candidates × 50 episodes = 400 total
    - Stage 2: 4 candidates × 150 episodes = 600 total
    - Stage 3: 2 candidates × 400 episodes = 800 total
    - Stage 4: 1 candidate × 400 episodes = 400 total
    Total: 2200 episodes vs naive 8000 (8 × 1000)
    """

    def __init__(
        self,
        env: RWAREGraphWrapper,
        planner: LLMGraphPlanner,
        stages: Optional[List[Stage]] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        policy_hidden_dim: int = 128,
        policy_graph_dim: int = 64,
        trainer_lr: float = 3e-4,
        device: str = "cuda",
    ):
        """
        Initialize progressive depth evolution.

        Args:
            env: RWARE environment wrapper
            planner: LLM graph planner
            stages: List of stage configurations
            selection_strategy: Strategy for selecting survivors
            obs_dim: Observation dimension (auto-detected if None)
            action_dim: Action dimension (auto-detected if None)
            policy_hidden_dim: Hidden dimension for policy network
            policy_graph_dim: Graph embedding dimension
            trainer_lr: Learning rate for MARL training
            device: Torch device
        """
        self.env = env
        self.planner = planner
        self.device = device

        # Auto-detect dimensions
        self.obs_dim = obs_dim or env.get_obs_dim()
        self.action_dim = action_dim or env.get_action_dim()
        self.n_agents = env.n_agents

        # Policy config
        self.policy_hidden_dim = policy_hidden_dim
        self.policy_graph_dim = policy_graph_dim
        self.trainer_lr = trainer_lr

        # Default stages
        self.stages = stages or [
            Stage(candidates=8, episodes=50),
            Stage(candidates=4, episodes=150),
            Stage(candidates=2, episodes=400),
            Stage(candidates=1, episodes=400),
        ]

        # Selection strategy
        self.selection = selection_strategy or ElitistSelection()

        # Policy transfer manager
        self.transfer_manager = PolicyTransferManager()

        # Track all results
        self.all_stage_results: List[StageResult] = []

        logger.info(
            f"ProgressiveDepthEvolution initialized: "
            f"{len(self.stages)} stages, "
            f"{sum(s.candidates * s.episodes for s in self.stages)} total episodes"
        )

    def run(
        self, task: str, wandb_log: bool = True
    ) -> Tuple[GraphCandidate, GraphConditionedPolicy, List[StageResult]]:
        """
        Run full progressive depth evolution.

        Args:
            task: Task description
            wandb_log: Whether to log to W&B

        Returns:
            best_candidate: Best graph found
            best_policy: Trained policy for best graph
            history: List of stage results
        """
        # Try to import wandb
        try:
            import wandb

            wandb_available = wandb_log and wandb.run is not None
        except ImportError:
            wandb_available = False

        env_state = self.env.get_env_state()

        # Initialize first stage population
        logger.info("Generating initial graph candidates...")
        candidates = self.planner.generate_initial_graphs(
            env_state=env_state,
            task=task,
            n_candidates=self.stages[0].candidates,
        )

        # Initialize policies for each candidate
        policies = [self._create_policy() for _ in candidates]
        trajectories = [[] for _ in candidates]

        self.all_stage_results = []
        total_episodes = 0

        for stage_idx, stage in enumerate(self.stages):
            logger.info(
                f"\n{'='*50}\n"
                f"Stage {stage_idx + 1}/{len(self.stages)}: "
                f"{len(candidates)} candidates × {stage.episodes} episodes\n"
                f"{'='*50}"
            )

            # Run stage
            stage_result = self._run_stage(
                stage_idx=stage_idx,
                stage=stage,
                candidates=candidates,
                policies=policies,
                trajectories=trajectories,
                task=task,
                env_state=env_state,
            )

            self.all_stage_results.append(stage_result)
            total_episodes += len(candidates) * stage.episodes

            # Log stage summary
            if wandb_available:
                wandb.log(
                    {
                        f"stage_{stage_idx}/best_success_rate": stage_result.best_performance[
                            "success_rate"
                        ],
                        f"stage_{stage_idx}/mean_success_rate": np.mean(
                            [p["success_rate"] for p in stage_result.performances]
                        ),
                        f"stage_{stage_idx}/n_candidates": len(candidates),
                        "total_episodes": total_episodes,
                    }
                )

            logger.info(
                f"Stage {stage_idx + 1} complete: "
                f"best={stage_result.best_performance['success_rate']:.1%}, "
                f"mean={np.mean([p['success_rate'] for p in stage_result.performances]):.1%}"
            )

            # Evolve to next stage (except last)
            if stage_idx < len(self.stages) - 1:
                next_stage = self.stages[stage_idx + 1]
                candidates, policies, trajectories = self._evolve_to_next_stage(
                    stage_result=stage_result,
                    target_n=next_stage.candidates,
                    task=task,
                    env_state=env_state,
                )

        # Return best from final stage
        final_result = self.all_stage_results[-1]
        best_idx = final_result.best_idx
        best_candidate = final_result.candidates[best_idx]
        best_policy = final_result.policies[best_idx]

        logger.info(
            f"\nEvolution complete: "
            f"best success rate = {best_candidate.performance['success_rate']:.1%}"
        )

        return best_candidate, best_policy, self.all_stage_results

    def _create_policy(self) -> GraphConditionedPolicy:
        """Create a new policy network."""
        return GraphConditionedPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            n_agents=self.n_agents,
            graph_dim=self.policy_graph_dim,
            hidden_dim=self.policy_hidden_dim,
        )

    def _run_stage(
        self,
        stage_idx: int,
        stage: Stage,
        candidates: List[GraphCandidate],
        policies: List[GraphConditionedPolicy],
        trajectories: List[List[Dict]],
        task: str,
        env_state: Dict,
    ) -> StageResult:
        """Run a single evolution stage."""
        # Try to import wandb for logging
        try:
            import wandb
            wandb_available = wandb.run is not None
        except ImportError:
            wandb_available = False

        performances = []
        global_episode = sum(
            sr.stage_idx * len(sr.candidates) * self.stages[sr.stage_idx].episodes
            for sr in self.all_stage_results
        ) if self.all_stage_results else 0

        for i, (candidate, policy) in enumerate(zip(candidates, policies)):
            logger.info(f"  Training candidate {i + 1}/{len(candidates)}: {candidate.origin}")

            # Create trainer
            trainer = MARLTrainer(
                env=self.env,
                policy=policy,
                lr=self.trainer_lr,
                device=self.device,
            )

            # Train and log episode metrics
            new_trajectory = trainer.train_episodes(
                graph=candidate.graph,
                n_episodes=stage.episodes,
            )
            trajectories[i].extend(new_trajectory)

            # Log episode-level metrics to W&B
            if wandb_available:
                for ep_idx, ep_metrics in enumerate(new_trajectory):
                    global_episode += 1
                    wandb.log({
                        "episode": global_episode,
                        "reward": ep_metrics["reward"],
                        "episode_length": ep_metrics["episode_length"],
                        "policy_loss": ep_metrics.get("policy_loss", 0),
                        "value_loss": ep_metrics.get("value_loss", 0),
                        "entropy": ep_metrics.get("entropy", 0),
                        "success": float(ep_metrics.get("success", False)),
                        "stage": stage_idx,
                        "candidate": candidate.origin,
                    })

            # Log running average to console
            if len(new_trajectory) >= 10:
                recent = new_trajectory[-10:]
                avg_reward = np.mean([t["reward"] for t in recent])
                avg_loss = np.mean([t.get("policy_loss", 0) for t in recent])
                success_rate = np.mean([float(t.get("success", False)) for t in recent])
                logger.info(
                    f"    [Last 10 eps] reward={avg_reward:.3f}, "
                    f"policy_loss={avg_loss:.4f}, success={success_rate:.0%}"
                )

            # Evaluate
            performance = trainer.evaluate(graph=candidate.graph, n_episodes=20)

            # Analyze failures
            failure_analysis = trainer.analyze_failures(
                graph=candidate.graph, n_episodes=10
            )
            performance["failure_analysis"] = failure_analysis

            # Compute strengths
            performance["strengths"] = self._analyze_strengths(candidate, performance)

            performances.append(performance)

            # Update candidate
            candidate.performance = performance
            candidate.trajectory = trajectories[i]

            # Cache policy
            self.transfer_manager.cache_policy(candidate, policy, performance)

            logger.info(
                f"    success_rate={performance['success_rate']:.1%}, "
                f"avg_reward={performance['avg_reward']:.2f}"
            )

        # Find best
        best_idx = max(
            range(len(performances)), key=lambda i: performances[i]["success_rate"]
        )

        # Aggregate insights
        insights = self._aggregate_insights(candidates, performances, trajectories)

        return StageResult(
            stage_idx=stage_idx,
            candidates=candidates,
            policies=policies,
            performances=performances,
            trajectories=trajectories,
            best_idx=best_idx,
            best_performance=performances[best_idx],
            insights=insights,
        )

    def _evolve_to_next_stage(
        self,
        stage_result: StageResult,
        target_n: int,
        task: str,
        env_state: Dict,
    ) -> Tuple[List[GraphCandidate], List[GraphConditionedPolicy], List[List[Dict]]]:
        """Evolve population to next stage."""
        logger.info(f"  Evolving: {len(stage_result.candidates)} -> {target_n} candidates")

        candidates = stage_result.candidates
        policies = stage_result.policies
        performances = stage_result.performances
        trajectories = stage_result.trajectories
        insights = stage_result.insights

        # Add n_agents to insights for planner
        insights["n_agents"] = self.n_agents
        insights["current_generation"] = stage_result.stage_idx

        # Rank by performance
        ranked_indices = self.selection.select(
            candidates, policies, performances, n_select=len(candidates)
        )

        new_candidates = []
        new_policies = []
        new_trajectories = []

        # How many elites vs evolved?
        n_elites = max(1, target_n // 2)
        n_evolved = target_n - n_elites

        # === ELITES: Keep top performers as-is ===
        for i in ranked_indices[:n_elites]:
            new_candidates.append(candidates[i])
            new_policies.append(policies[i])
            new_trajectories.append(trajectories[i])
            logger.info(f"    Elite: {candidates[i].origin} (success={performances[i]['success_rate']:.1%})")

        # === EVOLVED: Generate improved variants ===
        evolved_count = 0

        # 1. Crossover top 2 (if we need evolved candidates)
        if n_evolved > 0 and len(ranked_indices) >= 2:
            p1, p2 = ranked_indices[0], ranked_indices[1]

            child_candidate = self.planner.crossover(
                parent1=candidates[p1],
                parent1_perf=performances[p1],
                parent2=candidates[p2],
                parent2_perf=performances[p2],
                insights=insights,
            )

            # Transfer policy from better parent
            child_policy = create_child_policy(
                parent_policies=[
                    (policies[p1], performances[p1]["success_rate"]),
                    (policies[p2], performances[p2]["success_rate"]),
                ],
                target_graph=child_candidate.graph,
            )

            new_candidates.append(child_candidate)
            new_policies.append(child_policy)
            new_trajectories.append([])
            evolved_count += 1
            logger.info(f"    Crossover: from {candidates[p1].origin} × {candidates[p2].origin}")

        # 2. Mutations
        while evolved_count < n_evolved:
            parent_idx = ranked_indices[evolved_count % len(ranked_indices)]

            mutated_candidate = self.planner.mutate(
                parent=candidates[parent_idx],
                parent_perf=performances[parent_idx],
                insights=insights,
            )

            mutated_policy = self.transfer_manager.transfer_policy(
                source_policy=policies[parent_idx],
                source_graph=candidates[parent_idx].graph,
                target_graph=mutated_candidate.graph,
            )

            new_candidates.append(mutated_candidate)
            new_policies.append(mutated_policy)
            new_trajectories.append([])
            evolved_count += 1
            logger.info(f"    Mutation: from {candidates[parent_idx].origin}")

        return new_candidates, new_policies, new_trajectories

    def _aggregate_insights(
        self,
        candidates: List[GraphCandidate],
        performances: List[Dict],
        trajectories: List[List[Dict]],
    ) -> Dict[str, Any]:
        """Extract patterns from all candidates."""
        # Sort by performance
        sorted_indices = sorted(
            range(len(candidates)),
            key=lambda i: performances[i]["success_rate"],
            reverse=True,
        )

        n_top = max(1, len(candidates) // 3)
        n_bottom = max(1, len(candidates) // 3)

        top_indices = sorted_indices[:n_top]
        bottom_indices = sorted_indices[-n_bottom:]

        # Extract patterns
        successful_patterns = self._extract_patterns(
            [candidates[i] for i in top_indices],
            [performances[i] for i in top_indices],
        )

        failure_patterns = self._extract_patterns(
            [candidates[i] for i in bottom_indices],
            [performances[i] for i in bottom_indices],
        )

        # Aggregate failure analysis
        all_failures = []
        for i in bottom_indices:
            fa = performances[i].get("failure_analysis", {})
            all_failures.extend(fa.get("common_failure_points", []))

        return {
            "successful_patterns": successful_patterns,
            "failure_patterns": failure_patterns,
            "common_failures": list(set(all_failures)),
            "generation_best_scores": [
                max(p["success_rate"] for p in performances)
            ],
            "n_agents": self.n_agents,
        }

    def _extract_patterns(
        self, candidates: List[GraphCandidate], performances: List[Dict]
    ) -> str:
        """Extract common patterns from a set of candidates."""
        patterns = []

        for candidate, perf in zip(candidates, performances):
            graph = candidate.graph

            # Structure metrics
            parallelism = graph.get_parallelism_score()
            n_subtasks = len(graph.subtasks)
            load_balance = graph.get_agent_load_balance()

            patterns.append(
                f"- {candidate.origin}: parallelism={parallelism:.2f}, "
                f"subtasks={n_subtasks}, load={load_balance}, "
                f"success={perf['success_rate']:.1%}"
            )

        return "\n".join(patterns) if patterns else "No patterns found"

    def _analyze_strengths(
        self, candidate: GraphCandidate, performance: Dict
    ) -> str:
        """Analyze what's working well in this candidate."""
        strengths = []

        # High parallelism
        parallelism = candidate.graph.get_parallelism_score()
        if parallelism > 1.5:
            strengths.append("Good parallelism")

        # Balanced load
        load = candidate.graph.get_agent_load_balance()
        if load:
            load_values = list(load.values())
            if max(load_values) - min(load_values) <= 1:
                strengths.append("Balanced workload")

        # Low failure rate
        failure_analysis = performance.get("failure_analysis", {})
        if failure_analysis.get("failure_rate", 1) < 0.3:
            strengths.append("Few coordination failures")

        # Fast completion
        if performance.get("avg_episode_length", 500) < 300:
            strengths.append("Fast completion")

        return "; ".join(strengths) if strengths else "No notable strengths"

    def get_compute_summary(self) -> Dict[str, Any]:
        """Get summary of compute used."""
        total_episodes = sum(
            s.candidates * s.episodes for s in self.stages
        )
        naive_episodes = self.stages[0].candidates * sum(s.episodes for s in self.stages)

        return {
            "total_episodes": total_episodes,
            "naive_episodes": naive_episodes,
            "savings": 1 - total_episodes / naive_episodes,
            "stages": [
                {"candidates": s.candidates, "episodes": s.episodes}
                for s in self.stages
            ],
        }
