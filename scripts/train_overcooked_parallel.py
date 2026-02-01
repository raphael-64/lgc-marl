#!/usr/bin/env python3
"""Parallel training script for LGC-MARL on Overcooked using Ray."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import copy

from dotenv import load_dotenv
load_dotenv()

import weave
import torch
import wandb
import ray

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.overcooked_wrapper import make_overcooked_env
from src.graph_generation.overcooked_planner import OvercookedGraphPlanner
from src.lgc_marl.marl_trainer import MARLTrainer
from src.lgc_marl.graph_policy import GraphConditionedPolicy
from src.graph_generation.graph_types import GraphCandidate, TaskGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@ray.remote
def train_candidate_remote(
    graph_dict: Dict,
    n_episodes: int,
    layout: str,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    device: str,
    policy_state_dict: Dict = None,
) -> Tuple[Dict, Dict, str]:
    """
    Train a single candidate in a Ray worker.

    Returns: (performance_dict, policy_state_dict, origin)
    """
    # Reconstruct graph from dict
    from src.graph_generation.graph_types import TaskGraph, Subtask, SubtaskType

    graph = TaskGraph()
    for st_dict in graph_dict["subtasks"]:
        subtask = Subtask(
            id=st_dict["id"],
            task_type=SubtaskType(st_dict["task_type"]),
            agent_id=st_dict["agent_id"],
            target=st_dict.get("target"),
            dependencies=st_dict.get("dependencies", []),
        )
        graph.add_subtask(subtask)

    # Create fresh env and policy for this worker
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)

    policy = GraphConditionedPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        hidden_dim=128,
        graph_dim=32,
    )

    # Load policy state if provided (for later stages)
    if policy_state_dict:
        policy.load_state_dict(policy_state_dict)

    trainer = MARLTrainer(env=env, policy=policy, device=device)

    metrics_history = []
    for ep in range(n_episodes):
        metrics = trainer.train_episode(graph)
        metrics_history.append(metrics)

        if (ep + 1) % 50 == 0:
            recent = metrics_history[-50:]
            avg_reward = sum(m["reward"] for m in recent) / len(recent)
            avg_deliveries = sum(m.get("deliveries", 0) for m in recent) / len(recent)
            print(f"    [{graph_dict['origin']}] Ep {ep+1}/{n_episodes}: reward={avg_reward:.2f}, deliveries={avg_deliveries:.1f}")

    # Compute final performance
    recent = metrics_history[-max(10, n_episodes // 5):]
    performance = {
        "avg_reward": sum(m["reward"] for m in recent) / len(recent),
        "avg_deliveries": sum(m.get("deliveries", 0) for m in recent) / len(recent),
        "success_rate": sum(1 for m in recent if m.get("deliveries", 0) > 0) / len(recent),
        "avg_policy_loss": sum(m.get("policy_loss", 0) for m in recent) / len(recent),
        "avg_value_loss": sum(m.get("value_loss", 0) for m in recent) / len(recent),
        "avg_entropy": sum(m.get("entropy", 0) for m in recent) / len(recent),
        "avg_episode_length": sum(m.get("episode_length", 0) for m in recent) / len(recent),
    }

    return performance, policy.state_dict(), graph_dict["origin"]


def graph_to_dict(graph: TaskGraph, origin: str) -> Dict:
    """Convert TaskGraph to serializable dict for Ray."""
    subtasks = []
    for st in graph.subtasks.values():
        subtasks.append({
            "id": st.id,
            "task_type": st.task_type.value,
            "agent_id": st.agent_id,
            "target": st.target,
            "dependencies": st.dependencies,
        })
    return {"subtasks": subtasks, "origin": origin}


def run_evolution_parallel(
    layout: str = "cramped_room",
    n_stages: int = 7,
    model: str = "gpt-5.2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_wandb: bool = True,
    n_parallel: int = 4,  # Number of parallel workers
):
    """Run progressive depth evolution with parallel candidate training."""

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=n_parallel, num_gpus=1 if device == "cuda" else 0)

    # Stage configs - wide exploration, slow convergence, many transitions
    # Early stages: shallow depth, many candidates (parallel)
    # Later stages: deeper training, fewer candidates
    # Final stage: long polish run
    stage_configs = [
        (16, 100),   # Stage 1: Very wide exploration - PARALLEL
        (14, 150),   # Stage 2: Still wide - PARALLEL
        (12, 200),   # Stage 3: Narrowing slightly - PARALLEL
        (10, 250),   # Stage 4: More focus - PARALLEL
        (8, 300),    # Stage 5: Getting serious - PARALLEL
        (6, 400),    # Stage 6: Top performers
        (5, 500),    # Stage 7: Deeper training
        (4, 700),    # Stage 8: Narrowing
        (3, 1000),   # Stage 9: Top 3
        (2, 1500),   # Stage 10: Final 2
        (1, 4000),   # Stage 11: Best candidate, long polish
    ][:n_stages]

    logger.info("=" * 60)
    logger.info("LGC-MARL Overcooked Training (PARALLEL)")
    logger.info("=" * 60)
    logger.info(f"Layout: {layout}")
    logger.info(f"Model: {model}")
    logger.info(f"Device: {device}")
    logger.info(f"Parallel workers: {n_parallel}")
    logger.info(f"Stages: {stage_configs}")

    if use_wandb:
        wandb.init(
            project="lgc-marl-overcooked",
            config={
                "layout": layout,
                "model": model,
                "stages": stage_configs,
                "parallel": True,
                "n_parallel": n_parallel,
            },
            tags=["parallel"],
        )

    weave.init("lgc-marl-overcooked")
    logger.info("Weave initialized for LLM tracing")

    # Create environment for graph generation
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)
    env_state = env.get_env_state()

    obs_dim = env._obs_shape_per_agent
    action_dim = env.n_actions
    n_agents = env.n_agents

    logger.info(f"\nEnvironment:")
    logger.info(f"  Layout: {layout}")
    logger.info(f"  Grid: {env.mdp.width}x{env.mdp.height}")
    logger.info(f"  Obs dim: {obs_dim}")
    logger.info(f"  Actions: {action_dim}")

    # Create planner
    planner = OvercookedGraphPlanner(model=model, temperature=0.7)

    task = "Make and serve onion soup"
    global_step = 0
    best_policy_state = None  # Track best policy across stages

    # Generate initial candidates
    logger.info(f"\nGenerating initial graphs...")
    candidates = planner.generate_initial_graphs(env_state, task, n_candidates=stage_configs[0][0])

    for candidate in candidates:
        logger.info(f"  {candidate.origin}: {len(candidate.graph)} subtasks")

    # Progressive depth evolution
    for stage_idx, (n_candidates, n_episodes) in enumerate(stage_configs):
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE {stage_idx + 1}: {n_candidates} candidates × {n_episodes} episodes")
        logger.info(f"{'='*60}")

        if use_wandb:
            wandb.log({"stage": stage_idx + 1}, step=global_step)

        # Decide whether to parallelize this stage
        # Parallelize early stages (many candidates, fewer episodes)
        use_parallel = (stage_idx < 2 and n_candidates >= 4)

        if use_parallel:
            logger.info(f"  Running {n_candidates} candidates in PARALLEL")

            # Convert graphs to dicts for Ray serialization
            graph_dicts = [
                graph_to_dict(c.graph, c.origin)
                for c in candidates[:n_candidates]
            ]

            # Launch parallel training
            futures = [
                train_candidate_remote.remote(
                    graph_dict=gd,
                    n_episodes=n_episodes,
                    layout=layout,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    device=device,
                    policy_state_dict=best_policy_state,
                )
                for gd in graph_dicts
            ]

            # Collect results
            results = ray.get(futures)

            # Update candidates with performance
            best_reward = -float('inf')
            for i, (perf, policy_state, origin) in enumerate(results):
                candidates[i].performance = perf
                logger.info(f"  {origin}: reward={perf['avg_reward']:.2f}, "
                           f"deliveries={perf['avg_deliveries']:.1f}")

                # Track best policy
                if perf['avg_reward'] > best_reward:
                    best_reward = perf['avg_reward']
                    best_policy_state = policy_state

                if use_wandb:
                    wandb.log({
                        "reward": perf["avg_reward"],
                        "deliveries": perf["avg_deliveries"],
                        "success_rate": perf["success_rate"],
                        "candidate_origin": origin,
                    }, step=global_step + i * n_episodes)

            global_step += n_candidates * n_episodes

        else:
            logger.info(f"  Running {n_candidates} candidates SEQUENTIALLY")

            # Create policy with best state so far
            policy = GraphConditionedPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                hidden_dim=128,
                graph_dim=32,
            )
            if best_policy_state:
                policy.load_state_dict(best_policy_state)

            trainer = MARLTrainer(env=env, policy=policy, device=device)

            for i, candidate in enumerate(candidates[:n_candidates]):
                logger.info(f"\n  Training candidate {i+1}/{n_candidates}: {candidate.origin}")

                metrics_history = []
                for ep in range(n_episodes):
                    metrics = trainer.train_episode(candidate.graph)
                    metrics_history.append(metrics)

                    if use_wandb:
                        wandb.log({
                            "ep_reward": metrics["reward"],
                            "ep_deliveries": metrics.get("deliveries", 0),
                            "ep_length": metrics.get("episode_length", 0),
                            "policy_loss": metrics.get("policy_loss", 0),
                            "value_loss": metrics.get("value_loss", 0),
                            "entropy": metrics.get("entropy", 0),
                        }, step=global_step + ep)

                    if (ep + 1) % 50 == 0:
                        recent = metrics_history[-50:]
                        avg_reward = sum(m["reward"] for m in recent) / len(recent)
                        avg_deliveries = sum(m.get("deliveries", 0) for m in recent) / len(recent)
                        logger.info(f"    [Ep {ep+1}/{n_episodes}] reward={avg_reward:.2f}, deliveries={avg_deliveries:.1f}")

                # Compute performance
                recent = metrics_history[-max(10, n_episodes // 5):]
                performance = {
                    "avg_reward": sum(m["reward"] for m in recent) / len(recent),
                    "avg_deliveries": sum(m.get("deliveries", 0) for m in recent) / len(recent),
                    "success_rate": sum(1 for m in recent if m.get("deliveries", 0) > 0) / len(recent),
                    "avg_policy_loss": sum(m.get("policy_loss", 0) for m in recent) / len(recent),
                    "avg_value_loss": sum(m.get("value_loss", 0) for m in recent) / len(recent),
                    "avg_entropy": sum(m.get("entropy", 0) for m in recent) / len(recent),
                    "avg_episode_length": sum(m.get("episode_length", 0) for m in recent) / len(recent),
                }
                candidate.performance = performance
                global_step += n_episodes

                logger.info(f"    Final: reward={performance['avg_reward']:.2f}, "
                           f"deliveries={performance['avg_deliveries']:.1f}")

                if use_wandb:
                    wandb.log({
                        "reward": performance["avg_reward"],
                        "deliveries": performance["avg_deliveries"],
                        "success_rate": performance["success_rate"],
                        "candidate_origin": candidate.origin,
                    }, step=global_step)

            # Save best policy state
            best_policy_state = policy.state_dict()

        # Select best candidates
        candidates_with_perf = [c for c in candidates if c.performance]
        candidates_with_perf.sort(key=lambda c: c.performance["avg_reward"], reverse=True)

        logger.info(f"\n  Rankings:")
        for i, c in enumerate(candidates_with_perf[:5]):
            logger.info(f"    {i+1}. {c.origin}: reward={c.performance['avg_reward']:.2f}")

        # Evolve for next stage
        if stage_idx < len(stage_configs) - 1:
            next_n = stage_configs[stage_idx + 1][0]
            top_candidates = candidates_with_perf[:max(2, next_n // 2)]

            insights = {
                "n_agents": n_agents,
                "env_state": env_state,
                "successful_patterns": f"Best performing: {top_candidates[0].origin}",
                "failure_patterns": "Lower ranked candidates had less efficient task ordering",
            }

            new_candidates = list(top_candidates)

            if len(top_candidates) >= 2:
                child = planner.crossover(
                    top_candidates[0], top_candidates[0].performance,
                    top_candidates[1], top_candidates[1].performance,
                    insights,
                )
                new_candidates.append(child)
                logger.info(f"  Created crossover child")

            for parent in top_candidates[:2]:
                mutant = planner.mutate(parent, parent.performance, insights)
                new_candidates.append(mutant)
                logger.info(f"  Created mutation of {parent.origin}")

            if len(new_candidates) < next_n:
                novel = planner.generate_novel(env_state, task, insights)
                new_candidates.append(novel)
                logger.info(f"  Created novel candidate")

            candidates = new_candidates[:next_n]

    # Final results
    best = max(candidates, key=lambda c: c.performance["avg_reward"] if c.performance else 0)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nBest candidate: {best.origin}")
    logger.info(f"  Reward: {best.performance['avg_reward']:.2f}")
    logger.info(f"  Deliveries: {best.performance['avg_deliveries']:.1f}")
    logger.info(f"  Success rate: {best.performance['success_rate']:.1%}")
    logger.info(f"\nBest graph:")
    logger.info(best.graph.to_prompt_string())

    if use_wandb:
        wandb.log({
            "final_reward": best.performance["avg_reward"],
            "final_deliveries": best.performance["avg_deliveries"],
            "final_success_rate": best.performance["success_rate"],
        })

        # Save artifacts
        import tempfile
        import os
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "policy.pt")
            torch.save({
                "model_state_dict": best_policy_state,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "n_agents": n_agents,
            }, policy_path)

            artifact = wandb.Artifact(
                name=f"policy-{layout}-parallel",
                type="model",
                description=f"Trained policy for {layout} (parallel training)",
                metadata={
                    "best_reward": best.performance["avg_reward"],
                    "best_deliveries": best.performance["avg_deliveries"],
                    "best_origin": best.origin,
                }
            )
            artifact.add_file(policy_path)
            wandb.log_artifact(artifact)

            graphs_path = os.path.join(tmpdir, "candidate_graphs.json")
            graphs_data = [
                {
                    "origin": c.origin,
                    "generation": c.generation,
                    "graph": c.graph.to_prompt_string(),
                    "performance": {k: v for k, v in c.performance.items() if k != "full_history"},
                }
                for c in candidates if c.performance
            ]
            with open(graphs_path, "w") as f:
                json.dump(graphs_data, f, indent=2)

            graphs_artifact = wandb.Artifact(name=f"graphs-{layout}-parallel", type="task_graphs")
            graphs_artifact.add_file(graphs_path)
            wandb.log_artifact(graphs_artifact)

        wandb.finish()

    ray.shutdown()
    return best, best_policy_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room", help="Overcooked layout")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model")
    parser.add_argument("--stages", type=int, default=7, help="Number of evolution stages")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_evolution_parallel(
        layout=args.layout,
        n_stages=args.stages,
        model=args.model,
        device=args.device,
        use_wandb=not args.no_wandb,
        n_parallel=args.parallel,
    )
