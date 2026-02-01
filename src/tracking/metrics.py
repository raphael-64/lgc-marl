"""Metrics computation and logging utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("W&B not available. Install with: pip install wandb")


class MetricsLogger:
    """
    Metrics logger for W&B integration.

    Provides structured logging for:
    - Stage-level metrics
    - Candidate-level metrics
    - Training curves
    - Evolution statistics
    """

    def __init__(
        self,
        project: str = "lgc-marl-rware",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        """
        Initialize metrics logger.

        Args:
            project: W&B project name
            run_name: Run name (auto-generated if None)
            config: Experiment configuration
            enabled: Whether logging is enabled
        """
        self.project = project
        self.enabled = enabled and WANDB_AVAILABLE

        if self.enabled:
            try:
                if wandb.run is None:
                    wandb.init(
                        project=project,
                        name=run_name,
                        config=config or {},
                    )
                logger.info(f"W&B initialized: {project}/{run_name or 'auto'}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        if not self.enabled:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def log_stage_metrics(
        self,
        stage_idx: int,
        performances: List[Dict[str, Any]],
        best_idx: int,
    ) -> None:
        """
        Log stage-level metrics.

        Args:
            stage_idx: Stage index
            performances: Performance metrics per candidate
            best_idx: Index of best performer
        """
        success_rates = [p.get("success_rate", 0) for p in performances]
        rewards = [p.get("avg_reward", 0) for p in performances]

        metrics = {
            f"stage_{stage_idx}/n_candidates": len(performances),
            f"stage_{stage_idx}/best_success_rate": success_rates[best_idx],
            f"stage_{stage_idx}/mean_success_rate": np.mean(success_rates),
            f"stage_{stage_idx}/std_success_rate": np.std(success_rates),
            f"stage_{stage_idx}/best_reward": rewards[best_idx],
            f"stage_{stage_idx}/mean_reward": np.mean(rewards),
        }

        self.log(metrics)

    def log_candidate_metrics(
        self,
        stage_idx: int,
        candidate_idx: int,
        performance: Dict[str, Any],
        candidate_info: Dict[str, Any],
    ) -> None:
        """
        Log candidate-level metrics.

        Args:
            stage_idx: Stage index
            candidate_idx: Candidate index
            performance: Performance metrics
            candidate_info: Candidate metadata
        """
        prefix = f"stage_{stage_idx}/candidate_{candidate_idx}"

        metrics = {
            f"{prefix}/success_rate": performance.get("success_rate", 0),
            f"{prefix}/avg_reward": performance.get("avg_reward", 0),
            f"{prefix}/avg_episode_length": performance.get("avg_episode_length", 0),
            f"{prefix}/origin": candidate_info.get("origin", "unknown"),
        }

        self.log(metrics)

    def log_training_progress(
        self,
        stage_idx: int,
        candidate_idx: int,
        episode: int,
        reward: float,
        success: bool,
        loss: float,
    ) -> None:
        """
        Log training progress.

        Args:
            stage_idx: Stage index
            candidate_idx: Candidate index
            episode: Episode number
            reward: Episode reward
            success: Whether episode was successful
            loss: Training loss
        """
        prefix = f"train/s{stage_idx}_c{candidate_idx}"

        metrics = {
            f"{prefix}/episode": episode,
            f"{prefix}/reward": reward,
            f"{prefix}/success": int(success),
            f"{prefix}/loss": loss,
        }

        self.log(metrics)

    def log_evolution_metrics(
        self,
        from_stage: int,
        to_stage: int,
        n_elites: int,
        n_evolved: int,
        parent_success_rates: List[float],
    ) -> None:
        """
        Log evolution step metrics.

        Args:
            from_stage: Source stage
            to_stage: Target stage
            n_elites: Number of elites kept
            n_evolved: Number of evolved candidates
            parent_success_rates: Success rates of parents
        """
        metrics = {
            f"evolution/{from_stage}_to_{to_stage}/n_elites": n_elites,
            f"evolution/{from_stage}_to_{to_stage}/n_evolved": n_evolved,
            f"evolution/{from_stage}_to_{to_stage}/parent_mean_success": np.mean(
                parent_success_rates
            ),
        }

        self.log(metrics)

    def log_final_results(
        self,
        best_success_rate: float,
        best_reward: float,
        total_episodes: int,
        compute_savings: float,
    ) -> None:
        """
        Log final experiment results.

        Args:
            best_success_rate: Best success rate achieved
            best_reward: Best reward achieved
            total_episodes: Total training episodes
            compute_savings: Compute savings vs naive approach
        """
        metrics = {
            "final/best_success_rate": best_success_rate,
            "final/best_reward": best_reward,
            "final/total_episodes": total_episodes,
            "final/compute_savings": compute_savings,
        }

        self.log(metrics)

        # Log summary to W&B
        if self.enabled and wandb.run is not None:
            wandb.run.summary["best_success_rate"] = best_success_rate
            wandb.run.summary["best_reward"] = best_reward
            wandb.run.summary["total_episodes"] = total_episodes

    def finish(self) -> None:
        """Finish logging and close W&B run."""
        if self.enabled and wandb.run is not None:
            wandb.finish()


def compute_evolution_metrics(
    stage_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute aggregate metrics across evolution stages.

    Args:
        stage_results: List of stage result dictionaries

    Returns:
        Dictionary of aggregate metrics
    """
    if not stage_results:
        return {}

    # Extract performance trajectories
    best_per_stage = []
    mean_per_stage = []
    n_candidates_per_stage = []

    for stage in stage_results:
        performances = stage.get("performances", [])
        if performances:
            success_rates = [p.get("success_rate", 0) for p in performances]
            best_per_stage.append(max(success_rates))
            mean_per_stage.append(np.mean(success_rates))
            n_candidates_per_stage.append(len(performances))

    # Compute metrics
    metrics = {
        # Performance trajectory
        "best_per_stage": best_per_stage,
        "mean_per_stage": mean_per_stage,
        "n_candidates_per_stage": n_candidates_per_stage,
        # Improvement
        "initial_best": best_per_stage[0] if best_per_stage else 0,
        "final_best": best_per_stage[-1] if best_per_stage else 0,
        "improvement": (
            (best_per_stage[-1] - best_per_stage[0]) / max(best_per_stage[0], 0.01)
            if best_per_stage
            else 0
        ),
        # Efficiency
        "total_candidates_evaluated": sum(n_candidates_per_stage),
        "n_stages": len(stage_results),
    }

    return metrics


def compute_graph_metrics(graph_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute metrics for a task graph.

    Args:
        graph_dict: Graph dictionary with subtasks

    Returns:
        Dictionary of graph metrics
    """
    subtasks = graph_dict.get("subtasks", [])

    if not subtasks:
        return {"n_subtasks": 0}

    # Count by type
    type_counts = {}
    for s in subtasks:
        t = s.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Count by agent
    agent_counts = {}
    for s in subtasks:
        a = s.get("agent", 0)
        agent_counts[a] = agent_counts.get(a, 0) + 1

    # Compute dependency depth
    deps_per_subtask = [len(s.get("dependencies", [])) for s in subtasks]

    return {
        "n_subtasks": len(subtasks),
        "type_counts": type_counts,
        "agent_counts": agent_counts,
        "max_dependencies": max(deps_per_subtask) if deps_per_subtask else 0,
        "avg_dependencies": np.mean(deps_per_subtask) if deps_per_subtask else 0,
        "load_balance": (
            max(agent_counts.values()) - min(agent_counts.values())
            if agent_counts
            else 0
        ),
    }
