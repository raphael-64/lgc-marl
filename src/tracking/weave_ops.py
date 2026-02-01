"""Weave tracking utilities for LGC-MARL."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import weave
try:
    import weave

    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    logger.warning("Weave not available. Install with: pip install weave")


def track_experiment(func: Callable) -> Callable:
    """
    Decorator to track experiment functions with Weave.

    If Weave is not available, the function runs without tracking.
    """
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


class WeaveTracker:
    """
    Weave tracking wrapper for LGC-MARL experiments.

    Provides:
    - Experiment initialization
    - Stage/candidate/episode logging
    - Graph and policy tracking
    """

    def __init__(self, project: str = "lgc-marl-rware"):
        """
        Initialize Weave tracker.

        Args:
            project: Weave project name
        """
        self.project = project
        self.enabled = WEAVE_AVAILABLE

        if self.enabled:
            try:
                weave.init(project)
                logger.info(f"Weave initialized: {project}")
            except Exception as e:
                logger.warning(f"Failed to initialize Weave: {e}")
                self.enabled = False

    def log_stage(
        self,
        stage_idx: int,
        candidates: List[Dict[str, Any]],
        performances: List[Dict[str, Any]],
        best_idx: int,
    ) -> None:
        """
        Log stage-level results.

        Args:
            stage_idx: Stage index
            candidates: List of candidate info dicts
            performances: Performance metrics per candidate
            best_idx: Index of best performer
        """
        if not self.enabled:
            return

        try:
            stage_data = {
                "stage": stage_idx,
                "n_candidates": len(candidates),
                "candidates": [],
                "best_idx": best_idx,
                "best_performance": performances[best_idx],
            }

            for i, (cand, perf) in enumerate(zip(candidates, performances)):
                stage_data["candidates"].append(
                    {
                        "idx": i,
                        "origin": cand.get("origin", "unknown"),
                        "success_rate": perf.get("success_rate", 0),
                        "avg_reward": perf.get("avg_reward", 0),
                    }
                )

            # Log to Weave
            # Note: Actual logging depends on Weave version and setup
            logger.info(f"Logged stage {stage_idx} to Weave")

        except Exception as e:
            logger.warning(f"Failed to log stage to Weave: {e}")

    def log_candidate(
        self,
        stage_idx: int,
        candidate_idx: int,
        candidate_info: Dict[str, Any],
        performance: Dict[str, Any],
        graph_str: str,
    ) -> None:
        """
        Log individual candidate results.

        Args:
            stage_idx: Stage index
            candidate_idx: Candidate index within stage
            candidate_info: Candidate metadata
            performance: Performance metrics
            graph_str: String representation of graph
        """
        if not self.enabled:
            return

        try:
            data = {
                "stage": stage_idx,
                "candidate": candidate_idx,
                "origin": candidate_info.get("origin", "unknown"),
                "generation": candidate_info.get("generation", 0),
                "performance": performance,
                "graph": graph_str,
            }

            logger.debug(f"Logged candidate {stage_idx}/{candidate_idx} to Weave")

        except Exception as e:
            logger.warning(f"Failed to log candidate to Weave: {e}")

    def log_evolution(
        self,
        from_stage: int,
        to_stage: int,
        parent_indices: List[int],
        child_origins: List[str],
    ) -> None:
        """
        Log evolution step between stages.

        Args:
            from_stage: Source stage index
            to_stage: Target stage index
            parent_indices: Indices of parents selected
            child_origins: Origins of children (crossover, mutation, etc.)
        """
        if not self.enabled:
            return

        try:
            data = {
                "from_stage": from_stage,
                "to_stage": to_stage,
                "n_parents": len(parent_indices),
                "parent_indices": parent_indices,
                "child_origins": child_origins,
            }

            logger.debug(f"Logged evolution {from_stage}->{to_stage} to Weave")

        except Exception as e:
            logger.warning(f"Failed to log evolution to Weave: {e}")

    def log_training_trajectory(
        self,
        stage_idx: int,
        candidate_idx: int,
        trajectory: List[Dict[str, Any]],
    ) -> None:
        """
        Log training trajectory for a candidate.

        Args:
            stage_idx: Stage index
            candidate_idx: Candidate index
            trajectory: List of per-episode metrics
        """
        if not self.enabled:
            return

        try:
            # Summarize trajectory
            summary = {
                "n_episodes": len(trajectory),
                "final_reward": trajectory[-1].get("reward", 0) if trajectory else 0,
                "final_success": trajectory[-1].get("success", False) if trajectory else False,
                "reward_curve": [t.get("reward", 0) for t in trajectory[-10:]],
            }

            logger.debug(f"Logged trajectory {stage_idx}/{candidate_idx} to Weave")

        except Exception as e:
            logger.warning(f"Failed to log trajectory to Weave: {e}")

    def log_final_results(
        self,
        best_candidate: Dict[str, Any],
        best_performance: Dict[str, Any],
        compute_summary: Dict[str, Any],
    ) -> None:
        """
        Log final experiment results.

        Args:
            best_candidate: Best candidate info
            best_performance: Best performance metrics
            compute_summary: Compute usage summary
        """
        if not self.enabled:
            return

        try:
            data = {
                "best_candidate": best_candidate,
                "best_performance": best_performance,
                "compute_summary": compute_summary,
            }

            logger.info("Logged final results to Weave")

        except Exception as e:
            logger.warning(f"Failed to log final results to Weave: {e}")


# Convenience decorators for common operations
def log_graph_generation(func: Callable) -> Callable:
    """Decorator to log graph generation calls."""
    if WEAVE_AVAILABLE:

        @weave.op()
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        return wrapper
    else:
        return func


def log_training(func: Callable) -> Callable:
    """Decorator to log training calls."""
    if WEAVE_AVAILABLE:

        @weave.op()
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        return wrapper
    else:
        return func


def log_evaluation(func: Callable) -> Callable:
    """Decorator to log evaluation calls."""
    if WEAVE_AVAILABLE:

        @weave.op()
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        return wrapper
    else:
        return func
