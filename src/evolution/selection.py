"""Selection strategies for evolutionary graph refinement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from ..graph_generation.graph_types import GraphCandidate
from ..lgc_marl.graph_policy import GraphConditionedPolicy


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""

    @abstractmethod
    def select(
        self,
        candidates: List[GraphCandidate],
        policies: List[GraphConditionedPolicy],
        performances: List[Dict[str, Any]],
        n_select: int,
    ) -> List[int]:
        """
        Select candidates for next generation.

        Args:
            candidates: List of graph candidates
            policies: Corresponding policies
            performances: Performance metrics for each candidate
            n_select: Number of candidates to select

        Returns:
            List of selected indices
        """
        pass


class ElitistSelection(SelectionStrategy):
    """
    Elitist selection: always pick the top performers.

    Simple but effective for exploitation.
    """

    def __init__(self, metric: str = "success_rate"):
        """
        Initialize elitist selection.

        Args:
            metric: Performance metric to use for ranking
        """
        self.metric = metric

    def select(
        self,
        candidates: List[GraphCandidate],
        policies: List[GraphConditionedPolicy],
        performances: List[Dict[str, Any]],
        n_select: int,
    ) -> List[int]:
        """Select top n_select candidates by metric."""
        scores = [p.get(self.metric, 0) for p in performances]
        ranked_indices = np.argsort(scores)[::-1]
        return list(ranked_indices[:n_select])


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection: randomly sample subsets and pick winners.

    Adds stochasticity while still favoring better candidates.
    """

    def __init__(
        self,
        tournament_size: int = 3,
        metric: str = "success_rate",
    ):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of candidates per tournament
            metric: Performance metric to use
        """
        self.tournament_size = tournament_size
        self.metric = metric

    def select(
        self,
        candidates: List[GraphCandidate],
        policies: List[GraphConditionedPolicy],
        performances: List[Dict[str, Any]],
        n_select: int,
    ) -> List[int]:
        """Select n_select candidates via tournament."""
        n_candidates = len(candidates)
        selected = []

        scores = [p.get(self.metric, 0) for p in performances]

        while len(selected) < n_select:
            # Random tournament
            tournament_indices = np.random.choice(
                n_candidates,
                size=min(self.tournament_size, n_candidates),
                replace=False
            )

            # Winner is the one with highest score
            tournament_scores = [scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]

            if winner_idx not in selected:
                selected.append(winner_idx)

        return selected


class DiversityAwareSelection(SelectionStrategy):
    """
    Selection that balances performance with diversity.

    Helps maintain exploration by not just picking the top performers.
    """

    def __init__(
        self,
        performance_weight: float = 0.7,
        diversity_weight: float = 0.3,
        metric: str = "success_rate",
    ):
        """
        Initialize diversity-aware selection.

        Args:
            performance_weight: Weight for performance score
            diversity_weight: Weight for diversity score
            metric: Performance metric to use
        """
        self.performance_weight = performance_weight
        self.diversity_weight = diversity_weight
        self.metric = metric

    def _compute_diversity(
        self,
        candidates: List[GraphCandidate],
        selected_indices: List[int],
        candidate_idx: int,
    ) -> float:
        """Compute diversity score for a candidate relative to selected set."""
        if not selected_indices:
            return 1.0

        candidate_graph = candidates[candidate_idx].graph

        # Simple diversity: different origin types
        origin_diversity = 1.0
        candidate_origin = candidates[candidate_idx].origin
        selected_origins = [candidates[i].origin for i in selected_indices]
        if candidate_origin in selected_origins:
            origin_diversity = 0.5

        # Structure diversity: different graph properties
        candidate_parallelism = candidate_graph.get_parallelism_score()
        selected_parallelisms = [
            candidates[i].graph.get_parallelism_score() for i in selected_indices
        ]

        avg_selected = np.mean(selected_parallelisms) if selected_parallelisms else 0
        structure_diversity = abs(candidate_parallelism - avg_selected)

        return 0.5 * origin_diversity + 0.5 * min(structure_diversity, 1.0)

    def select(
        self,
        candidates: List[GraphCandidate],
        policies: List[GraphConditionedPolicy],
        performances: List[Dict[str, Any]],
        n_select: int,
    ) -> List[int]:
        """Select balancing performance and diversity."""
        n_candidates = len(candidates)
        selected = []

        # Normalize performance scores
        scores = np.array([p.get(self.metric, 0) for p in performances])
        if scores.max() > scores.min():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            norm_scores = np.ones_like(scores)

        while len(selected) < n_select and len(selected) < n_candidates:
            best_idx = -1
            best_combined = -float("inf")

            for i in range(n_candidates):
                if i in selected:
                    continue

                perf_score = norm_scores[i]
                div_score = self._compute_diversity(candidates, selected, i)

                combined = (
                    self.performance_weight * perf_score
                    + self.diversity_weight * div_score
                )

                if combined > best_combined:
                    best_combined = combined
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
            else:
                break

        return selected


class UCBSelection(SelectionStrategy):
    """
    Upper Confidence Bound selection.

    Balances exploitation (high performers) with exploration (uncertain candidates).
    """

    def __init__(
        self,
        exploration_weight: float = 0.5,
        metric: str = "success_rate",
        uncertainty_metric: str = "reward_std",
    ):
        """
        Initialize UCB selection.

        Args:
            exploration_weight: Weight for uncertainty bonus
            metric: Primary performance metric
            uncertainty_metric: Metric for uncertainty (e.g., reward variance)
        """
        self.exploration_weight = exploration_weight
        self.metric = metric
        self.uncertainty_metric = uncertainty_metric

    def select(
        self,
        candidates: List[GraphCandidate],
        policies: List[GraphConditionedPolicy],
        performances: List[Dict[str, Any]],
        n_select: int,
    ) -> List[int]:
        """Select using UCB-style scoring."""
        scores = []

        for perf in performances:
            mean_score = perf.get(self.metric, 0)
            uncertainty = perf.get(self.uncertainty_metric, 0.1)

            # UCB score: mean + exploration_weight * uncertainty
            ucb = mean_score + self.exploration_weight * uncertainty
            scores.append(ucb)

        ranked_indices = np.argsort(scores)[::-1]
        return list(ranked_indices[:n_select])
