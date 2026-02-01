"""Policy transfer utilities for evolution between graphs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

from ..graph_generation.graph_types import GraphCandidate, TaskGraph
from ..lgc_marl.graph_policy import GraphConditionedPolicy, PolicyTransfer

logger = logging.getLogger(__name__)


class PolicyTransferManager:
    """
    Manages policy transfer during evolution.

    Handles:
    - Transferring weights from parent to child graphs
    - Maintaining a cache of trained policies
    - Selecting best source policy for transfer
    """

    def __init__(
        self,
        noise_scale: float = 0.01,
        cache_size: int = 20,
    ):
        """
        Initialize transfer manager.

        Args:
            noise_scale: Noise to add during transfer for exploration
            cache_size: Maximum number of policies to cache
        """
        self.noise_scale = noise_scale
        self.cache_size = cache_size

        # Cache: candidate_id -> (policy, performance)
        self.policy_cache: Dict[str, Tuple[GraphConditionedPolicy, Dict]] = {}

    def transfer_policy(
        self,
        source_policy: GraphConditionedPolicy,
        source_graph: TaskGraph,
        target_graph: TaskGraph,
        add_noise: bool = True,
    ) -> GraphConditionedPolicy:
        """
        Transfer policy from source to target graph.

        Args:
            source_policy: Trained source policy
            source_graph: Graph the source was trained on
            target_graph: New target graph
            add_noise: Whether to add noise for exploration

        Returns:
            New policy initialized from source
        """
        noise = self.noise_scale if add_noise else 0.0
        return PolicyTransfer.transfer(
            source_policy, source_graph, target_graph, noise_scale=noise
        )

    def get_best_source_policy(
        self,
        target_candidate: GraphCandidate,
        available_policies: List[Tuple[GraphCandidate, GraphConditionedPolicy, Dict]],
    ) -> Tuple[GraphConditionedPolicy, GraphCandidate]:
        """
        Find the best source policy for transfer.

        Considers:
        - Performance of source policy
        - Similarity between source and target graphs

        Args:
            target_candidate: Target graph candidate
            available_policies: List of (candidate, policy, performance) tuples

        Returns:
            Best source policy and its candidate
        """
        if not available_policies:
            raise ValueError("No source policies available")

        best_source = None
        best_score = -float("inf")
        best_candidate = None

        for candidate, policy, performance in available_policies:
            # Base score: performance
            perf_score = performance.get("success_rate", 0)

            # Similarity bonus
            similarity = self._compute_graph_similarity(
                candidate.graph, target_candidate.graph
            )

            # Combined score
            score = 0.7 * perf_score + 0.3 * similarity

            if score > best_score:
                best_score = score
                best_source = policy
                best_candidate = candidate

        return best_source, best_candidate

    def _compute_graph_similarity(
        self, graph1: TaskGraph, graph2: TaskGraph
    ) -> float:
        """
        Compute similarity between two graphs.

        Simple metrics:
        - Same number of subtasks
        - Similar parallelism score
        - Same agents used
        """
        if not graph1.subtasks or not graph2.subtasks:
            return 0.0

        # Size similarity
        size1 = len(graph1.subtasks)
        size2 = len(graph2.subtasks)
        size_sim = 1.0 - abs(size1 - size2) / max(size1, size2)

        # Parallelism similarity
        par1 = graph1.get_parallelism_score()
        par2 = graph2.get_parallelism_score()
        par_sim = 1.0 - abs(par1 - par2) / max(par1, par2, 1)

        # Agent usage similarity
        agents1 = set(s.agent_id for s in graph1.subtasks.values())
        agents2 = set(s.agent_id for s in graph2.subtasks.values())
        agent_sim = len(agents1 & agents2) / max(len(agents1 | agents2), 1)

        return (size_sim + par_sim + agent_sim) / 3

    def cache_policy(
        self,
        candidate: GraphCandidate,
        policy: GraphConditionedPolicy,
        performance: Dict,
    ) -> None:
        """
        Cache a trained policy.

        Args:
            candidate: Graph candidate
            policy: Trained policy
            performance: Performance metrics
        """
        # Evict if cache is full
        if len(self.policy_cache) >= self.cache_size:
            # Remove worst performer
            worst_id = min(
                self.policy_cache.keys(),
                key=lambda k: self.policy_cache[k][1].get("success_rate", 0)
            )
            del self.policy_cache[worst_id]

        self.policy_cache[candidate.candidate_id] = (policy.clone(), performance.copy())

    def get_cached_policy(
        self, candidate_id: str
    ) -> Optional[Tuple[GraphConditionedPolicy, Dict]]:
        """Get cached policy if available."""
        return self.policy_cache.get(candidate_id)

    def clear_cache(self) -> None:
        """Clear the policy cache."""
        self.policy_cache.clear()


def create_child_policy(
    parent_policies: List[Tuple[GraphConditionedPolicy, float]],
    target_graph: TaskGraph,
    noise_scale: float = 0.01,
) -> GraphConditionedPolicy:
    """
    Create child policy from parent policies.

    Supports:
    - Single parent: direct transfer
    - Multiple parents: weighted interpolation

    Args:
        parent_policies: List of (policy, weight) tuples
        target_graph: Target graph for the child
        noise_scale: Noise to add for exploration

    Returns:
        New child policy
    """
    if len(parent_policies) == 1:
        # Single parent: clone with noise
        parent, _ = parent_policies[0]
        child = parent.clone()

        if noise_scale > 0:
            with torch.no_grad():
                for param in child.parameters():
                    param.add_(torch.randn_like(param) * noise_scale)

        return child

    # Multiple parents: weighted average
    total_weight = sum(w for _, w in parent_policies)
    normalized_weights = [(p, w / total_weight) for p, w in parent_policies]

    # Start from first parent
    child = parent_policies[0][0].clone()

    with torch.no_grad():
        # Zero out parameters
        for param in child.parameters():
            param.zero_()

        # Weighted sum
        for parent, weight in normalized_weights:
            for child_param, parent_param in zip(
                child.parameters(), parent.parameters()
            ):
                child_param.add_(parent_param * weight)

        # Add noise
        if noise_scale > 0:
            for param in child.parameters():
                param.add_(torch.randn_like(param) * noise_scale)

    return child
