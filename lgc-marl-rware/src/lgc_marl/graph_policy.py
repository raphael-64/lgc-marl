"""Graph-conditioned multi-agent policy networks."""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_generation.graph_types import TaskGraph

logger = logging.getLogger(__name__)


class GraphEncoder(nn.Module):
    """
    Encode task graph into a fixed-size embedding vector.

    Uses a simple graph neural network approach:
    1. Embed each node (subtask) based on type and agent
    2. Apply message passing between connected nodes
    3. Global pooling to get graph-level embedding
    """

    def __init__(
        self,
        n_subtask_types: int = 8,
        n_agents: int = 8,
        node_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers: int = 2,
    ):
        """
        Initialize graph encoder.

        Args:
            n_subtask_types: Number of subtask type embeddings
            n_agents: Maximum number of agents
            node_dim: Dimension of node embeddings
            hidden_dim: Hidden dimension for message passing
            output_dim: Output embedding dimension
            n_layers: Number of message passing layers
        """
        super().__init__()

        self.output_dim = output_dim

        # Node embeddings
        self.type_embed = nn.Embedding(n_subtask_types, node_dim)
        self.agent_embed = nn.Embedding(n_agents, node_dim)

        # Message passing layers
        self.layers = nn.ModuleList()
        in_dim = node_dim * 2
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else output_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim + hidden_dim if i > 0 else in_dim, out_dim),
                    nn.ReLU(),
                    nn.LayerNorm(out_dim),
                )
            )
            in_dim = out_dim

        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, graph: TaskGraph, device: torch.device) -> torch.Tensor:
        """
        Encode graph to fixed-size vector.

        Args:
            graph: TaskGraph to encode
            device: Torch device

        Returns:
            Graph embedding tensor of shape [output_dim]
        """
        if not graph.subtasks:
            return torch.zeros(self.output_dim, device=device)

        # Build node features
        node_features = []
        node_ids = list(graph.subtasks.keys())
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        for subtask_id in node_ids:
            subtask = graph.subtasks[subtask_id]

            # Hash task type to embedding index
            type_idx = hash(subtask.task_type.value) % self.type_embed.num_embeddings
            type_idx = torch.tensor(type_idx, device=device)
            type_emb = self.type_embed(type_idx)

            # Agent embedding
            agent_idx = torch.tensor(
                subtask.agent_id % self.agent_embed.num_embeddings, device=device
            )
            agent_emb = self.agent_embed(agent_idx)

            # Concatenate embeddings
            node_feat = torch.cat([type_emb, agent_emb])
            node_features.append(node_feat)

        # Stack to tensor [n_nodes, node_dim*2]
        x = torch.stack(node_features)

        # Build adjacency info for message passing
        # Simple approach: aggregate neighbor features
        neighbors = {i: [] for i in range(len(node_ids))}
        for subtask_id in node_ids:
            subtask = graph.subtasks[subtask_id]
            idx = node_to_idx[subtask_id]
            for dep_id in subtask.dependencies:
                if dep_id in node_to_idx:
                    dep_idx = node_to_idx[dep_id]
                    neighbors[idx].append(dep_idx)
                    neighbors[dep_idx].append(idx)  # Bidirectional

        # Message passing
        for layer_idx, layer in enumerate(self.layers):
            # Aggregate neighbor features
            agg_features = []
            for i in range(len(node_ids)):
                if neighbors[i]:
                    neighbor_feats = x[neighbors[i]]
                    agg = neighbor_feats.mean(dim=0)
                else:
                    agg = torch.zeros_like(x[i])
                agg_features.append(agg)

            agg_tensor = torch.stack(agg_features)

            # Combine with own features
            if layer_idx == 0:
                # First layer: just use x (input dim = node_dim * 2)
                combined = x
            else:
                # Later layers: concatenate x with aggregated neighbor features
                combined = torch.cat([x, agg_tensor], dim=-1)

            x = layer(combined)

        # Global pooling (mean)
        graph_embed = x.mean(dim=0)

        # Output projection
        graph_embed = self.output_proj(graph_embed)

        return graph_embed


class GraphConditionedPolicy(nn.Module):
    """
    Multi-agent policy conditioned on task graph.

    Each agent receives:
    - Its own observation
    - The graph embedding (shared across agents)

    Architecture:
    - Graph encoder: TaskGraph -> embedding
    - Per-agent policy heads: (obs, graph_embed) -> action logits
    - Shared value head: (all_obs, graph_embed) -> value
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        graph_dim: int = 64,
        hidden_dim: int = 128,
        share_policy: bool = False,
    ):
        """
        Initialize policy network.

        Args:
            obs_dim: Observation dimension per agent
            action_dim: Action dimension (discrete actions)
            n_agents: Number of agents
            graph_dim: Graph embedding dimension
            hidden_dim: Hidden layer dimension
            share_policy: Whether to share policy weights across agents
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.share_policy = share_policy

        # Graph encoder
        self.graph_encoder = GraphEncoder(
            n_agents=n_agents,
            output_dim=graph_dim,
        )

        # Per-agent policy heads
        if share_policy:
            # Shared policy with agent ID embedding
            self.agent_embed = nn.Embedding(n_agents, 16)
            self.policy_net = nn.Sequential(
                nn.Linear(obs_dim + graph_dim + 16, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.agent_policies = None
        else:
            # Separate policy per agent
            self.agent_policies = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(obs_dim + graph_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim),
                    )
                    for _ in range(n_agents)
                ]
            )
            self.agent_embed = None
            self.policy_net = None

        # Shared value head
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim * n_agents + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observations: List[torch.Tensor],
        graph: TaskGraph,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            observations: List of [batch, obs_dim] tensors, one per agent
            graph: TaskGraph for conditioning
            device: Torch device

        Returns:
            action_logits: List of [batch, action_dim] tensors
            value: [batch, 1] tensor
        """
        batch_size = observations[0].size(0) if observations[0].dim() > 1 else 1

        # Encode graph (same for all agents in batch)
        graph_embed = self.graph_encoder(graph, device)  # [graph_dim]

        # Expand for batch
        if batch_size > 1:
            graph_embed_expanded = graph_embed.unsqueeze(0).expand(batch_size, -1)
        else:
            graph_embed_expanded = graph_embed.unsqueeze(0)

        # Per-agent action logits
        action_logits = []

        if self.share_policy:
            for i, obs in enumerate(observations):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                agent_id = torch.tensor([i], device=device).expand(obs.size(0))
                agent_emb = self.agent_embed(agent_id)

                combined = torch.cat([obs, graph_embed_expanded, agent_emb], dim=-1)
                logits = self.policy_net(combined)
                action_logits.append(logits)
        else:
            for i, (obs, policy) in enumerate(zip(observations, self.agent_policies)):
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                combined = torch.cat([obs, graph_embed_expanded], dim=-1)
                logits = policy(combined)
                action_logits.append(logits)

        # Value from joint observation + graph
        # Ensure all observations have batch dimension
        obs_for_value = []
        for obs in observations:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            obs_for_value.append(obs)

        joint_obs = torch.cat(obs_for_value, dim=-1)
        value_input = torch.cat([joint_obs, graph_embed_expanded], dim=-1)
        value = self.value_head(value_input)

        return action_logits, value

    def get_actions(
        self,
        observations: List[torch.Tensor],
        graph: TaskGraph,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """
        Get actions for all agents.

        Args:
            observations: List of observation tensors
            graph: TaskGraph
            device: Torch device
            deterministic: If True, take argmax instead of sampling

        Returns:
            actions: List of action indices
            log_probs: List of log probabilities
        """
        with torch.no_grad():
            action_logits, _ = self.forward(observations, graph, device)

        actions = []
        log_probs = []

        for logits in action_logits:
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                action = torch.multinomial(probs, 1)

            log_prob = F.log_softmax(logits, dim=-1)
            action_log_prob = log_prob.gather(-1, action.view(-1, 1))

            actions.append(action.item() if action.numel() == 1 else action.squeeze().item())
            log_probs.append(
                action_log_prob.item()
                if action_log_prob.numel() == 1
                else action_log_prob.squeeze().item()
            )

        return actions, log_probs

    def evaluate_actions(
        self,
        observations: List[torch.Tensor],
        actions: List[torch.Tensor],
        graph: TaskGraph,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Evaluate actions for PPO update.

        Args:
            observations: List of observation tensors
            actions: List of action tensors
            graph: TaskGraph
            device: Torch device

        Returns:
            log_probs: List of log probability tensors
            value: Value tensor
            entropy: List of entropy tensors
        """
        action_logits, value = self.forward(observations, graph, device)

        log_probs = []
        entropies = []

        for logits, action in zip(action_logits, actions):
            log_prob = F.log_softmax(logits, dim=-1)
            action_log_prob = log_prob.gather(-1, action.view(-1, 1))
            log_probs.append(action_log_prob)

            # Entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_prob).sum(dim=-1, keepdim=True)
            entropies.append(entropy)

        return log_probs, value, entropies

    def clone(self) -> GraphConditionedPolicy:
        """Create a deep copy of this policy."""
        return copy.deepcopy(self)


class PolicyTransfer:
    """Transfer policy weights between different graph structures."""

    @staticmethod
    def transfer(
        source_policy: GraphConditionedPolicy,
        source_graph: TaskGraph,
        target_graph: TaskGraph,
        noise_scale: float = 0.01,
    ) -> GraphConditionedPolicy:
        """
        Transfer learned policy to new graph structure.

        The key insight: graph encoder and agent policies are mostly reusable.
        The graph encoder handles any graph structure.
        Agent policies are per-agent, not per-graph.

        Args:
            source_policy: Trained policy
            source_graph: Graph the policy was trained on
            target_graph: New graph to adapt to
            noise_scale: Scale of noise to add for exploration

        Returns:
            New policy initialized from source
        """
        # Clone the source policy
        target_policy = source_policy.clone()

        # Add small noise to encourage exploration with new graph
        if noise_scale > 0:
            with torch.no_grad():
                for param in target_policy.parameters():
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)

        return target_policy

    @staticmethod
    def interpolate(
        policy1: GraphConditionedPolicy,
        policy2: GraphConditionedPolicy,
        alpha: float = 0.5,
    ) -> GraphConditionedPolicy:
        """
        Interpolate between two policies.

        Args:
            policy1: First policy
            policy2: Second policy
            alpha: Interpolation weight (0 = policy1, 1 = policy2)

        Returns:
            Interpolated policy
        """
        result = policy1.clone()

        with torch.no_grad():
            for p_result, p1, p2 in zip(
                result.parameters(), policy1.parameters(), policy2.parameters()
            ):
                p_result.data = (1 - alpha) * p1.data + alpha * p2.data

        return result
