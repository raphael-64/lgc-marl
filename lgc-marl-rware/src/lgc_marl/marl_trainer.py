"""Multi-agent PPO trainer for graph-conditioned policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

# Support multiple environment types
# from ..environments.rware_wrapper import RWAREGraphWrapper
from ..graph_generation.graph_types import TaskGraph
from .graph_policy import GraphConditionedPolicy

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""

    observations: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    actions: Dict[int, List[int]] = field(default_factory=dict)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: Dict[int, List[float]] = field(default_factory=dict)
    dones: List[bool] = field(default_factory=list)

    # Episode-level stats
    total_reward: float = 0.0
    success: bool = False
    episode_length: int = 0

    def __post_init__(self):
        """Initialize per-agent storage."""
        pass

    def init_agents(self, n_agents: int):
        """Initialize storage for agents."""
        for i in range(n_agents):
            self.observations[i] = []
            self.actions[i] = []
            self.log_probs[i] = []

    def add(
        self,
        obs: List[np.ndarray],
        actions: List[int],
        reward: float,
        value: float,
        log_probs: List[float],
        done: bool,
    ):
        """Add a timestep to the buffer."""
        for i, (o, a, lp) in enumerate(zip(obs, actions, log_probs)):
            self.observations[i].append(o)
            self.actions[i].append(a)
            self.log_probs[i].append(lp)

        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.total_reward += reward
        self.episode_length += 1

    def clear(self):
        """Clear the buffer."""
        for i in self.observations:
            self.observations[i] = []
            self.actions[i] = []
            self.log_probs[i] = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.total_reward = 0.0
        self.success = False
        self.episode_length = 0


class MARLTrainer:
    """
    Multi-agent PPO trainer for graph-conditioned policies.

    Implements PPO with:
    - GAE for advantage estimation
    - Clipped surrogate objective
    - Entropy bonus for exploration
    - Shared value function
    """

    def __init__(
        self,
        env,  # Any multi-agent env with n_agents, step(), reset()
        policy: GraphConditionedPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            env: RWARE environment wrapper
            policy: Graph-conditioned policy network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_eps: PPO clipping epsilon
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of PPO epochs per update
            batch_size: Mini-batch size for PPO
            device: Torch device
        """
        self.env = env
        self.policy = policy
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Move policy to device
        self.policy = self.policy.to(self.device)

        # Optimizer
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Buffer
        self.buffer = RolloutBuffer()
        self.buffer.init_agents(env.n_agents)

        logger.info(f"MARLTrainer initialized on {self.device}")

    def train_episodes(
        self,
        graph: TaskGraph,
        n_episodes: int,
        steps_per_episode: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Train policy on graph for n_episodes.

        Args:
            graph: Task decomposition graph
            n_episodes: Number of episodes to train
            steps_per_episode: Max steps per episode (uses env default if None)

        Returns:
            List of training metrics per episode
        """
        trajectory = []

        for ep in range(n_episodes):
            # Collect rollout
            self._collect_rollout(graph, steps_per_episode)

            # Compute advantages and returns
            advantages, returns = self._compute_gae()

            # PPO update
            metrics = self._ppo_update(graph, advantages, returns)

            # Record episode stats
            ep_metrics = {
                "episode": ep,
                "reward": self.buffer.total_reward,
                "success": self.buffer.success,
                "episode_length": self.buffer.episode_length,
                **metrics,
            }
            trajectory.append(ep_metrics)

            # Clear buffer
            self.buffer.clear()
            self.buffer.init_agents(self.env.n_agents)

            # Logging
            if (ep + 1) % 10 == 0:
                recent_rewards = [t["reward"] for t in trajectory[-10:]]
                recent_success = [t["success"] for t in trajectory[-10:]]
                logger.debug(
                    f"Episode {ep + 1}/{n_episodes}: "
                    f"reward={np.mean(recent_rewards):.2f}, "
                    f"success={np.mean(recent_success):.1%}"
                )

        return trajectory

    def train_episode(self, graph: TaskGraph) -> Dict[str, Any]:
        """Train for a single episode and return metrics."""
        results = self.train_episodes(graph, n_episodes=1)
        return results[0] if results else {}

    def _collect_rollout(
        self, graph: TaskGraph, max_steps: Optional[int] = None
    ) -> None:
        """Collect rollout data from environment."""
        # Support both RWARE (takes graph) and Overcooked (no graph in reset)
        try:
            obs, info = self.env.reset(graph=graph)
        except TypeError:
            obs, info = self.env.reset()
        max_steps = max_steps or getattr(self.env, 'max_steps', getattr(self.env, 'horizon', 500))

        for step in range(max_steps):
            # Convert observations to tensors
            obs_tensors = [
                torch.tensor(obs[i], dtype=torch.float32, device=self.device)
                for i in range(self.env.n_agents)
            ]

            # Get actions and values
            with torch.no_grad():
                actions, log_probs = self.policy.get_actions(
                    obs_tensors, graph, self.device, deterministic=False
                )
                _, value = self.policy.forward(obs_tensors, graph, self.device)
                value = value.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(actions)

            # Handle reward - could be scalar (RWARE) or list (Overcooked)
            if isinstance(reward, (list, tuple)):
                reward = sum(reward)  # Sum agent rewards for shared reward

            # Store in buffer
            done = terminated or truncated
            self.buffer.add(
                obs=[obs[i] for i in range(self.env.n_agents)],
                actions=actions,
                reward=reward,
                value=value,
                log_probs=log_probs,
                done=done,
            )

            obs = next_obs

            if done:
                # Check for deliveries (works for both RWARE and Overcooked)
                episode_stats = info.get("episode_stats", {})
                deliveries = episode_stats.get("deliveries", info.get("deliveries", 0))
                self.buffer.success = deliveries > 0
                break

    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        gae = 0

        # Iterate backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _ppo_update(
        self, graph: TaskGraph, advantages: torch.Tensor, returns: torch.Tensor
    ) -> Dict[str, float]:
        """Perform PPO update."""
        n_steps = len(self.buffer.rewards)
        n_agents = self.env.n_agents

        # Prepare data
        all_obs = {
            i: torch.tensor(
                np.array(self.buffer.observations[i]), dtype=torch.float32, device=self.device
            )
            for i in range(n_agents)
        }
        all_actions = {
            i: torch.tensor(self.buffer.actions[i], dtype=torch.long, device=self.device)
            for i in range(n_agents)
        }
        all_old_log_probs = {
            i: torch.tensor(self.buffer.log_probs[i], dtype=torch.float32, device=self.device)
            for i in range(n_agents)
        }

        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        indices = np.arange(n_steps)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_steps, self.batch_size):
                end = min(start + self.batch_size, n_steps)
                batch_indices = indices[start:end]
                batch_size = len(batch_indices)

                # Get batch data
                batch_obs = [all_obs[i][batch_indices] for i in range(n_agents)]
                batch_actions = [all_actions[i][batch_indices] for i in range(n_agents)]
                batch_old_log_probs = [
                    all_old_log_probs[i][batch_indices] for i in range(n_agents)
                ]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                log_probs, value, entropies = self.policy.evaluate_actions(
                    batch_obs, batch_actions, graph, self.device
                )

                # Policy loss (clipped surrogate)
                policy_loss = 0
                entropy_loss = 0

                for i in range(n_agents):
                    ratio = torch.exp(log_probs[i].squeeze() - batch_old_log_probs[i])

                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * batch_advantages
                    )

                    policy_loss += -torch.min(surr1, surr2).mean()
                    entropy_loss += entropies[i].mean()

                policy_loss /= n_agents
                entropy_loss /= n_agents

                # Value loss
                value_loss = F.mse_loss(value.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        # Average metrics
        metrics = {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "total_loss": (total_policy_loss + total_value_loss) / max(n_updates, 1),
        }

        return metrics

    def evaluate(
        self,
        graph: TaskGraph,
        n_episodes: int = 20,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate policy on graph.

        Args:
            graph: Task decomposition graph
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions

        Returns:
            Evaluation metrics
        """
        successes = []
        total_rewards = []
        episode_lengths = []
        subtasks_completed = []

        for _ in range(n_episodes):
            obs, info = self.env.reset(graph=graph)
            total_reward = 0
            steps = 0

            while True:
                obs_tensors = [
                    torch.tensor(obs[i], dtype=torch.float32, device=self.device)
                    for i in range(self.env.n_agents)
                ]

                actions, _ = self.policy.get_actions(
                    obs_tensors, graph, self.device, deterministic=deterministic
                )
                obs, reward, terminated, truncated, info = self.env.step(actions)

                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            # Success = made at least one delivery
            episode_stats = info.get("episode_stats", {})
            deliveries = episode_stats.get("deliveries", 0)
            successes.append(float(deliveries > 0))
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            subtasks_completed.append(len(info.get("completed_subtasks", [])))

        return {
            "success_rate": np.mean(successes),
            "avg_reward": np.mean(total_rewards),
            "reward_std": np.std(total_rewards),
            "avg_episode_length": np.mean(episode_lengths),
            "avg_subtasks_completed": np.mean(subtasks_completed),
        }

    def analyze_failures(
        self,
        graph: TaskGraph,
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze what goes wrong with this graph.

        Args:
            graph: Task decomposition graph
            n_episodes: Number of episodes to analyze

        Returns:
            Failure analysis dict
        """
        failures = []
        coordination_issues = []
        incomplete_subtasks_all = []

        for _ in range(n_episodes):
            obs, info = self.env.reset(graph=graph)
            episode_collisions = 0

            while True:
                obs_tensors = [
                    torch.tensor(obs[i], dtype=torch.float32, device=self.device)
                    for i in range(self.env.n_agents)
                ]

                actions, _ = self.policy.get_actions(
                    obs_tensors, graph, self.device, deterministic=True
                )
                obs, reward, terminated, truncated, info = self.env.step(actions)

                # Track coordination issues
                if info.get("collision", False):
                    episode_collisions += 1
                    coordination_issues.append("collision")

                if terminated or truncated:
                    if not terminated:  # Failed (truncated)
                        completed = set(info.get("completed_subtasks", []))
                        incomplete = [
                            sid for sid in graph.subtasks.keys() if sid not in completed
                        ]
                        incomplete_subtasks_all.extend(incomplete)
                        failures.append(
                            {
                                "incomplete_subtasks": incomplete,
                                "collisions": episode_collisions,
                            }
                        )
                    break

        # Analyze failure patterns
        failure_points = {}
        for subtask_id in incomplete_subtasks_all:
            failure_points[subtask_id] = failure_points.get(subtask_id, 0) + 1

        # Find most common failure points
        common_failures = sorted(failure_points.items(), key=lambda x: x[1], reverse=True)[:5]

        # Identify bottleneck subtasks
        bottlenecks = []
        for subtask_id, count in common_failures:
            if count > n_episodes * 0.3:  # Failed in >30% of episodes
                bottlenecks.append(subtask_id)

        return {
            "common_failure_points": [f[0] for f in common_failures],
            "coordination_failures": list(set(coordination_issues)),
            "bottleneck_subtasks": bottlenecks,
            "failure_rate": len(failures) / n_episodes,
            "n_agents": self.env.n_agents,
        }
