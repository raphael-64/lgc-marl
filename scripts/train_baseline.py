#!/usr/bin/env python3
"""Baseline PPO training for Overcooked (no LLM graphs) for comparison."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.overcooked_wrapper import make_overcooked_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class BaselinePolicy(nn.Module):
    """Simple MLP policy without graph conditioning - same architecture size."""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int, hidden_dim: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-agent actor heads
        self.actors = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(n_agents)
        ])

        # Shared critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_list, device):
        """Forward pass for all agents."""
        features = []
        for i, obs in enumerate(obs_list):
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            feat = self.shared(obs)
            features.append(feat)

        # Actor outputs
        action_logits = [self.actors[i](features[i]) for i in range(self.n_agents)]

        # Critic output
        combined = torch.cat(features, dim=-1)
        value = self.critic(combined)

        return action_logits, value

    def get_actions(self, obs_list, device, deterministic=False):
        """Get actions for all agents."""
        action_logits, _ = self.forward(obs_list, device)

        actions = []
        log_probs = []

        for i, logits in enumerate(action_logits):
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = probs.argmax(dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

            log_prob = F.log_softmax(logits, dim=-1)
            log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

            actions.append(action.item() if action.dim() == 0 else action[0].item())
            log_probs.append(log_prob.item() if log_prob.dim() == 0 else log_prob[0].item())

        return actions, log_probs

    def evaluate_actions(self, obs_batch, actions_batch, device):
        """Evaluate actions for PPO update."""
        batch_size = obs_batch[0].shape[0]

        features = []
        for i, obs in enumerate(obs_batch):
            feat = self.shared(obs)
            features.append(feat)

        log_probs = []
        entropies = []

        for i in range(self.n_agents):
            logits = self.actors[i](features[i])
            probs = F.softmax(logits, dim=-1)
            log_prob_all = F.log_softmax(logits, dim=-1)

            action = actions_batch[i]
            log_prob = log_prob_all.gather(-1, action.unsqueeze(-1))
            entropy = -(probs * log_prob_all).sum(dim=-1)

            log_probs.append(log_prob)
            entropies.append(entropy)

        combined = torch.cat(features, dim=-1)
        value = self.critic(combined)

        return log_probs, value, entropies


class BaselineTrainer:
    """PPO trainer for baseline policy."""

    def __init__(
        self,
        env,
        policy: BaselinePolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.env = env
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def train_episode(self):
        """Train for one episode."""
        obs, info = self.env.reset()
        max_steps = getattr(self.env, 'horizon', 400)

        # Storage
        obs_buffer = {i: [] for i in range(self.env.n_agents)}
        actions_buffer = {i: [] for i in range(self.env.n_agents)}
        log_probs_buffer = {i: [] for i in range(self.env.n_agents)}
        rewards_buffer = []
        values_buffer = []
        dones_buffer = []

        total_reward = 0
        deliveries = 0

        for step in range(max_steps):
            obs_tensors = [
                torch.tensor(obs[i], dtype=torch.float32, device=self.device)
                for i in range(self.env.n_agents)
            ]

            with torch.no_grad():
                actions, log_probs = self.policy.get_actions(obs_tensors, self.device)
                _, value = self.policy.forward(obs_tensors, self.device)
                value = value.item()

            next_obs, reward, terminated, truncated, info = self.env.step(actions)

            if isinstance(reward, (list, tuple)):
                reward = sum(reward)

            # Store
            for i in range(self.env.n_agents):
                obs_buffer[i].append(obs[i])
                actions_buffer[i].append(actions[i])
                log_probs_buffer[i].append(log_probs[i])

            rewards_buffer.append(reward)
            values_buffer.append(value)
            dones_buffer.append(terminated or truncated)

            total_reward += reward
            obs = next_obs

            if terminated or truncated:
                deliveries = info.get("deliveries", 0)
                break

        # Compute GAE
        advantages, returns = self._compute_gae(rewards_buffer, values_buffer, dones_buffer)

        # PPO update
        metrics = self._ppo_update(obs_buffer, actions_buffer, log_probs_buffer, advantages, returns)

        return {
            "reward": total_reward,
            "deliveries": deliveries,
            "episode_length": len(rewards_buffer),
            **metrics,
        }

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=self.device)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _ppo_update(self, obs_buffer, actions_buffer, log_probs_buffer, advantages, returns):
        n_steps = len(advantages)
        n_agents = self.env.n_agents

        all_obs = {
            i: torch.tensor(np.array(obs_buffer[i]), dtype=torch.float32, device=self.device)
            for i in range(n_agents)
        }
        all_actions = {
            i: torch.tensor(actions_buffer[i], dtype=torch.long, device=self.device)
            for i in range(n_agents)
        }
        all_old_log_probs = {
            i: torch.tensor(log_probs_buffer[i], dtype=torch.float32, device=self.device)
            for i in range(n_agents)
        }

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

                batch_obs = [all_obs[i][batch_indices] for i in range(n_agents)]
                batch_actions = [all_actions[i][batch_indices] for i in range(n_agents)]
                batch_old_log_probs = [all_old_log_probs[i][batch_indices] for i in range(n_agents)]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                log_probs, value, entropies = self.policy.evaluate_actions(
                    batch_obs, batch_actions, self.device
                )

                policy_loss = 0
                entropy_loss = 0

                for i in range(n_agents):
                    ratio = torch.exp(log_probs[i].squeeze() - batch_old_log_probs[i])
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                    policy_loss += -torch.min(surr1, surr2).mean()
                    entropy_loss += entropies[i].mean()

                policy_loss /= n_agents
                entropy_loss /= n_agents
                value_loss = F.mse_loss(value.squeeze(), batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }


def run_baseline(
    layout: str = "cramped_room",
    total_episodes: int = 20800,  # Match LGC-MARL total: 8*200 + 6*400 + 5*600 + 4*800 + 3*1200 + 2*2000 + 1*3000
    device: str = "cpu",
    use_wandb: bool = True,
):
    """Run baseline PPO training."""

    logger.info("=" * 60)
    logger.info("BASELINE PPO Training (No LLM Graphs)")
    logger.info("=" * 60)
    logger.info(f"Layout: {layout}")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Device: {device}")

    if use_wandb:
        wandb.init(
            project="lgc-marl-overcooked",
            config={
                "layout": layout,
                "total_episodes": total_episodes,
                "method": "baseline_ppo",
            },
            tags=["baseline"],
        )

    # Create environment - same settings as LGC-MARL
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)

    logger.info(f"\nEnvironment:")
    logger.info(f"  Layout: {layout}")
    logger.info(f"  Grid: {env.mdp.width}x{env.mdp.height}")
    logger.info(f"  Obs dim: {env._obs_shape_per_agent}")
    logger.info(f"  Actions: {env.n_actions}")

    # Create policy - same hidden_dim as LGC-MARL
    policy = BaselinePolicy(
        obs_dim=env._obs_shape_per_agent,
        action_dim=env.n_actions,
        n_agents=env.n_agents,
        hidden_dim=128,
    )

    trainer = BaselineTrainer(env=env, policy=policy, device=device)

    logger.info(f"\nTraining for {total_episodes} episodes...")

    metrics_history = []
    for ep in range(total_episodes):
        metrics = trainer.train_episode()
        metrics_history.append(metrics)

        if use_wandb:
            wandb.log({
                "ep_reward": metrics["reward"],
                "ep_deliveries": metrics["deliveries"],
                "ep_length": metrics["episode_length"],
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
            }, step=ep)

        if (ep + 1) % 100 == 0:
            recent = metrics_history[-100:]
            avg_reward = sum(m["reward"] for m in recent) / len(recent)
            avg_deliveries = sum(m["deliveries"] for m in recent) / len(recent)
            success_rate = sum(1 for m in recent if m["deliveries"] > 0) / len(recent)
            logger.info(f"[Ep {ep+1}/{total_episodes}] reward={avg_reward:.2f}, "
                       f"deliveries={avg_deliveries:.1f}, success={success_rate:.1%}")

            if use_wandb:
                wandb.log({
                    "reward": avg_reward,
                    "deliveries": avg_deliveries,
                    "success_rate": success_rate,
                }, step=ep)

    # Final results
    final = metrics_history[-100:]
    final_reward = sum(m["reward"] for m in final) / len(final)
    final_deliveries = sum(m["deliveries"] for m in final) / len(final)
    final_success = sum(1 for m in final if m["deliveries"] > 0) / len(final)

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final reward: {final_reward:.2f}")
    logger.info(f"Final deliveries: {final_deliveries:.1f}")
    logger.info(f"Final success rate: {final_success:.1%}")

    if use_wandb:
        wandb.log({
            "final_reward": final_reward,
            "final_deliveries": final_deliveries,
            "final_success_rate": final_success,
        })

        # Save policy artifact
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "baseline_policy.pt")
            torch.save({
                "model_state_dict": policy.state_dict(),
                "obs_dim": env._obs_shape_per_agent,
                "action_dim": env.n_actions,
                "n_agents": env.n_agents,
            }, policy_path)

            artifact = wandb.Artifact(
                name=f"baseline-policy-{layout}",
                type="model",
                description=f"Baseline PPO policy for {layout}",
                metadata={
                    "final_reward": final_reward,
                    "final_deliveries": final_deliveries,
                    "final_success_rate": final_success,
                }
            )
            artifact.add_file(policy_path)
            wandb.log_artifact(artifact)
            logger.info("Saved baseline policy to wandb")

        wandb.finish()

    return policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room", help="Overcooked layout")
    parser.add_argument("--episodes", type=int, default=20800, help="Total training episodes")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    run_baseline(
        layout=args.layout,
        total_episodes=args.episodes,
        device=args.device,
        use_wandb=not args.no_wandb,
    )
