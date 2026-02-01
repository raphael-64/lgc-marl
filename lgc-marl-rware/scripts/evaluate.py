#!/usr/bin/env python3
"""Evaluation script for trained LGC-MARL policies."""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.rware_wrapper import RWAREGraphWrapper
from src.graph_generation.graph_types import TaskGraph
from src.lgc_marl.graph_policy import GraphConditionedPolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_policy(
    policy_path: Path,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    device: str = "cpu",
) -> GraphConditionedPolicy:
    """Load a trained policy from checkpoint."""
    policy = GraphConditionedPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
    )
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.to(device)
    policy.eval()
    return policy


def load_graph(graph_path: Path) -> TaskGraph:
    """Load a task graph from file."""
    # Try JSON first
    if graph_path.suffix == ".json":
        with open(graph_path) as f:
            data = json.load(f)
        return TaskGraph.from_dict(data)

    # Otherwise parse from text
    with open(graph_path) as f:
        content = f.read()

    # Find the graph section
    if "Graph:" in content:
        graph_section = content.split("Graph:")[-1].strip()
    else:
        graph_section = content

    # Parse simple format
    from src.graph_generation.graph_types import Subtask, SubtaskType

    graph = TaskGraph()
    for line in graph_section.split("\n"):
        line = line.strip()
        if not line or not line.startswith("-"):
            continue

        # Parse: "- subtask_id: Agent_X type target (after: deps)"
        import re

        match = re.match(
            r"-\s*(\w+):\s*Agent_(\d+)\s+(\w+)\s*(\S*)\s*(?:\(after:\s*([^)]+)\))?",
            line,
        )
        if match:
            subtask_id = match.group(1)
            agent_id = int(match.group(2))
            task_type = match.group(3).lower()
            target = match.group(4) if match.group(4) else None
            deps = [d.strip() for d in match.group(5).split(",")] if match.group(5) else []

            try:
                st_type = SubtaskType(task_type)
            except ValueError:
                st_type = SubtaskType.NAVIGATE

            graph.add_subtask(
                Subtask(
                    id=subtask_id,
                    task_type=st_type,
                    agent_id=agent_id,
                    target=target,
                    dependencies=deps,
                )
            )

    return graph


def evaluate(
    env: RWAREGraphWrapper,
    policy: GraphConditionedPolicy,
    graph: TaskGraph,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    device: str = "cpu",
) -> dict:
    """
    Evaluate policy on environment.

    Args:
        env: Environment
        policy: Trained policy
        graph: Task graph
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic actions
        render: Whether to render episodes
        device: Torch device

    Returns:
        Evaluation metrics
    """
    successes = []
    rewards = []
    episode_lengths = []
    subtasks_completed = []

    for ep in range(n_episodes):
        obs, info = env.reset(graph=graph)
        total_reward = 0
        steps = 0

        while True:
            if render:
                env.render()

            obs_tensors = [
                torch.tensor(obs[i], dtype=torch.float32, device=device)
                for i in range(env.n_agents)
            ]

            with torch.no_grad():
                actions, _ = policy.get_actions(
                    obs_tensors, graph, device, deterministic=deterministic
                )

            obs, reward, terminated, truncated, info = env.step(actions)
            total_reward += sum(reward) / len(reward)
            steps += 1

            if terminated or truncated:
                break

        successes.append(float(terminated))
        rewards.append(total_reward)
        episode_lengths.append(steps)
        subtasks_completed.append(len(info.get("completed_subtasks", [])))

        if (ep + 1) % 10 == 0:
            logger.info(
                f"Episode {ep + 1}/{n_episodes}: "
                f"success={np.mean(successes[-10:]):.1%}, "
                f"reward={np.mean(rewards[-10:]):.2f}"
            )

    return {
        "success_rate": np.mean(successes),
        "success_std": np.std(successes),
        "avg_reward": np.mean(rewards),
        "reward_std": np.std(rewards),
        "avg_episode_length": np.mean(episode_lengths),
        "length_std": np.std(episode_lengths),
        "avg_subtasks_completed": np.mean(subtasks_completed),
        "n_episodes": n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LGC-MARL policy")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory containing policy and graph files",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Path to policy checkpoint (overrides checkpoint-dir)",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="Path to graph file (overrides checkpoint-dir)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="rware-small-4ag-v2",
        help="Environment name",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda)",
    )
    args = parser.parse_args()

    # Resolve paths
    policy_path = args.policy_path or args.checkpoint_dir / "best_policy.pt"
    graph_path = args.graph_path or args.checkpoint_dir / "best_graph.txt"

    if not policy_path.exists():
        logger.error(f"Policy not found: {policy_path}")
        return 1
    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        return 1

    logger.info(f"Loading policy from: {policy_path}")
    logger.info(f"Loading graph from: {graph_path}")

    # Create environment
    env = RWAREGraphWrapper(env_name=args.env)
    logger.info(f"Environment: {args.env}, {env.n_agents} agents")

    # Load graph
    graph = load_graph(graph_path)
    logger.info(f"Graph: {len(graph.subtasks)} subtasks")
    logger.info(graph.to_prompt_string())

    # Load policy
    policy = load_policy(
        policy_path,
        obs_dim=env.get_obs_dim(),
        action_dim=env.get_action_dim(),
        n_agents=env.n_agents,
        device=args.device,
    )
    logger.info("Policy loaded")

    # Evaluate
    logger.info(f"\nEvaluating for {args.n_episodes} episodes...")
    results = evaluate(
        env=env,
        policy=policy,
        graph=graph,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        render=args.render,
        device=args.device,
    )

    # Print results
    logger.info("\n" + "=" * 40)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Success Rate: {results['success_rate']:.1%} (±{results['success_std']:.1%})")
    logger.info(f"Avg Reward: {results['avg_reward']:.2f} (±{results['reward_std']:.2f})")
    logger.info(f"Avg Episode Length: {results['avg_episode_length']:.1f} (±{results['length_std']:.1f})")
    logger.info(f"Avg Subtasks Completed: {results['avg_subtasks_completed']:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
