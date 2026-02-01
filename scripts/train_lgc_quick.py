#!/usr/bin/env python3
"""Quick parallel LGC-MARL training for local visualization demo."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

import torch
import wandb
import ray

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.overcooked_wrapper import make_overcooked_env
from src.graph_generation.overcooked_planner import OvercookedGraphPlanner
from src.lgc_marl.marl_trainer import MARLTrainer
from src.lgc_marl.graph_policy import GraphConditionedPolicy
from src.graph_generation.graph_types import TaskGraph, Subtask, SubtaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2)
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
    """Train a single candidate in a Ray worker."""
    # Reconstruct graph
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

    # Create env and policy
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)
    policy = GraphConditionedPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        hidden_dim=128,
        graph_dim=32,
    )

    if policy_state_dict:
        policy.load_state_dict(policy_state_dict)

    trainer = MARLTrainer(env=env, policy=policy, device=device)

    metrics_history = []
    for ep in range(n_episodes):
        metrics = trainer.train_episode(graph)
        metrics_history.append(metrics)

        if (ep + 1) % 25 == 0:
            recent = metrics_history[-25:]
            avg_reward = sum(m["reward"] for m in recent) / len(recent)
            print(f"    [{graph_dict['origin']}] Ep {ep+1}/{n_episodes}: reward={avg_reward:.2f}")

    recent = metrics_history[-max(10, n_episodes // 5):]
    performance = {
        "avg_reward": sum(m["reward"] for m in recent) / len(recent),
        "avg_deliveries": sum(m.get("deliveries", 0) for m in recent) / len(recent),
        "success_rate": sum(1 for m in recent if m.get("deliveries", 0) > 0) / len(recent),
    }

    cpu_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    return performance, cpu_state_dict, graph_dict["origin"]


def graph_to_dict(graph: TaskGraph, origin: str) -> Dict:
    """Convert TaskGraph to serializable dict."""
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


def run_quick_parallel(
    layout: str = "cramped_room",
    model: str = "gpt-4o",
    device: str = "cpu",
    use_wandb: bool = True,
    n_parallel: int = 4,
):
    """Quick 3-stage parallel evolution for local demo."""

    # Compact stages - ~900 total episodes
    stage_configs = [
        (4, 75),    # Stage 1: 4 candidates × 75 ep (parallel)
        (2, 150),   # Stage 2: 2 candidates × 150 ep
        (1, 300),   # Stage 3: Champion × 300 ep
    ]

    # Init Ray locally
    if not ray.is_initialized():
        ray.init(num_cpus=n_parallel * 2)
        logger.info(f"Started local Ray with {n_parallel * 2} CPUs")

    logger.info("=" * 50)
    logger.info("LGC-MARL Quick Training (PARALLEL)")
    logger.info("=" * 50)
    logger.info(f"Layout: {layout}")
    logger.info(f"Device: {device}")
    logger.info(f"Workers: {n_parallel}")

    if use_wandb:
        wandb.init(
            project="lgc-marl-overcooked",
            config={"layout": layout, "stages": stage_configs, "quick": True, "parallel": True},
            tags=["quick", "local", "parallel"],
        )

    # Create env for dimensions
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)
    env_state = env.get_env_state()
    obs_dim = env._obs_shape_per_agent
    action_dim = env.n_actions
    n_agents = env.n_agents

    logger.info(f"Env: {layout}, obs={obs_dim}, actions={action_dim}")

    # Create planner
    planner = OvercookedGraphPlanner(model=model, temperature=0.7)
    task = "Make and serve onion soup"

    # Generate initial candidates
    logger.info(f"\nGenerating {stage_configs[0][0]} initial graphs...")
    candidates = planner.generate_initial_graphs(env_state, task, n_candidates=stage_configs[0][0])

    for c in candidates:
        logger.info(f"  {c.origin}: {len(c.graph)} subtasks")

    best_policy_state = None

    # Evolution loop
    for stage_idx, (n_candidates, n_episodes) in enumerate(stage_configs):
        logger.info(f"\n{'='*50}")
        logger.info(f"STAGE {stage_idx + 1}: {n_candidates} candidates × {n_episodes} episodes")
        logger.info("=" * 50)

        graph_dicts = [graph_to_dict(c.graph, c.origin) for c in candidates[:n_candidates]]

        # Launch all in parallel
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
        results = ray.get(futures)

        # Update candidates
        best_reward = -float('inf')
        for i, (perf, policy_state, origin) in enumerate(results):
            candidates[i].performance = perf
            logger.info(f"  {origin}: reward={perf['avg_reward']:.2f}, deliveries={perf['avg_deliveries']:.1f}")

            if perf['avg_reward'] > best_reward:
                best_reward = perf['avg_reward']
                best_policy_state = policy_state

            if use_wandb:
                wandb.log({"reward": perf["avg_reward"], "deliveries": perf["avg_deliveries"], "stage": stage_idx + 1})

        # Sort and select best
        candidates.sort(key=lambda c: c.performance["avg_reward"] if c.performance else -999, reverse=True)
        logger.info(f"\n  Best: {candidates[0].origin} (reward={candidates[0].performance['avg_reward']:.2f})")

        # Evolve for next stage
        if stage_idx < len(stage_configs) - 1:
            next_n = stage_configs[stage_idx + 1][0]
            top = candidates[:max(1, next_n)]
            insights = {"n_agents": n_agents, "env_state": env_state}
            new_candidates = list(top)

            if len(new_candidates) < next_n and len(top) > 0:
                mutant = planner.mutate(top[0], top[0].performance, insights)
                new_candidates.append(mutant)

            candidates = new_candidates[:next_n]

    # Final results
    best = candidates[0]
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Best: {best.origin}")
    logger.info(f"  Reward: {best.performance['avg_reward']:.2f}")
    logger.info(f"  Deliveries: {best.performance['avg_deliveries']:.1f}")

    # Save
    save_path = f"checkpoints/lgc_quick_{layout}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": best_policy_state,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_agents": n_agents,
    }, save_path)
    logger.info(f"\nSaved to {save_path}")

    if use_wandb:
        artifact = wandb.Artifact(f"lgc-quick-{layout}", type="model")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb.finish()

    ray.shutdown()
    return best, best_policy_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--model", default="gpt-4o", help="LLM for graph generation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    run_quick_parallel(
        layout=args.layout,
        model=args.model,
        device=args.device,
        use_wandb=not args.no_wandb,
        n_parallel=args.parallel,
    )
