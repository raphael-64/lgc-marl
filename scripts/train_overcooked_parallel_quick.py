#!/usr/bin/env python3
"""Scaled-down parallel training script for LGC-MARL on Overcooked using Ray."""

import logging
import os
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
from src.graph_generation.graph_types import GraphCandidate, TaskGraph, Subtask, SubtaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@ray.remote(
    num_cpus=2,  # CPU for env simulation
    runtime_env={
        "pip": ["torch", "gymnasium", "overcooked-ai", "networkx", "numpy"]
    }
)
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

        if (ep + 1) % 25 == 0:
            recent = metrics_history[-25:]
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

    # Move state dict to CPU for serialization back to head node
    cpu_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    return performance, cpu_state_dict, graph_dict["origin"]


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
    n_stages: int = 6,
    model: str = "gpt-5.2",
    device: str = "cpu",
    use_wandb: bool = True,
    n_parallel: int = 4,
):
    """Run progressive depth evolution with parallel candidate training (QUICK version)."""

    # Initialize Ray
    if not ray.is_initialized():
        if os.environ.get("ANYSCALE_SESSION_ID"):
            ray.init()
            logger.info("Connected to Anyscale cluster")
        else:
            ray.init(num_cpus=n_parallel * 2)
            logger.info(f"Started local Ray cluster with {n_parallel * 2} CPUs")

    # SCALED DOWN stage configs - ~6000 total episodes (vs ~28000 full)
    stage_configs = [
        (8, 75),    # Stage 1: Wide exploration - 600 ep
        (6, 150),   # Stage 2: Narrowing - 900 ep
        (4, 200),   # Stage 3: Top 4 - 800 ep
        (3, 300),   # Stage 4: Top 3 - 900 ep
        (2, 400),   # Stage 5: Final 2 - 800 ep
        (1, 2000),  # Stage 6: Champion polish - 2000 ep
    ][:n_stages]

    logger.info("=" * 60)
    logger.info("LGC-MARL Overcooked Training (PARALLEL - QUICK)")
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
                "quick": True,
            },
            tags=["parallel", "quick"],
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
    best_policy_state = None

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

        # Parallel if 3+ candidates
        run_parallel = (n_candidates >= 3)

        if run_parallel:
            logger.info(f"  Running {n_candidates} candidates in PARALLEL on workers")
        else:
            logger.info(f"  Running {n_candidates} candidates SEQUENTIALLY on workers")

        # Convert graphs to dicts for Ray serialization
        graph_dicts = [
            graph_to_dict(c.graph, c.origin)
            for c in candidates[:n_candidates]
        ]

        if run_parallel:
            # Launch all at once
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
        else:
            # Run one at a time on workers (sequential but still on workers)
            results = []
            seq_best_reward = -float('inf')
            for gd in graph_dicts:
                future = train_candidate_remote.remote(
                    graph_dict=gd,
                    n_episodes=n_episodes,
                    layout=layout,
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    n_agents=n_agents,
                    device=device,
                    policy_state_dict=best_policy_state,
                )
                result = ray.get(future)
                results.append(result)
                perf, policy_state, origin = result
                logger.info(f"    {origin}: reward={perf['avg_reward']:.2f}")
                if perf['avg_reward'] > seq_best_reward:
                    best_policy_state = policy_state
                    seq_best_reward = perf['avg_reward']

        # Update candidates with performance
        best_reward = -float('inf')
        for i, (perf, policy_state, origin) in enumerate(results):
            candidates[i].performance = perf
            logger.info(f"  {origin}: reward={perf['avg_reward']:.2f}, "
                       f"deliveries={perf['avg_deliveries']:.1f}")

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

    # Save locally
    save_path = f"checkpoints/lgc_parallel_quick_{layout}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": best_policy_state,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_agents": n_agents,
    }, save_path)
    logger.info(f"\nSaved to {save_path}")

    if use_wandb:
        wandb.log({
            "final_reward": best.performance["avg_reward"],
            "final_deliveries": best.performance["avg_deliveries"],
            "final_success_rate": best.performance["success_rate"],
        })

        # Save artifacts
        import tempfile
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
                name=f"policy-{layout}-parallel-quick",
                type="model",
                description=f"Trained policy for {layout} (quick parallel training)",
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

            graphs_artifact = wandb.Artifact(name=f"graphs-{layout}-parallel-quick", type="task_graphs")
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
    parser.add_argument("--stages", type=int, default=6, help="Number of evolution stages")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_evolution_parallel(
        layout=args.layout,
        n_stages=args.stages,
        model=args.model,
        device=args.device,
        use_wandb=not args.no_wandb,
        n_parallel=args.parallel,
    )
