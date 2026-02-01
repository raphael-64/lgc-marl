#!/usr/bin/env python3
"""Training script for LGC-MARL on Overcooked environment."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.overcooked_wrapper import make_overcooked_env
from src.graph_generation.overcooked_planner import OvercookedGraphPlanner
from src.lgc_marl.marl_trainer import MARLTrainer
from src.lgc_marl.graph_policy import GraphConditionedPolicy
from src.graph_generation.graph_types import GraphCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_candidate(
    env,
    policy: GraphConditionedPolicy,
    candidate: GraphCandidate,
    n_episodes: int,
    device: str,
    lr: float = 3e-4,
) -> dict:
    """Train policy on a single graph candidate and return metrics."""
    trainer = MARLTrainer(
        env=env,
        policy=policy,
        lr=lr,
        device=device,
    )

    metrics_history = []
    for ep in range(n_episodes):
        metrics = trainer.train_episode(candidate.graph)
        metrics_history.append(metrics)

        if (ep + 1) % 10 == 0:
            recent = metrics_history[-10:]
            avg_reward = sum(m["reward"] for m in recent) / len(recent)
            avg_deliveries = sum(m.get("deliveries", 0) for m in recent) / len(recent)
            logger.info(f"    [Ep {ep+1}/{n_episodes}] reward={avg_reward:.2f}, deliveries={avg_deliveries:.1f}")

    # Compute final performance
    recent = metrics_history[-max(10, n_episodes // 5):]
    performance = {
        "avg_reward": sum(m["reward"] for m in recent) / len(recent),
        "avg_deliveries": sum(m.get("deliveries", 0) for m in recent) / len(recent),
        "success_rate": sum(1 for m in recent if m.get("deliveries", 0) > 0) / len(recent),
    }

    return performance


def run_evolution(
    layout: str = "cramped_room",
    n_stages: int = 3,
    model: str = "gpt-4o-mini",
    device: str = "cpu",
    use_wandb: bool = True,
):
    """Run progressive depth evolution on Overcooked."""

    # Stage configs: (n_candidates, n_episodes_per_candidate)
    stage_configs = [
        (6, 50),   # Stage 1: Wide exploration, shallow training
        (4, 100),  # Stage 2: Narrower, deeper
        (2, 200),  # Stage 3: Final candidates, full training
    ][:n_stages]

    logger.info("=" * 60)
    logger.info("LGC-MARL Overcooked Training")
    logger.info("=" * 60)
    logger.info(f"Layout: {layout}")
    logger.info(f"Model: {model}")
    logger.info(f"Device: {device}")
    logger.info(f"Stages: {stage_configs}")

    # Init wandb
    if use_wandb:
        wandb.init(
            project="lgc-marl-overcooked",
            config={
                "layout": layout,
                "model": model,
                "stages": stage_configs,
            },
        )

    # Create environment
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True)
    env_state = env.get_env_state()

    logger.info(f"\nEnvironment:")
    logger.info(f"  Layout: {layout}")
    logger.info(f"  Grid: {env.mdp.width}x{env.mdp.height}")
    logger.info(f"  Obs dim: {env._obs_shape_per_agent}")
    logger.info(f"  Actions: {env.n_actions}")

    # Create planner
    planner = OvercookedGraphPlanner(model=model, temperature=0.7)

    # Create initial shared policy
    policy = GraphConditionedPolicy(
        obs_dim=env._obs_shape_per_agent,
        action_dim=env.n_actions,
        n_agents=env.n_agents,
        hidden_dim=128,
        graph_dim=32,
    )

    task = "Make and serve onion soup"
    global_step = 0

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

        # Train each candidate
        for i, candidate in enumerate(candidates[:n_candidates]):
            logger.info(f"\n  Training candidate {i+1}/{n_candidates}: {candidate.origin}")
            logger.info(f"  Graph: {candidate.graph.to_prompt_string()[:200]}...")

            # Train
            performance = train_candidate(
                env=env,
                policy=policy,
                candidate=candidate,
                n_episodes=n_episodes,
                device=device,
            )

            candidate.performance = performance
            global_step += n_episodes

            logger.info(f"    Final: reward={performance['avg_reward']:.2f}, "
                       f"deliveries={performance['avg_deliveries']:.1f}, "
                       f"success={performance['success_rate']:.1%}")

            if use_wandb:
                wandb.log({
                    "reward": performance["avg_reward"],
                    "deliveries": performance["avg_deliveries"],
                    "success_rate": performance["success_rate"],
                    "candidate_origin": candidate.origin,
                }, step=global_step)

        # Select best candidates
        candidates_with_perf = [c for c in candidates if c.performance]
        candidates_with_perf.sort(key=lambda c: c.performance["avg_reward"], reverse=True)

        logger.info(f"\n  Rankings:")
        for i, c in enumerate(candidates_with_perf[:5]):
            logger.info(f"    {i+1}. {c.origin}: reward={c.performance['avg_reward']:.2f}")

        # Evolve for next stage (if not last)
        if stage_idx < len(stage_configs) - 1:
            next_n = stage_configs[stage_idx + 1][0]
            top_candidates = candidates_with_perf[:max(2, next_n // 2)]

            # Generate evolved candidates
            insights = {
                "n_agents": env.n_agents,
                "env_state": env_state,
                "successful_patterns": "High reward with role_based strategy",
                "failure_patterns": "Collisions when both chefs go same direction",
            }

            new_candidates = []

            # Keep top performers
            new_candidates.extend(top_candidates)

            # Crossover
            if len(top_candidates) >= 2:
                child = planner.crossover(
                    top_candidates[0],
                    top_candidates[0].performance,
                    top_candidates[1],
                    top_candidates[1].performance,
                    insights,
                )
                new_candidates.append(child)
                logger.info(f"  Created crossover child")

            # Mutation
            for parent in top_candidates[:2]:
                mutant = planner.mutate(parent, parent.performance, insights)
                new_candidates.append(mutant)
                logger.info(f"  Created mutation of {parent.origin}")

            # Novel
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

    if use_wandb:
        wandb.log({
            "final_reward": best.performance["avg_reward"],
            "final_deliveries": best.performance["avg_deliveries"],
            "final_success_rate": best.performance["success_rate"],
        })
        wandb.finish()

    return best, policy


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room", help="Overcooked layout")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--stages", type=int, default=3, help="Number of evolution stages")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    run_evolution(
        layout=args.layout,
        n_stages=args.stages,
        model=args.model,
        device=args.device,
        use_wandb=not args.no_wandb,
    )
