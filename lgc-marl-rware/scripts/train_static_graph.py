#!/usr/bin/env python3
"""Static graph baseline - LGC-MARL with ONE fixed graph, no evolution.

This ablation shows that the evolutionary self-improvement is valuable,
not just having any LLM-generated graph.

Uses the SAME graph generation as full LGC-MARL, just picks one and trains
without mutation/crossover/evolution.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import weave
import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.overcooked_wrapper import make_overcooked_env
from src.lgc_marl.marl_trainer import MARLTrainer
from src.lgc_marl.graph_policy import GraphConditionedPolicy
from src.graph_generation.graph_types import GraphCandidate
from src.graph_generation.overcooked_planner import OvercookedGraphPlanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_static_graph(
    layout: str = "cramped_room",
    total_episodes: int = 20800,
    model: str = "gpt-5.2",  # SAME model as full LGC-MARL
    strategy: str = "role_based",  # Pick ONE strategy, no evolution
    device: str = "cpu",
    use_wandb: bool = True,
):
    """Run training with a single static graph (no evolution)."""

    logger.info("=" * 60)
    logger.info("STATIC GRAPH Training (No Evolution)")
    logger.info("=" * 60)
    logger.info(f"Layout: {layout}")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Model: {model}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Device: {device}")

    if use_wandb:
        wandb.init(
            project="lgc-marl-overcooked",
            config={
                "layout": layout,
                "total_episodes": total_episodes,
                "method": "static_graph",
                "model": model,
                "strategy": strategy,
            },
            tags=["static_graph", "ablation"],
        )
        weave.init("lgc-marl-overcooked")

    # Create environment - SAME settings as full LGC-MARL
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)
    env_state = env.get_env_state()

    logger.info(f"\nEnvironment:")
    logger.info(f"  Layout: {layout}")
    logger.info(f"  Grid: {env.mdp.width}x{env.mdp.height}")

    # Use SAME planner as full LGC-MARL
    planner = OvercookedGraphPlanner(model=model, temperature=0.7)

    # Generate ONE initial graph using same method as LGC-MARL
    logger.info(f"\nGenerating graph with strategy={strategy} using {model}...")
    candidates = planner.generate_initial_graphs(env_state, "Make and serve onion soup", n_candidates=1)

    if not candidates:
        logger.error("Failed to generate graph!")
        return None, None

    graph = candidates[0].graph
    logger.info(f"Graph has {len(graph)} subtasks:")
    logger.info(graph.to_prompt_string())

    # Create policy
    policy = GraphConditionedPolicy(
        obs_dim=env._obs_shape_per_agent,
        action_dim=env.n_actions,
        n_agents=env.n_agents,
        hidden_dim=128,
        graph_dim=32,
    )

    trainer = MARLTrainer(env=env, policy=policy, device=device)

    logger.info(f"\nTraining for {total_episodes} episodes on static graph...")

    metrics_history = []
    for ep in range(total_episodes):
        metrics = trainer.train_episode(graph)
        metrics_history.append(metrics)

        if use_wandb:
            wandb.log({
                "ep_reward": metrics["reward"],
                "ep_deliveries": metrics.get("deliveries", 0),
                "ep_length": metrics.get("episode_length", 0),
                "policy_loss": metrics.get("policy_loss", 0),
                "value_loss": metrics.get("value_loss", 0),
                "entropy": metrics.get("entropy", 0),
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
    logger.info("STATIC GRAPH TRAINING COMPLETE")
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

        # Save artifacts
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save policy
            policy_path = os.path.join(tmpdir, "static_policy.pt")
            torch.save({
                "model_state_dict": policy.state_dict(),
                "obs_dim": env._obs_shape_per_agent,
                "action_dim": env.n_actions,
                "n_agents": env.n_agents,
            }, policy_path)

            artifact = wandb.Artifact(
                name=f"static-policy-{layout}",
                type="model",
                description=f"Static graph policy for {layout}",
                metadata={
                    "final_reward": final_reward,
                    "final_deliveries": final_deliveries,
                    "graph": graph.to_prompt_string(),
                }
            )
            artifact.add_file(policy_path)
            wandb.log_artifact(artifact)

        wandb.finish()

    return policy, graph


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room", help="Overcooked layout")
    parser.add_argument("--episodes", type=int, default=20800, help="Total training episodes")
    parser.add_argument("--model", default="gpt-5.2", help="LLM model (same as LGC-MARL)")
    parser.add_argument("--strategy", default="role_based", help="Graph strategy to use")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    run_static_graph(
        layout=args.layout,
        total_episodes=args.episodes,
        model=args.model,
        strategy=args.strategy,
        device=args.device,
        use_wandb=not args.no_wandb,
    )
