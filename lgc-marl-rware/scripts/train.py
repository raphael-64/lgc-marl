#!/usr/bin/env python3
"""Main training script for LGC-MARL Progressive Depth Evolution."""

import logging
import os
import sys
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.rware_wrapper import RWAREGraphWrapper, RWARETaskGenerator
from src.graph_generation.llm_planner import LLMGraphPlanner
from src.evolution.progressive_depth import ProgressiveDepthEvolution, Stage
from src.tracking.metrics import MetricsLogger, compute_evolution_metrics
from src.tracking.weave_ops import WeaveTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_device(cfg: DictConfig) -> str:
    """Setup compute device."""
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")
    return device


def create_stages(cfg: DictConfig) -> list[Stage]:
    """Create evolution stages from config."""
    return [
        Stage(candidates=s["candidates"], episodes=s["episodes"])
        for s in cfg.evolution.stages
    ]


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("=" * 60)
    logger.info("LGC-MARL Progressive Depth Evolution")
    logger.info("=" * 60)
    logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup device
    device = setup_device(cfg)

    # Initialize tracking
    weave_tracker = WeaveTracker(project=cfg.weave.project)
    metrics_logger = MetricsLogger(
        project=cfg.wandb.project,
        run_name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        # Create environment
        logger.info(f"\nCreating environment: {cfg.env.name}")
        env = RWAREGraphWrapper(
            env_name=cfg.env.name,
            max_steps=cfg.env.max_steps,
        )
        logger.info(f"  Grid size: {env.grid_size}")
        logger.info(f"  Agents: {env.n_agents}")
        logger.info(f"  Obs dim: {env.get_obs_dim()}")
        logger.info(f"  Action dim: {env.get_action_dim()}")

        # Create LLM planner
        logger.info(f"\nCreating LLM planner: {cfg.llm.model}")
        planner = LLMGraphPlanner(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            use_local=cfg.llm.use_local,
            max_retries=cfg.llm.max_retries,
        )

        # Create evolution stages
        stages = create_stages(cfg)
        total_episodes = sum(s.candidates * s.episodes for s in stages)
        logger.info(f"\nEvolution stages:")
        for i, stage in enumerate(stages):
            logger.info(f"  Stage {i + 1}: {stage.candidates} candidates × {stage.episodes} episodes")
        logger.info(f"  Total: {total_episodes} episodes")

        # Create evolution pipeline
        # Use actual dimensions from environment, not config
        actual_obs_dim = env.get_obs_dim()
        actual_action_dim = env.get_action_dim()

        evolution = ProgressiveDepthEvolution(
            env=env,
            planner=planner,
            stages=stages,
            obs_dim=actual_obs_dim,
            action_dim=actual_action_dim,
            policy_hidden_dim=cfg.policy.hidden_dim,
            policy_graph_dim=cfg.policy.graph_dim,
            trainer_lr=cfg.training.lr,
            device=device,
        )

        # Generate task
        task_generator = RWARETaskGenerator(n_agents=env.n_agents)
        task = task_generator.generate_task(difficulty=cfg.task.difficulty)
        logger.info(f"\nTask: {task}")

        # Run evolution
        logger.info("\nStarting evolution...")
        best_candidate, best_policy, history = evolution.run(
            task=task,
            wandb_log=metrics_logger.enabled,
        )

        # Compute and log final metrics
        compute_summary = evolution.get_compute_summary()
        evolution_metrics = compute_evolution_metrics(
            [
                {
                    "performances": [c.performance for c in sr.candidates if c.performance]
                }
                for sr in history
            ]
        )

        logger.info("\n" + "=" * 60)
        logger.info("EVOLUTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nBest Graph (success rate: {best_candidate.performance['success_rate']:.1%}):")
        logger.info(best_candidate.graph.to_prompt_string())
        logger.info(f"\nCompute Summary:")
        logger.info(f"  Total episodes: {compute_summary['total_episodes']}")
        logger.info(f"  Naive would be: {compute_summary['naive_episodes']}")
        logger.info(f"  Savings: {compute_summary['savings']:.1%}")

        # Log final results
        metrics_logger.log_final_results(
            best_success_rate=best_candidate.performance["success_rate"],
            best_reward=best_candidate.performance["avg_reward"],
            total_episodes=compute_summary["total_episodes"],
            compute_savings=compute_summary["savings"],
        )

        weave_tracker.log_final_results(
            best_candidate=best_candidate.to_dict(),
            best_performance=best_candidate.performance,
            compute_summary=compute_summary,
        )

        # Save results
        if cfg.checkpoint.save_best:
            checkpoint_dir = Path(cfg.checkpoint.save_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save policy
            policy_path = checkpoint_dir / "best_policy.pt"
            torch.save(best_policy.state_dict(), policy_path)
            logger.info(f"\nSaved best policy to: {policy_path}")

            # Save graph
            graph_path = checkpoint_dir / "best_graph.txt"
            with open(graph_path, "w") as f:
                f.write(f"Task: {task}\n\n")
                f.write(f"Success Rate: {best_candidate.performance['success_rate']:.1%}\n")
                f.write(f"Avg Reward: {best_candidate.performance['avg_reward']:.2f}\n\n")
                f.write("Graph:\n")
                f.write(best_candidate.graph.to_prompt_string())
            logger.info(f"Saved best graph to: {graph_path}")

            # Save full results as JSON
            import json

            results_path = checkpoint_dir / "results.json"
            results = {
                "task": task,
                "best_candidate": best_candidate.to_dict(),
                "compute_summary": compute_summary,
                "evolution_metrics": evolution_metrics,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved results to: {results_path}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        metrics_logger.finish()


if __name__ == "__main__":
    main()
