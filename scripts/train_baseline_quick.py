#!/usr/bin/env python3
"""Quick baseline PPO training - 2000 episodes for local testing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_baseline import run_baseline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_baseline(
        layout=args.layout,
        total_episodes=args.episodes,
        device=args.device,
        use_wandb=not args.no_wandb,
    )
