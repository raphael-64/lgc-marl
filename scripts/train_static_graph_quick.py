#!/usr/bin/env python3
"""Quick static graph training - 2000 episodes for local testing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_static_graph import run_static_graph

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_static_graph(
        layout=args.layout,
        total_episodes=args.episodes,
        use_llm_graph=args.use_llm,
        model=args.model,
        device=args.device,
        use_wandb=not args.no_wandb,
    )
