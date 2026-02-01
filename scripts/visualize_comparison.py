#!/usr/bin/env python3
"""Visualize trained policies side-by-side - pulls artifacts from wandb."""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import argparse
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.overcooked_wrapper import make_overcooked_env
from src.lgc_marl.graph_policy import GraphConditionedPolicy
from scripts.train_baseline import BaselinePolicy
from src.graph_generation.graph_types import TaskGraph, Subtask, SubtaskType

# Default wandb entity/project
WANDB_ENTITY = "raphael-64-university-of-waterloo"
WANDB_PROJECT = "lgc-marl-overcooked"


def download_from_wandb(artifact_name: str, entity: str = WANDB_ENTITY, project: str = WANDB_PROJECT):
    """Download artifact from wandb and return local path."""
    import wandb

    # Handle full artifact path or just name
    if "/" not in artifact_name:
        artifact_path = f"{entity}/{project}/{artifact_name}"
    else:
        artifact_path = artifact_name

    print(f"Downloading artifact: {artifact_path}")

    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    # Find the .pt file
    for f in os.listdir(artifact_dir):
        if f.endswith(".pt"):
            return os.path.join(artifact_dir, f)

    raise FileNotFoundError(f"No .pt file found in artifact {artifact_name}")


def load_policy(checkpoint_path: str, policy_type: str = "lgc"):
    """Load a trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]
    n_agents = checkpoint["n_agents"]

    if policy_type == "baseline":
        policy = BaselinePolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=128,
        )
    else:  # lgc or static
        policy = GraphConditionedPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            hidden_dim=128,
            graph_dim=32,
        )

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    return policy


def create_default_graph():
    """Create a default task graph for visualization."""
    graph = TaskGraph()
    tasks = [
        ("get1", SubtaskType.GET_INGREDIENT, 0, "onion", []),
        ("get2", SubtaskType.GET_INGREDIENT, 1, "onion", []),
        ("put1", SubtaskType.PUT_IN_POT, 0, "pot", ["get1"]),
        ("put2", SubtaskType.PUT_IN_POT, 1, "pot", ["get2"]),
        ("get3", SubtaskType.GET_INGREDIENT, 0, "onion", ["put1"]),
        ("put3", SubtaskType.PUT_IN_POT, 0, "pot", ["get3"]),
        ("wait", SubtaskType.WAIT_COOKING, 1, "pot", ["put2", "put3"]),
        ("dish", SubtaskType.GET_DISH, 0, "dish", ["wait"]),
        ("plate", SubtaskType.PLATE_SOUP, 0, "pot", ["dish"]),
        ("serve", SubtaskType.SERVE, 1, "serving", ["plate"]),
    ]
    for task_id, task_type, agent, target, deps in tasks:
        graph.add_subtask(Subtask(
            id=task_id, task_type=task_type, agent_id=agent,
            target=target, dependencies=deps,
        ))
    return graph


def run_episode_with_render(env, policy, graph=None, max_steps=200, device="cpu"):
    """Run an episode and collect rendered frames."""
    frames = []
    obs, info = env.reset()

    total_reward = 0
    deliveries = 0

    for step in range(max_steps):
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # Get actions
        obs_tensors = [
            torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
            for i in range(env.n_agents)
        ]

        with torch.no_grad():
            if graph is not None and hasattr(policy, 'get_actions'):
                # LGC policy - check signature
                try:
                    actions, _ = policy.get_actions(obs_tensors, graph, device)
                except TypeError:
                    actions, _ = policy.get_actions(obs_tensors, device)
            else:
                # Baseline policy
                actions, _ = policy.get_actions(obs_tensors, device)

        next_obs, reward, terminated, truncated, info = env.step(actions)

        if isinstance(reward, (list, tuple)):
            reward = sum(reward)
        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            deliveries = info.get("deliveries", 0)
            break

    # Final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    return frames, total_reward, deliveries


def save_gif(frames, output_path, fps=10):
    """Save frames as GIF."""
    if not frames:
        print("No frames to save")
        return

    # Convert numpy arrays to PIL Images
    images = [Image.fromarray(f) for f in frames]

    # Save as GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Saved GIF to {output_path}")


def save_mp4(frames, output_path, fps=10):
    """Save frames as MP4 (requires imageio-ffmpeg)."""
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved MP4 to {output_path}")
    except ImportError:
        print("Install imageio-ffmpeg for MP4: pip install imageio-ffmpeg")
        # Fallback to GIF
        save_gif(frames, output_path.replace(".mp4", ".gif"), fps)


def add_label_to_frames(frames, label, font_size=20):
    """Add text label to frames."""
    from PIL import ImageDraw, ImageFont

    labeled = []
    for frame in frames:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Add black outline + white text
        x, y = 10, 10
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((x+dx, y+dy), label, font=font, fill=(0, 0, 0))
        draw.text((x, y), label, font=font, fill=(255, 255, 255))

        labeled.append(np.array(img))

    return labeled


def create_side_by_side(frames_list, labels, output_path, fps=10):
    """Create side-by-side comparison GIF with labels below each video."""
    from PIL import ImageDraw, ImageFont

    if not all(frames_list):
        print("Missing frames for some policies")
        return

    # Ensure all have same number of frames (pad shorter ones)
    max_len = max(len(f) for f in frames_list)
    for frames in frames_list:
        while len(frames) < max_len:
            frames.append(frames[-1])  # Repeat last frame

    # Get frame dimensions
    frame_height, frame_width = frames_list[0][0].shape[:2]
    label_height = 40  # Height of label bar

    combined_frames = []
    for frame_idx in range(max_len):
        # Stack frames horizontally
        row_frames = [frames_list[i][frame_idx] for i in range(len(frames_list))]
        combined = np.hstack(row_frames)

        # Create label bar below
        total_width = combined.shape[1]
        label_bar = np.ones((label_height, total_width, 3), dtype=np.uint8) * 40  # Dark gray

        # Add labels centered under each frame
        label_img = Image.fromarray(label_bar)
        draw = ImageDraw.Draw(label_img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()

        for i, label in enumerate(labels):
            # Center text under each frame
            x_center = i * frame_width + frame_width // 2
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            x = x_center - text_width // 2
            y = (label_height - 24) // 2
            draw.text((x, y), label, font=font, fill=(255, 255, 255))

        label_bar = np.array(label_img)

        # Combine frame + label bar
        combined_with_label = np.vstack([combined, label_bar])
        combined_frames.append(combined_with_label)

    save_gif(combined_frames, output_path, fps)
    print(f"Saved side-by-side comparison to {output_path}")


def run_random_episode(env, max_steps=200):
    """Run random policy episode."""
    frames = []
    obs, _ = env.reset()
    total_reward = 0
    deliveries = 0

    for _ in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        actions = [env.action_space.sample() for _ in range(env.n_agents)]
        obs, r, term, trunc, info = env.step(actions)
        if isinstance(r, (list, tuple)):
            r = sum(r)
        total_reward += r
        if term or trunc:
            deliveries = info.get("deliveries", 0)
            break

    return frames, total_reward, deliveries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare trained Overcooked policies")

    # Wandb artifact names (easier than file paths)
    parser.add_argument("--baseline-artifact", type=str,
                        help="Wandb artifact name for baseline (e.g., 'baseline-policy-cramped_room:latest')")
    parser.add_argument("--static-artifact", type=str,
                        help="Wandb artifact name for static graph (e.g., 'static-policy-cramped_room:latest')")
    parser.add_argument("--lgc-artifact", type=str,
                        help="Wandb artifact name for LGC-MARL (e.g., 'policy-cramped_room-parallel:latest')")

    # Or local file paths
    parser.add_argument("--baseline", type=str, help="Local path to baseline checkpoint")
    parser.add_argument("--static", type=str, help="Local path to static graph checkpoint")
    parser.add_argument("--lgc", type=str, help="Local path to LGC-MARL checkpoint")

    # Wandb settings
    parser.add_argument("--entity", default=WANDB_ENTITY, help="Wandb entity")
    parser.add_argument("--project", default=WANDB_PROJECT, help="Wandb project")

    # Other options
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--output", default="comparison.gif")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--random", action="store_true", help="Include random baseline")
    parser.add_argument("--list-artifacts", action="store_true", help="List available artifacts and exit")

    args = parser.parse_args()

    # List artifacts mode
    if args.list_artifacts:
        import wandb
        api = wandb.Api()
        print(f"\nAvailable model artifacts in {args.entity}/{args.project}:\n")
        try:
            # Try newer API
            for artifact in api.artifact_collection(f"{args.entity}/{args.project}", "model").artifacts():
                print(f"  {artifact.name}")
        except:
            # Fallback - list runs and their artifacts
            runs = api.runs(f"{args.entity}/{args.project}")
            seen = set()
            for run in runs:
                for artifact in run.logged_artifacts():
                    if artifact.type == "model" and artifact.name not in seen:
                        print(f"  {artifact.name}")
                        seen.add(artifact.name)
        print("\nUse with: --baseline-artifact <name> --lgc-artifact <name> etc.")
        sys.exit(0)

    # Resolve artifact names to local paths
    baseline_path = args.baseline
    static_path = args.static
    lgc_path = args.lgc

    if args.baseline_artifact:
        baseline_path = download_from_wandb(args.baseline_artifact, args.entity, args.project)
    if args.static_artifact:
        static_path = download_from_wandb(args.static_artifact, args.entity, args.project)
    if args.lgc_artifact:
        lgc_path = download_from_wandb(args.lgc_artifact, args.entity, args.project)

    # Check if we have anything to visualize
    if not any([baseline_path, static_path, lgc_path, args.random]):
        print("No policies specified. Running random demo...")
        print("Use --list-artifacts to see available wandb artifacts")
        print("Or specify --baseline-artifact, --static-artifact, --lgc-artifact")
        env = make_overcooked_env(layout_name=args.layout, horizon=400, render_mode="rgb_array")
        frames, reward, deliveries = run_random_episode(env, args.max_steps)
        print(f"Random: reward={reward:.2f}, deliveries={deliveries}")
        save_gif(frames, "random_demo.gif", args.fps)
        sys.exit(0)

    # Collect frames from each policy
    graph = create_default_graph()
    all_frames = []
    labels = []

    if args.random:
        print("\n=== Random Policy ===")
        env = make_overcooked_env(layout_name=args.layout, horizon=400, render_mode="rgb_array")
        frames, reward, deliveries = run_random_episode(env, args.max_steps)
        print(f"Random: reward={reward:.2f}, deliveries={deliveries}")
        all_frames.append(frames)
        labels.append("Random")

    if baseline_path:
        print("\n=== Baseline PPO ===")
        env = make_overcooked_env(layout_name=args.layout, horizon=400, render_mode="rgb_array")
        policy = load_policy(baseline_path, "baseline")
        frames, reward, deliveries = run_episode_with_render(env, policy, None, args.max_steps)
        print(f"Baseline: reward={reward:.2f}, deliveries={deliveries}")
        all_frames.append(frames)
        labels.append("Baseline")

    if static_path:
        print("\n=== Static Graph ===")
        env = make_overcooked_env(layout_name=args.layout, horizon=400, render_mode="rgb_array")
        policy = load_policy(static_path, "static")
        frames, reward, deliveries = run_episode_with_render(env, policy, graph, args.max_steps)
        print(f"Static: reward={reward:.2f}, deliveries={deliveries}")
        all_frames.append(frames)
        labels.append("Static")

    if lgc_path:
        print("\n=== LGC-MARL ===")
        env = make_overcooked_env(layout_name=args.layout, horizon=400, render_mode="rgb_array")
        policy = load_policy(lgc_path, "lgc")
        frames, reward, deliveries = run_episode_with_render(env, policy, graph, args.max_steps)
        print(f"LGC-MARL: reward={reward:.2f}, deliveries={deliveries}")
        all_frames.append(frames)
        labels.append("LGC-MARL")

    # Create output
    if len(all_frames) > 1:
        create_side_by_side(all_frames, labels, args.output, args.fps)
    elif len(all_frames) == 1:
        frames = add_label_to_frames(all_frames[0], labels[0])
        save_gif(frames, args.output, args.fps)

    print(f"\nDone! Output saved to {args.output}")
