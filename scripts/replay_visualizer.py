#!/usr/bin/env python3
"""
Interactive replay visualizer for trained Overcooked policies.

Controls:
    SPACE     - Play/Pause
    R         - Reset (run new episode)
    RIGHT     - Step forward (when paused)
    LEFT      - Step backward (when paused)
    UP/DOWN   - Adjust playback speed
    L         - Toggle loop mode (auto-restart)
    D         - Toggle deterministic/stochastic actions
    G         - Toggle graph overlay
    S         - Save current episode as GIF
    Q/ESC     - Quit

Runs FULL episodes (400 steps) to show complete policy behavior.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pygame
from PIL import Image

from src.environments.overcooked_wrapper import make_overcooked_env
from src.lgc_marl.graph_policy import GraphConditionedPolicy
from src.graph_generation.graph_types import TaskGraph, Subtask, SubtaskType


class ReplayVisualizer:
    """Interactive visualizer for trained policies."""

    def __init__(
        self,
        policy: GraphConditionedPolicy,
        graph: TaskGraph,
        layout: str = "cramped_room",
        device: str = "cpu",
        scale: float = 1.5,
    ):
        self.policy = policy
        self.graph = graph
        self.layout = layout
        self.device = device
        self.scale = scale

        # Create environment
        self.env = make_overcooked_env(
            layout_name=layout,
            horizon=400,
            render_mode="rgb_array",
        )

        # Playback state
        self.playing = False
        self.deterministic = True
        self.show_graph = True
        self.loop_mode = True  # Auto-restart when episode ends
        self.fps = 10
        self.frame_idx = 0

        # Episode data
        self.frames: List[np.ndarray] = []
        self.actions_history: List[List[int]] = []
        self.rewards_history: List[float] = []  # Per-step rewards
        self.cumulative_rewards: List[float] = []  # Running total at each step
        self.deliveries_history: List[int] = []  # Deliveries at each step
        self.infos_history: List[Dict[str, Any]] = []
        self.obs_history: List[List[np.ndarray]] = []

        # Stats
        self.final_reward = 0.0
        self.final_deliveries = 0
        self.episode_count = 0

        # Initialize pygame
        pygame.init()
        pygame.font.init()

        # Get initial frame to determine size
        self._reset_episode()
        if self.frames:
            h, w = self.frames[0].shape[:2]
        else:
            h, w = 300, 300

        self.frame_w = int(w * scale)
        self.frame_h = int(h * scale)
        self.sidebar_w = 280
        self.window_w = self.frame_w + self.sidebar_w
        self.window_h = self.frame_h + 60  # Extra for controls bar

        self.screen = pygame.display.set_mode((self.window_w, self.window_h))
        pygame.display.set_caption(f"LGC-MARL Replay - {layout}")

        # Fonts
        self.font_large = pygame.font.SysFont("Monaco", 18)
        self.font_small = pygame.font.SysFont("Monaco", 14)
        self.font_title = pygame.font.SysFont("Monaco", 22, bold=True)

        # Colors
        self.bg_color = (30, 30, 35)
        self.panel_color = (45, 45, 50)
        self.text_color = (220, 220, 220)
        self.accent_color = (100, 180, 255)
        self.success_color = (100, 220, 100)
        self.warn_color = (255, 180, 80)

    def _reset_episode(self):
        """Reset and run a fresh episode."""
        self.frames = []
        self.actions_history = []
        self.rewards_history = []
        self.cumulative_rewards = []
        self.deliveries_history = []
        self.infos_history = []
        self.obs_history = []
        self.frame_idx = 0

        obs, info = self.env.reset()
        self.obs_history.append(obs)

        # Capture initial frame
        frame = self.env.render()
        if frame is not None:
            self.frames.append(frame)
        self.infos_history.append(info)
        self.cumulative_rewards.append(0.0)  # Starting cumulative
        self.deliveries_history.append(info.get("deliveries", 0))

        # Run FULL episode (entire horizon - 400 steps default)
        done = False
        cumulative = 0.0
        while not done:
            # Get actions from policy
            obs_tensors = [
                torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                for i in range(self.env.n_agents)
            ]

            with torch.no_grad():
                actions, _ = self.policy.get_actions(
                    obs_tensors, self.graph, self.device, deterministic=self.deterministic
                )

            self.actions_history.append(actions)

            # Step
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated

            reward = sum(rewards) if isinstance(rewards, (list, tuple)) else rewards
            cumulative += reward
            self.rewards_history.append(reward)
            self.cumulative_rewards.append(cumulative)
            self.deliveries_history.append(info.get("deliveries", 0))

            # Capture frame
            frame = self.env.render()
            if frame is not None:
                self.frames.append(frame)

            self.infos_history.append(info)
            obs = next_obs
            self.obs_history.append(obs)

        self.final_reward = cumulative
        self.final_deliveries = self.deliveries_history[-1] if self.deliveries_history else 0
        self.episode_count += 1

    def _draw_frame(self):
        """Draw the current game frame."""
        if not self.frames or self.frame_idx >= len(self.frames):
            return

        frame = self.frames[self.frame_idx]

        # Scale frame
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled = pygame.transform.scale(surface, (self.frame_w, self.frame_h))

        self.screen.blit(scaled, (0, 0))

    def _draw_sidebar(self):
        """Draw stats sidebar."""
        x = self.frame_w
        y = 0

        # Background
        pygame.draw.rect(self.screen, self.panel_color, (x, y, self.sidebar_w, self.frame_h))

        # Title
        title = self.font_title.render("Episode Stats", True, self.accent_color)
        self.screen.blit(title, (x + 15, y + 15))

        y_offset = 55

        # Get current frame stats (not final totals)
        current_reward = self.cumulative_rewards[self.frame_idx] if self.frame_idx < len(self.cumulative_rewards) else 0
        current_deliveries = self.deliveries_history[self.frame_idx] if self.frame_idx < len(self.deliveries_history) else 0

        # Episode info - showing CURRENT values that update during playback
        stats = [
            ("Episode", f"#{self.episode_count}"),
            ("Frame", f"{self.frame_idx + 1}/{len(self.frames)}"),
            ("", ""),
            ("Reward", f"{current_reward:.1f}"),
            ("Deliveries", f"{current_deliveries}"),
            ("Final", f"{self.final_reward:.1f} / {self.final_deliveries}d"),
            ("", ""),
            ("Speed", f"{self.fps} FPS"),
            ("Mode", "Deterministic" if self.deterministic else "Stochastic"),
            ("Loop", "ON" if self.loop_mode else "OFF"),
            ("Layout", self.layout),
        ]

        for label, value in stats:
            if label:
                label_surf = self.font_small.render(f"{label}:", True, (150, 150, 150))
                self.screen.blit(label_surf, (x + 15, y + y_offset))

                # Highlight deliveries when they happen
                color = self.text_color
                if label == "Deliveries" and current_deliveries > 0:
                    color = self.success_color
                elif label == "Final":
                    color = (150, 150, 150)  # Dimmer for final stats
                value_surf = self.font_large.render(str(value), True, color)
                self.screen.blit(value_surf, (x + 120, y + y_offset - 2))

            y_offset += 25

        # Current action (if available) - actions taken FROM this frame
        if self.frame_idx < len(self.actions_history):
            y_offset += 10
            action_label = self.font_small.render("Next Action:", True, (150, 150, 150))
            self.screen.blit(action_label, (x + 15, y + y_offset))
            y_offset += 22

            actions = self.actions_history[self.frame_idx]
            action_names = ["N", "S", "E", "W", "Stay", "Interact"]
            for i, a in enumerate(actions):
                action_str = action_names[a] if a < len(action_names) else str(a)
                agent_surf = self.font_small.render(f"  Agent {i}: {action_str}", True, self.text_color)
                self.screen.blit(agent_surf, (x + 15, y + y_offset))
                y_offset += 20

        # Current step reward (rewards_history is offset by 1 since frame 0 is initial state)
        reward_idx = self.frame_idx - 1  # Frame 1 corresponds to reward from step 0
        if reward_idx >= 0 and reward_idx < len(self.rewards_history):
            y_offset += 15
            reward = self.rewards_history[reward_idx]
            # Highlight big rewards (deliveries give +20 sparse)
            if reward > 5:
                reward_color = self.success_color
            elif reward > 0:
                reward_color = self.warn_color
            else:
                reward_color = self.text_color
            reward_surf = self.font_small.render(f"Step reward: {reward:.2f}", True, reward_color)
            self.screen.blit(reward_surf, (x + 15, y + y_offset))

        # Graph info (if enabled)
        if self.show_graph and self.graph:
            y_offset += 35
            graph_label = self.font_small.render("Task Graph:", True, self.accent_color)
            self.screen.blit(graph_label, (x + 15, y + y_offset))
            y_offset += 22

            for i, (st_id, st) in enumerate(list(self.graph.subtasks.items())[:6]):
                task_str = f"{st_id[:8]}: A{st.agent_id}"
                task_surf = self.font_small.render(task_str, True, (180, 180, 180))
                self.screen.blit(task_surf, (x + 20, y + y_offset))
                y_offset += 18

            if len(self.graph.subtasks) > 6:
                more_surf = self.font_small.render(f"  +{len(self.graph.subtasks) - 6} more...", True, (120, 120, 120))
                self.screen.blit(more_surf, (x + 20, y + y_offset))

    def _draw_controls(self):
        """Draw controls bar at bottom."""
        y = self.frame_h
        pygame.draw.rect(self.screen, (40, 40, 45), (0, y, self.window_w, 60))

        # Progress bar
        bar_x, bar_y = 15, y + 10
        bar_w, bar_h = self.window_w - 30, 8
        pygame.draw.rect(self.screen, (60, 60, 65), (bar_x, bar_y, bar_w, bar_h), border_radius=4)

        if self.frames:
            progress = (self.frame_idx + 1) / len(self.frames)
            pygame.draw.rect(
                self.screen, self.accent_color,
                (bar_x, bar_y, int(bar_w * progress), bar_h),
                border_radius=4
            )

        # Controls text
        status = "PLAYING" if self.playing else "PAUSED"
        status_color = self.success_color if self.playing else self.warn_color
        status_surf = self.font_large.render(status, True, status_color)
        self.screen.blit(status_surf, (15, y + 30))

        # Keybindings hint
        hints = "SPACE: Play | R: Reset | Arrows: Step/Speed | L: Loop | S: Save | Q: Quit"
        hint_surf = self.font_small.render(hints, True, (120, 120, 120))
        self.screen.blit(hint_surf, (120, y + 33))

    def _save_gif(self):
        """Save current episode as GIF."""
        if not self.frames:
            return

        output_path = f"replay_{self.layout}_ep{self.episode_count}.gif"
        images = [Image.fromarray(f) for f in self.frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000 // self.fps,
            loop=0,
        )
        print(f"Saved GIF to {output_path}")

    def run(self):
        """Main visualization loop."""
        clock = pygame.time.Clock()
        running = True
        last_advance = 0

        while running:
            current_time = pygame.time.get_ticks()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False

                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing

                    elif event.key == pygame.K_r:
                        self._reset_episode()

                    elif event.key == pygame.K_RIGHT and not self.playing:
                        if self.frame_idx < len(self.frames) - 1:
                            self.frame_idx += 1

                    elif event.key == pygame.K_LEFT and not self.playing:
                        if self.frame_idx > 0:
                            self.frame_idx -= 1

                    elif event.key == pygame.K_UP:
                        self.fps = min(60, self.fps + 2)

                    elif event.key == pygame.K_DOWN:
                        self.fps = max(1, self.fps - 2)

                    elif event.key == pygame.K_d:
                        self.deterministic = not self.deterministic
                        self._reset_episode()

                    elif event.key == pygame.K_g:
                        self.show_graph = not self.show_graph

                    elif event.key == pygame.K_l:
                        self.loop_mode = not self.loop_mode

                    elif event.key == pygame.K_s:
                        self._save_gif()

            # Auto-advance when playing
            if self.playing and self.frames:
                if current_time - last_advance > 1000 / self.fps:
                    last_advance = current_time
                    if self.frame_idx < len(self.frames) - 1:
                        self.frame_idx += 1
                    else:
                        # Episode ended
                        if self.loop_mode:
                            # Run a new episode
                            self._reset_episode()
                        else:
                            # Just loop the same recording
                            self.frame_idx = 0

            # Draw
            self.screen.fill(self.bg_color)
            self._draw_frame()
            self._draw_sidebar()
            self._draw_controls()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


def load_policy(checkpoint_path: str):
    """Load a trained policy from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]
    n_agents = checkpoint.get("n_agents", 2)

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
    """Create a default task graph."""
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


def download_from_wandb(artifact_name: str, entity: str, project: str):
    """Download artifact from wandb."""
    import os
    import wandb

    if "/" not in artifact_name:
        artifact_path = f"{entity}/{project}/{artifact_name}"
    else:
        artifact_path = artifact_name

    print(f"Downloading artifact: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    for f in os.listdir(artifact_dir):
        if f.endswith(".pt"):
            return os.path.join(artifact_dir, f)

    raise FileNotFoundError(f"No .pt file found in artifact {artifact_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive replay visualizer for trained policies")

    parser.add_argument("--checkpoint", "-c", type=str, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--artifact", "-a", type=str, help="Wandb artifact name")
    parser.add_argument("--layout", "-l", default="cramped_room", help="Overcooked layout name")
    parser.add_argument("--entity", default="raphael-64-university-of-waterloo", help="Wandb entity")
    parser.add_argument("--project", default="lgc-marl-overcooked", help="Wandb project")
    parser.add_argument("--scale", type=float, default=1.5, help="Display scale factor")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if args.artifact:
        checkpoint_path = download_from_wandb(args.artifact, args.entity, args.project)

    if not checkpoint_path:
        # Try default
        default_path = "checkpoints/best_policy.pt"
        if Path(default_path).exists():
            checkpoint_path = default_path
            print(f"Using default checkpoint: {default_path}")
        else:
            print("No checkpoint specified. Use --checkpoint or --artifact")
            print("Example: python replay_visualizer.py --checkpoint checkpoints/best_policy.pt")
            sys.exit(1)

    print(f"Loading policy from {checkpoint_path}")
    policy = load_policy(checkpoint_path)
    graph = create_default_graph()

    print(f"\nStarting visualizer for layout: {args.layout}")
    print("Running FULL 400-step episodes")
    print("Controls: SPACE=play/pause, R=new episode, L=toggle loop, Q=quit")

    visualizer = ReplayVisualizer(
        policy=policy,
        graph=graph,
        layout=args.layout,
        device=args.device,
        scale=args.scale,
    )
    visualizer.run()
