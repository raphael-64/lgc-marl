#!/usr/bin/env python3
"""Static graph baseline - LGC-MARL with ONE fixed graph, no evolution.

This ablation shows that the evolutionary self-improvement is valuable,
not just having any LLM-generated graph.
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
from src.graph_generation.graph_types import TaskGraph, Subtask, SubtaskType, GraphCandidate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Super basic prompt - intentionally simple/naive
BASIC_GRAPH_PROMPT = """Generate a simple task list for two chefs making soup.

Output JSON:
```json
{{"subtasks": [
  {{"id": "step1", "type": "get_ingredient", "agent": 0, "target": "onion"}},
  {{"id": "step2", "type": "get_ingredient", "agent": 0, "target": "onion"}},
  ...
]}}
```
"""


def generate_basic_graph_llm(model: str = "gpt-4o-mini") -> TaskGraph:
    """Generate a basic graph using simple prompt."""
    from openai import OpenAI
    import json
    import re

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": BASIC_GRAPH_PROMPT}],
        max_tokens=1024,
        temperature=0.3,  # Low temp for consistency
    )

    content = response.choices[0].message.content

    # Parse JSON
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\{[\s\S]*\}", content)
        json_str = json_match.group(0) if json_match else "{}"

    data = json.loads(json_str)

    graph = TaskGraph()
    for i, st in enumerate(data.get("subtasks", [])):
        agent_id = st.get("agent", 0)
        if isinstance(agent_id, str):
            agent_id = int(agent_id) if agent_id.isdigit() else 0

        task_type_str = st.get("type", "navigate").lower()
        try:
            task_type = SubtaskType(task_type_str)
        except ValueError:
            task_type = SubtaskType.NAVIGATE

        subtask = Subtask(
            id=st.get("id", f"task_{i}"),
            task_type=task_type,
            agent_id=agent_id % 2,
            target=st.get("target"),
            dependencies=st.get("dependencies", []),
        )
        graph.add_subtask(subtask)

    return graph


def create_hardcoded_basic_graph() -> TaskGraph:
    """Create a simple hardcoded graph - very basic sequential approach."""
    graph = TaskGraph()

    # Super basic: Agent 0 does everything sequentially
    # This is intentionally suboptimal - no parallelism, no role division
    tasks = [
        ("get1", SubtaskType.GET_INGREDIENT, 0, "onion", []),
        ("put1", SubtaskType.PUT_IN_POT, 0, "pot", ["get1"]),
        ("get2", SubtaskType.GET_INGREDIENT, 0, "onion", ["put1"]),
        ("put2", SubtaskType.PUT_IN_POT, 0, "pot", ["get2"]),
        ("get3", SubtaskType.GET_INGREDIENT, 0, "onion", ["put2"]),
        ("put3", SubtaskType.PUT_IN_POT, 0, "pot", ["get3"]),
        ("wait", SubtaskType.WAIT_COOKING, 0, "pot", ["put3"]),
        ("dish", SubtaskType.GET_DISH, 0, "dish", ["wait"]),
        ("plate", SubtaskType.PLATE_SOUP, 0, "pot", ["dish"]),
        ("serve", SubtaskType.SERVE, 0, "serving", ["plate"]),
    ]

    for task_id, task_type, agent, target, deps in tasks:
        graph.add_subtask(Subtask(
            id=task_id,
            task_type=task_type,
            agent_id=agent,
            target=target,
            dependencies=deps,
        ))

    return graph


def run_static_graph(
    layout: str = "cramped_room",
    total_episodes: int = 20800,
    use_llm_graph: bool = False,  # If False, use hardcoded basic graph
    model: str = "gpt-4o-mini",  # Weaker model for basic graph
    device: str = "cpu",
    use_wandb: bool = True,
):
    """Run training with a single static graph (no evolution)."""

    logger.info("=" * 60)
    logger.info("STATIC GRAPH Training (No Evolution)")
    logger.info("=" * 60)
    logger.info(f"Layout: {layout}")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Graph source: {'LLM (basic prompt)' if use_llm_graph else 'hardcoded'}")
    logger.info(f"Device: {device}")

    if use_wandb:
        wandb.init(
            project="lgc-marl-overcooked",
            config={
                "layout": layout,
                "total_episodes": total_episodes,
                "method": "static_graph",
                "graph_source": "llm_basic" if use_llm_graph else "hardcoded",
            },
            tags=["static_graph", "ablation"],
        )
        weave.init("lgc-marl-overcooked")

    # Create environment
    env = make_overcooked_env(layout_name=layout, horizon=400, reward_shaping=True, reward_shaping_factor=0.1)

    logger.info(f"\nEnvironment:")
    logger.info(f"  Layout: {layout}")
    logger.info(f"  Grid: {env.mdp.width}x{env.mdp.height}")

    # Generate ONE graph - no evolution
    if use_llm_graph:
        logger.info(f"\nGenerating basic graph with {model}...")
        graph = generate_basic_graph_llm(model)
    else:
        logger.info("\nUsing hardcoded basic graph...")
        graph = create_hardcoded_basic_graph()

    logger.info(f"Graph has {len(graph)} subtasks:")
    logger.info(graph.to_prompt_string())

    candidate = GraphCandidate(graph=graph, origin="static_basic", generation=0)

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
        import json

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
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for basic graph (else hardcoded)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for basic graph")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    run_static_graph(
        layout=args.layout,
        total_episodes=args.episodes,
        use_llm_graph=args.use_llm,
        model=args.model,
        device=args.device,
        use_wandb=not args.no_wandb,
    )
