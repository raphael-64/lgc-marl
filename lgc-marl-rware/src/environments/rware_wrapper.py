"""RWARE environment wrapper with graph-conditioned execution."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from ..graph_generation.graph_types import Subtask, SubtaskType, TaskGraph

logger = logging.getLogger(__name__)


class RWAREGraphWrapper(gym.Wrapper):
    """
    Wraps RWARE environment with graph-conditioned execution.

    Provides:
    - Graph-based subtask tracking
    - Enhanced reward shaping based on subtask completion
    - Environment state extraction for LLM prompts
    """

    def __init__(
        self,
        env_name: str = "rware-small-4ag-v2",
        max_steps: int = 500,
        subtask_reward: float = 0.1,
        collision_penalty: float = -0.05,
    ):
        """
        Initialize the RWARE wrapper.

        Args:
            env_name: Name of the RWARE environment
            max_steps: Maximum steps per episode
            subtask_reward: Bonus reward for completing subtasks
            collision_penalty: Penalty for agent collisions
        """
        try:
            import rware  # noqa: F401
        except ImportError:
            raise ImportError("RWARE not installed. Install with: pip install rware")

        env = gym.make(env_name)
        super().__init__(env)

        self.env_name = env_name
        self.max_steps = max_steps
        self.subtask_reward = subtask_reward
        self.collision_penalty = collision_penalty

        # Get environment properties
        self.n_agents = env.unwrapped.n_agents
        self.grid_size = self._get_grid_size()

        # State tracking
        self.current_graph: Optional[TaskGraph] = None
        self.completed_subtasks: Set[str] = set()
        self.step_count = 0
        self.episode_stats: Dict[str, Any] = {}

        # Action mapping (RWARE uses 5 discrete actions)
        self.ACTIONS = {
            "NOOP": 0,
            "FORWARD": 1,
            "LEFT": 2,
            "RIGHT": 3,
            "TOGGLE": 4,  # Pick up / drop shelf
        }

        logger.info(f"Initialized RWAREGraphWrapper: {env_name}, {self.n_agents} agents")

    def _get_grid_size(self) -> Tuple[int, int]:
        """Extract grid size from environment."""
        try:
            env = self.env.unwrapped
            if hasattr(env, "grid_size"):
                return env.grid_size
            elif hasattr(env, "_grid_shape"):
                return env._grid_shape
            else:
                # Parse from env name
                if "tiny" in self.env_name:
                    return (11, 11)
                elif "small" in self.env_name:
                    return (11, 20)
                elif "medium" in self.env_name:
                    return (16, 20)
                elif "large" in self.env_name:
                    return (16, 29)
                return (11, 20)  # Default to small
        except Exception:
            return (11, 20)

    def reset(
        self,
        graph: Optional[TaskGraph] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Any, Dict]:
        """
        Reset environment with optional graph.

        Args:
            graph: Task decomposition graph for this episode
            seed: Random seed
            options: Additional options

        Returns:
            observations, info dict
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset state tracking
        self.current_graph = graph
        self.completed_subtasks = set()
        self.step_count = 0
        self.episode_stats = {
            "total_reward": 0.0,
            "subtasks_completed": 0,
            "collisions": 0,
            "deliveries": 0,
        }

        # Add graph info to initial info
        if graph:
            info["graph"] = graph
            info["ready_subtasks"] = [s.id for s in graph.get_ready_subtasks(self.completed_subtasks)]
            info["completed_subtasks"] = list(self.completed_subtasks)

        info["n_agents"] = self.n_agents
        info["step"] = 0

        return obs, info

    def step(self, actions: List[int]) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Step environment and track subtask completion.

        Args:
            actions: List of actions, one per agent

        Returns:
            observations, reward (aggregated), terminated, truncated, info
            Individual agent rewards are in info["agent_rewards"]
        """
        # Take environment step
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self.step_count += 1

        # Convert rewards to list if needed
        if not isinstance(rewards, (list, np.ndarray)):
            rewards = [rewards] * self.n_agents
        rewards = list(rewards)

        # Track deliveries
        deliveries_this_step = info.get("delivered", 0)
        if isinstance(deliveries_this_step, (list, np.ndarray)):
            deliveries_this_step = sum(deliveries_this_step)
        self.episode_stats["deliveries"] += deliveries_this_step

        # Check for subtask completion if we have a graph
        if self.current_graph:
            newly_completed = self._check_subtask_completion(info)
            self.completed_subtasks.update(newly_completed)

            # Add subtask completion bonus
            if newly_completed:
                subtask_bonus = len(newly_completed) * self.subtask_reward
                rewards = [r + subtask_bonus / self.n_agents for r in rewards]
                self.episode_stats["subtasks_completed"] += len(newly_completed)

            info["completed_subtasks"] = list(self.completed_subtasks)
            info["newly_completed"] = list(newly_completed)
            info["ready_subtasks"] = [
                s.id for s in self.current_graph.get_ready_subtasks(self.completed_subtasks)
            ]

            # Check if all subtasks are complete
            if len(self.completed_subtasks) == len(self.current_graph):
                info["all_subtasks_complete"] = True

        # Check for collisions (if available in info)
        if info.get("collision", False):
            collision_penalty = self.collision_penalty
            rewards = [r + collision_penalty for r in rewards]
            self.episode_stats["collisions"] += 1

        # Truncate if max steps reached
        if self.step_count >= self.max_steps:
            truncated = True

        # Update episode stats
        self.episode_stats["total_reward"] += sum(rewards) / self.n_agents

        # Add step info
        info["n_agents"] = self.n_agents
        info["step"] = self.step_count
        info["episode_stats"] = self.episode_stats.copy()
        info["agent_rewards"] = rewards  # Individual agent rewards for MARL training

        # Return aggregated reward for Gymnasium compatibility
        aggregated_reward = float(sum(rewards) / len(rewards))

        return obs, aggregated_reward, terminated, truncated, info

    def _check_subtask_completion(self, info: Dict) -> Set[str]:
        """
        Check which subtasks were completed this step.

        This uses heuristics based on environment state.
        """
        if not self.current_graph:
            return set()

        completed = set()

        for subtask_id, subtask in self.current_graph.subtasks.items():
            if subtask_id in self.completed_subtasks:
                continue

            # Check if dependencies are met
            if not all(dep in self.completed_subtasks for dep in subtask.dependencies):
                continue

            # Check based on subtask type
            if self._is_subtask_complete(subtask, info):
                completed.add(subtask_id)

        return completed

    def _is_subtask_complete(self, subtask: Subtask, info: Dict) -> bool:
        """
        Check if a specific subtask is complete.

        Uses heuristics based on available info.
        """
        # Get agent state
        agent_id = subtask.agent_id

        # Get carrying status if available
        carrying = info.get("carrying", [False] * self.n_agents)
        if isinstance(carrying, np.ndarray):
            carrying = carrying.tolist()

        is_carrying = carrying[agent_id] if agent_id < len(carrying) else False

        if subtask.task_type == SubtaskType.FETCH:
            # Fetch is complete if agent is now carrying something
            # This is a simplification - ideally we'd track which shelf
            return is_carrying

        elif subtask.task_type == SubtaskType.DELIVER:
            # Deliver is complete if agent was carrying and now delivered
            delivered = info.get("delivered", [0] * self.n_agents)
            if isinstance(delivered, (int, float)):
                # Environment might report total deliveries
                return delivered > self.episode_stats.get("deliveries", 0)
            else:
                agent_delivered = delivered[agent_id] if agent_id < len(delivered) else 0
                return agent_delivered > 0

        elif subtask.task_type == SubtaskType.NAVIGATE:
            # Navigate is complete if agent reached target position
            # This would require position tracking
            return True  # Simplified: assume navigate tasks complete immediately

        elif subtask.task_type == SubtaskType.WAIT:
            # Wait is complete if dependencies are met (already checked above)
            return True

        elif subtask.task_type == SubtaskType.COORDINATE:
            # Coordination is complete if all agents in dependencies are ready
            return True

        return False

    def get_env_state(self) -> Dict[str, Any]:
        """
        Get current environment state for LLM prompt.

        Returns dict with environment information for task decomposition.
        """
        env = self.env.unwrapped

        state = {
            "grid_size": f"{self.grid_size[0]}x{self.grid_size[1]}",
            "n_agents": self.n_agents,
            "env_name": self.env_name,
        }

        # Try to get agent positions
        try:
            if hasattr(env, "agents"):
                state["agent_positions"] = [
                    (a.x, a.y) if hasattr(a, "x") else None for a in env.agents
                ]
        except Exception:
            pass

        # Try to get requested shelves
        try:
            if hasattr(env, "request_queue"):
                state["requested_shelves"] = [f"shelf_{i}" for i in range(len(env.request_queue))]
            elif hasattr(env, "_request_queue"):
                state["requested_shelves"] = [f"shelf_{i}" for i in range(len(env._request_queue))]
            else:
                # Default based on difficulty
                n_requests = self.n_agents  # Default
                if "easy" in self.env_name:
                    n_requests = self.n_agents * 2
                elif "hard" in self.env_name:
                    n_requests = self.n_agents // 2
                state["requested_shelves"] = [f"shelf_{i}" for i in range(n_requests)]
        except Exception:
            state["requested_shelves"] = [f"shelf_{i}" for i in range(self.n_agents)]

        # Try to get workstation position
        try:
            if hasattr(env, "goals"):
                state["workstation_position"] = "goal area"
        except Exception:
            state["workstation_position"] = "designated delivery area"

        return state

    def get_obs_dim(self) -> int:
        """Get observation dimension for a single agent."""
        obs_space = self.observation_space
        if hasattr(obs_space, "spaces"):
            # Tuple observation space
            return obs_space.spaces[0].shape[0]
        elif hasattr(obs_space, "shape"):
            return obs_space.shape[0]
        return 78  # Default RWARE observation size

    def get_action_dim(self) -> int:
        """Get action dimension."""
        action_space = self.action_space
        if hasattr(action_space, "spaces"):
            # Tuple action space
            return action_space.spaces[0].n
        elif hasattr(action_space, "n"):
            return action_space.n
        return 5  # Default RWARE action size

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render()


class RWARETaskGenerator:
    """Generate warehouse tasks of varying difficulty."""

    def __init__(self, n_agents: int = 4):
        """
        Initialize task generator.

        Args:
            n_agents: Number of agents in the environment
        """
        self.n_agents = n_agents

    def generate_task(self, difficulty: str = "medium") -> str:
        """
        Generate a task description.

        Args:
            difficulty: Task difficulty (easy, medium, hard)

        Returns:
            Task description string
        """
        if difficulty == "easy":
            n_shelves = max(1, self.n_agents // 2)
            time_pressure = "at your own pace"
        elif difficulty == "medium":
            n_shelves = self.n_agents
            time_pressure = "efficiently"
        else:  # hard
            n_shelves = self.n_agents * 2
            time_pressure = "as quickly as possible"

        return (
            f"Coordinate {self.n_agents} warehouse robots to deliver {n_shelves} "
            f"requested shelves to the workstation {time_pressure}. "
            f"Robots must pick up shelves from their locations and bring them to the goal area. "
            f"Avoid collisions between robots and minimize total delivery time."
        )

    def generate_task_with_constraints(
        self,
        n_shelves: Optional[int] = None,
        priority_shelves: Optional[List[str]] = None,
        blocked_areas: Optional[List[Tuple[int, int]]] = None,
    ) -> str:
        """
        Generate a task with specific constraints.

        Args:
            n_shelves: Number of shelves to deliver
            priority_shelves: Shelves that should be delivered first
            blocked_areas: Grid positions that are blocked

        Returns:
            Task description string
        """
        if n_shelves is None:
            n_shelves = self.n_agents

        task = f"Coordinate {self.n_agents} warehouse robots to deliver {n_shelves} shelves. "

        if priority_shelves:
            task += f"Priority deliveries: {', '.join(priority_shelves)}. "

        if blocked_areas:
            task += f"Avoid blocked areas at positions: {blocked_areas}. "

        task += "Minimize total time and avoid collisions."

        return task
