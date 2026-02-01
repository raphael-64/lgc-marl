"""
Gymnasium-compatible wrapper for Overcooked-AI environment.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv


class OvercookedGymWrapper(gym.Env):
    """
    Gymnasium wrapper for Overcooked-AI.

    Converts the Overcooked environment to a format compatible with our MARL trainer.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        layout_name: str = "cramped_room",
        horizon: int = 400,
        reward_shaping: bool = True,
        reward_shaping_factor: float = 1.0,
    ):
        """
        Args:
            layout_name: Name of the Overcooked layout
            horizon: Maximum timesteps per episode
            reward_shaping: Whether to use dense reward shaping
            reward_shaping_factor: Multiplier for shaped rewards
        """
        super().__init__()

        self.layout_name = layout_name
        self.horizon = horizon
        self.reward_shaping = reward_shaping
        self.reward_shaping_factor = reward_shaping_factor

        # Create base environment
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        # Environment properties
        self.n_agents = 2  # Overcooked is always 2 players
        self.n_actions = len(Action.ALL_ACTIONS)  # 6 actions

        # Action space: discrete for each agent
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: lossless encoding gives (2, height, width, 26)
        # We'll flatten per agent
        obs_shape = self.mdp.lossless_state_encoding(self.base_env.state)
        self._obs_shape_per_agent = obs_shape[0].flatten().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_shape_per_agent,),
            dtype=np.float32
        )

        # Track metrics
        self.total_sparse_reward = 0
        self.deliveries = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        self.base_env.reset()
        self.total_sparse_reward = 0
        self.deliveries = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self,
        actions: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            actions: Tuple of action indices for each agent

        Returns:
            observations: List of obs arrays for each agent
            rewards: List of rewards for each agent
            terminated: Whether episode ended due to task completion
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Convert action indices to Overcooked actions
        joint_action = tuple(Action.ALL_ACTIONS[a] for a in actions)

        # Step environment
        next_state, sparse_reward, done, info = self.base_env.step(joint_action)

        # Track deliveries (each soup = +20 reward)
        if sparse_reward > 0:
            self.deliveries += int(sparse_reward / 20)
        self.total_sparse_reward += sparse_reward

        # Compute per-agent rewards
        if self.reward_shaping and "shaped_r_by_agent" in info:
            shaped_rewards = info["shaped_r_by_agent"]
            rewards = [
                sparse_reward / 2 + self.reward_shaping_factor * shaped_rewards[i]
                for i in range(self.n_agents)
            ]
        else:
            # Split sparse reward equally
            rewards = [sparse_reward / self.n_agents] * self.n_agents

        # Check termination
        terminated = done and self.base_env.state.timestep < self.horizon
        truncated = done and self.base_env.state.timestep >= self.horizon

        obs = self._get_obs()
        info = self._get_info()
        info["sparse_reward"] = sparse_reward

        return obs, rewards, terminated, truncated, info

    def _get_obs(self) -> List[np.ndarray]:
        """Get observations for each agent."""
        # Lossless encoding: (n_agents, height, width, 26)
        lossless = np.array(self.mdp.lossless_state_encoding(self.base_env.state))

        # Flatten per agent
        obs = [lossless[i].flatten().astype(np.float32) for i in range(self.n_agents)]
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict."""
        state = self.base_env.state
        return {
            "timestep": state.timestep if state else 0,
            "deliveries": self.deliveries,
            "total_sparse_reward": self.total_sparse_reward,
            "player_positions": [p.position for p in state.players] if state else [],
        }

    def render(self):
        """Render the environment."""
        # Could implement pygame rendering here
        pass

    def close(self):
        """Clean up."""
        pass

    def get_env_state(self) -> Dict[str, Any]:
        """Get current environment state for LLM planning."""
        state = self.base_env.state

        # Extract useful info for LLM
        env_state = {
            "layout": self.layout_name,
            "n_agents": self.n_agents,
            "timestep": state.timestep if state else 0,
            "horizon": self.horizon,
            "player_positions": [p.position for p in state.players] if state else [],
            "player_holdings": [
                str(p.held_object) if p.held_object else None
                for p in state.players
            ] if state else [],
            "orders": [o["ingredients"] for o in self.mdp.start_all_orders],
            "grid_width": self.mdp.width,
            "grid_height": self.mdp.height,
            "terrain": ["".join(row) for row in self.mdp.terrain_mtx],
        }

        # Add pot states if available
        if state:
            pot_states = []
            for obj in state.all_objects_list:
                if hasattr(obj, "name") and "soup" in str(obj).lower():
                    pot_states.append(str(obj))
            env_state["pot_states"] = pot_states

        return env_state


def make_overcooked_env(
    layout_name: str = "cramped_room",
    horizon: int = 400,
    reward_shaping: bool = True,
    **kwargs,
) -> OvercookedGymWrapper:
    """Factory function to create Overcooked environment."""
    return OvercookedGymWrapper(
        layout_name=layout_name,
        horizon=horizon,
        reward_shaping=reward_shaping,
        **kwargs,
    )
