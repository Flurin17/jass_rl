from __future__ import annotations

import random
from typing import Callable, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

from env.jass_aec_env import ACTION_COUNT, JassAECEnv


OpponentPolicy = Callable[[JassAECEnv, str], int]


def policy_lowest(env: JassAECEnv, agent: str) -> int:
    obs = env.observe(agent)
    mask = obs["action_mask"]
    legal = np.flatnonzero(mask)
    return int(legal[0])


def policy_random(rng: random.Random) -> OpponentPolicy:
    def _policy(env: JassAECEnv, agent: str) -> int:
        obs = env.observe(agent)
        mask = obs["action_mask"]
        legal = np.flatnonzero(mask)
        return int(rng.choice(list(legal)))

    return _policy


class JassSingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: Optional[int] = None,
        enable_bidding: bool = True,
        enable_weis: bool = True,
        mode: Optional[str] = None,
        trump_suit: Optional[str] = None,
        starter: int = 0,
        opponent_policy: Optional[OpponentPolicy] = None,
        opponent_sampler: Optional[Callable[[random.Random], OpponentPolicy]] = None,
    ) -> None:
        super().__init__()
        self.env = JassAECEnv(
            seed=seed,
            enable_bidding=enable_bidding,
            enable_weis=enable_weis,
            mode=mode,
            trump_suit=trump_suit,
            starter=starter,
        )
        self._rng = random.Random(seed)
        self.opponent_policy = opponent_policy or policy_lowest
        self.opponent_sampler = opponent_sampler
        self._current_opponent_policy: OpponentPolicy = self.opponent_policy

        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(118,), dtype=np.float32
        )

    def set_opponent_sampler(
        self, sampler: Optional[Callable[[random.Random], OpponentPolicy]]
    ) -> None:
        self.opponent_sampler = sampler

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = random.Random(seed)
        self.env.reset(seed=seed, options=options)
        if self.opponent_sampler is not None:
            self._current_opponent_policy = self.opponent_sampler(self._rng)
        else:
            self._current_opponent_policy = self.opponent_policy
        self._advance_to_agent()
        obs = self.env.observe("p0")["observation"]
        return obs, {}

    def step(self, action: int):
        if self.env.terminations.get("p0") or self.env.truncations.get("p0"):
            return self._terminal_step()

        reward = 0.0
        self.env.step(action)
        reward += self.env.last_rewards.get("p0", 0.0)

        reward += self._advance_to_agent()

        terminated = self.env.terminations.get("p0", False)
        truncated = self.env.truncations.get("p0", False)
        if terminated or truncated:
            return self._terminal_step(reward)

        obs = self.env.observe("p0")["observation"]
        return obs, reward, False, False, {}

    def _terminal_step(self, reward: float = 0.0):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, True, False, {}

    def _advance_to_agent(self) -> float:
        reward = 0.0
        while self.env.agents and self.env.agent_selection != "p0":
            agent = self.env.agent_selection
            if self.env.terminations.get(agent) or self.env.truncations.get(agent):
                self.env.step(None)
                reward += self.env.last_rewards.get("p0", 0.0)
                continue
            action = self._current_opponent_policy(self.env, agent)
            self.env.step(action)
            reward += self.env.last_rewards.get("p0", 0.0)
        return reward

    def get_action_mask(self) -> np.ndarray:
        obs = self.env.observe("p0")
        return obs["action_mask"]

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()
