from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

from core.ruleset import RulesetConfig
from env.jass_aec_env import (
    ACTION_COUNT,
    OBS_SIZE,
    OBSERVATION_SCHEMA_FIELDS,
    OBSERVATION_SCHEMA_NAME,
    OBSERVATION_SCHEMA_VERSION,
    JassAECEnv,
)
from rl.privileged_observation import CTDE_OBS_SIZE, augment_with_privileged_hands

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
    metadata = {
        "render_modes": [],
        "observation_schema": OBSERVATION_SCHEMA_NAME,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_fields": OBSERVATION_SCHEMA_FIELDS,
    }

    def __init__(
        self,
        seed: int | None = None,
        enable_bidding: bool = True,
        enable_weis: bool = True,
        enable_stock: bool = True,
        mode: str | None = None,
        trump_suit: str | None = None,
        starter: int = 0,
        profile: RulesetConfig | None = None,
        # Policy used for opponents p1/p3 (team B by default when agent is p0).
        opponent_policy: OpponentPolicy | None = None,
        # Policy used for partner p2 (team A when agent is p0).
        partner_policy: OpponentPolicy | None = None,
        # Optional sampler used to vary the opponent policy between episodes (p1/p3 only).
        opponent_sampler: Callable[[random.Random], OpponentPolicy] | None = None,
        trump_only_bidding: bool = False,
    ) -> None:
        super().__init__()
        self.env = JassAECEnv(
            seed=seed,
            enable_bidding=enable_bidding,
            enable_weis=enable_weis,
            enable_stock=enable_stock,
            trump_only_bidding=trump_only_bidding,
            profile=profile,
            mode=mode,
            trump_suit=trump_suit,
            starter=starter,
        )
        self._rng = random.Random(seed)
        self.metadata = dict(type(self).metadata)
        self.metadata["rules_profile_version"] = self.env.profile.version
        self.opponent_policy = opponent_policy or policy_lowest
        self.partner_policy = partner_policy or policy_lowest
        self.opponent_sampler = opponent_sampler
        self._current_opponent_policy: OpponentPolicy = self.opponent_policy

        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    def set_opponent_sampler(
        self, sampler: Callable[[random.Random], OpponentPolicy] | None
    ) -> None:
        self.opponent_sampler = sampler

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
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
            # p2 is partner of p0 in a fixed-seats game.
            if agent == "p2":
                action = self.partner_policy(self.env, agent)
            else:
                action = self._current_opponent_policy(self.env, agent)
            self.env.step(action)
            reward += self.env.last_rewards.get("p0", 0.0)
        return reward

    def get_action_mask(self) -> np.ndarray:
        obs = self.env.observe("p0")
        return obs["action_mask"]

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()


class JassTeamEnv(gym.Env):
    """Single-agent wrapper that controls both partners (p0 and p2) with one policy.

    This is a better default for "good Jass" because the learned policy can coordinate
    the team, instead of trying to compensate for a fixed partner baseline.
    """

    metadata = {
        "render_modes": [],
        "observation_schema": OBSERVATION_SCHEMA_NAME,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_fields": OBSERVATION_SCHEMA_FIELDS,
    }

    def __init__(
        self,
        seed: int | None = None,
        enable_bidding: bool = True,
        enable_weis: bool = True,
        enable_stock: bool = True,
        mode: str | None = None,
        trump_suit: str | None = None,
        starter: int = 0,
        profile: RulesetConfig | None = None,
        opponent_policy: OpponentPolicy | None = None,
        opponent_sampler: Callable[[random.Random], OpponentPolicy] | None = None,
        randomize_starter: bool = False,
        reward_scale: float = 1.0,
        terminal_win_bonus: float = 0.0,
        privileged_critic: bool = False,
        trump_only_bidding: bool = False,
        normalize_contract_reward: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be a positive finite number")
        if not math.isfinite(terminal_win_bonus) or terminal_win_bonus < 0:
            raise ValueError("terminal_win_bonus must be a non-negative finite number")
        if not isinstance(normalize_contract_reward, bool):
            raise TypeError("normalize_contract_reward must be a boolean")
        self.env = JassAECEnv(
            seed=seed,
            enable_bidding=enable_bidding,
            enable_weis=enable_weis,
            enable_stock=enable_stock,
            trump_only_bidding=trump_only_bidding,
            profile=profile,
            mode=mode,
            trump_suit=trump_suit,
            starter=starter,
        )
        self.metadata = dict(type(self).metadata)
        self.metadata["rules_profile_version"] = self.env.profile.version
        self._rng = random.Random(seed)
        self.opponent_policy = opponent_policy or policy_lowest
        self.opponent_sampler = opponent_sampler
        self._current_opponent_policy = self.opponent_policy
        self.randomize_starter = randomize_starter
        self.reward_scale = float(reward_scale)
        self.terminal_win_bonus = float(terminal_win_bonus)
        self.normalize_contract_reward = normalize_contract_reward
        self.privileged_critic = bool(privileged_critic)

        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(CTDE_OBS_SIZE if self.privileged_critic else OBS_SIZE,),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        if self.randomize_starter:
            self.env.starter = self._rng.randrange(4)
        if self.opponent_sampler is not None:
            self._current_opponent_policy = self.opponent_sampler(self._rng)
        else:
            self._current_opponent_policy = self.opponent_policy
        self.env.reset(seed=seed, options=options)
        self._advance_to_controlled()
        obs = self._obs_for_current()
        return obs, {}

    def step(self, action: int):
        if self.env.terminations.get("p0") or self.env.truncations.get("p0"):
            return self._terminal_step()

        before = self._team_point_difference()

        # Apply action for whichever controlled player is active.
        self.env.step(action)
        self._advance_to_controlled()
        point_delta = self._team_point_difference() - before
        if self.normalize_contract_reward:
            if self.env.contract_factor <= 0:  # pragma: no cover - profile validation guards it
                raise RuntimeError("contract factor must be positive")
            point_delta /= self.env.contract_factor
        reward = point_delta * self.reward_scale

        terminated = self.env.terminations.get("p0", False)
        truncated = self.env.truncations.get("p0", False)
        if terminated or truncated:
            if terminated and self.terminal_win_bonus:
                difference = self._team_point_difference()
                if difference > 0:
                    reward += self.terminal_win_bonus
                elif difference < 0:
                    reward -= self.terminal_win_bonus
            return self._terminal_step(reward)

        obs = self._obs_for_current()
        return obs, reward, False, False, {}

    def _terminal_step(self, reward: float = 0.0):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, True, False, {}

    def _advance_to_controlled(self) -> None:
        while self.env.agents and self.env.agent_selection not in ("p0", "p2"):
            agent = self.env.agent_selection
            if self.env.terminations.get(agent) or self.env.truncations.get(agent):
                self.env.step(None)
                continue
            action = self._current_opponent_policy(self.env, agent)
            self.env.step(action)

    def _team_point_difference(self) -> float:
        if self.env.state is None:
            return 0.0
        return float(self.env.state.team_points[0] - self.env.state.team_points[1])

    def set_opponent_sampler(
        self, sampler: Callable[[random.Random], OpponentPolicy] | None
    ) -> None:
        self.opponent_sampler = sampler

    def _obs_for_current(self) -> np.ndarray:
        agent = self.env.agent_selection
        public = self.policy_observation(agent)
        if not self.privileged_critic:
            return public

        observer = int(agent.removeprefix("p"))
        if self.env.state is not None:
            hands = self.env.state.hands
        elif self.env._pending_hands is not None:
            hands = self.env._pending_hands
        else:  # pragma: no cover - guarded by the active nonterminal phase
            hands = [[], [], [], []]
        return augment_with_privileged_hands(public, hands, observer)

    def policy_observation(self, agent: str) -> np.ndarray:
        """Return the exact canonical policy input for any requested seat.

        The schema is relative to ``agent``, so the same shared team policy can
        consume snapshots for p0/p2 or p1/p3 without an absolute-seat suffix.
        Each snapshot contains only that agent's private hand.
        """

        if agent not in self.env.possible_agents:
            raise ValueError(f"unknown agent: {agent}")
        return self.env.observe(agent)["observation"]

    def get_action_mask(self) -> np.ndarray:
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        return obs["action_mask"]

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask()
