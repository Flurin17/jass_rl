import pytest


def _skip_if_missing():
    try:
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
        import pettingzoo  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

import numpy as np

from core.cards import MODE_TRUMP
from env.jass_aec_env import OBS_SIZE
from rl.privileged_observation import CTDE_OBS_SIZE
from rl.single_agent_env import JassSingleAgentEnv, JassTeamEnv


def test_single_agent_env_step() -> None:
    env = JassSingleAgentEnv(
        enable_bidding=False, enable_weis=False, mode=MODE_TRUMP, trump_suit="schilten", seed=3
    )
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert env.observation_space.contains(obs)

    for _ in range(5):
        mask = env.get_action_mask()
        action = int(mask.nonzero()[0][0])
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
    env.close()


def test_team_env_step() -> None:
    env = JassTeamEnv(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="schilten",
        seed=3,
    )
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert env.observation_space.contains(obs)

    for _ in range(10):
        mask = env.get_action_mask()
        action = int(mask.nonzero()[0][0])
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("env_class", [JassSingleAgentEnv, JassTeamEnv])
def test_gym_reset_seed_is_deterministic(env_class) -> None:
    env = env_class(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="schilten",
    )
    first, _ = env.reset(seed=123)
    first_rng_state = env.np_random.bit_generator.state
    repeated, _ = env.reset(seed=123)
    repeated_rng_state = env.np_random.bit_generator.state
    different, _ = env.reset(seed=456)

    np.testing.assert_array_equal(repeated, first)
    assert repeated_rng_state == first_rng_state
    assert env.np_random is not None
    assert not np.array_equal(different[:36], first[:36])
    env.close()


def test_team_policy_observation_supports_every_seat() -> None:
    env = JassTeamEnv(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="schilten",
        seed=4,
    )
    reset_observation, _ = env.reset()
    np.testing.assert_array_equal(
        reset_observation, env.policy_observation(env.env.agent_selection)
    )

    for player, agent in enumerate(env.env.possible_agents):
        observation = env.policy_observation(agent)
        assert observation.shape == (OBS_SIZE,)
        assert env.observation_space.contains(observation)
        assert observation[:36].sum() == len(env.env.state.hands[player])

    with pytest.raises(ValueError, match="unknown agent"):
        env.policy_observation("p4")
    env.close()


def test_ctde_team_env_keeps_private_hands_out_of_actor_prefix() -> None:
    env = JassTeamEnv(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="schilten",
        seed=4,
        privileged_critic=True,
    )
    observation, _ = env.reset()
    public = env.policy_observation(env.env.agent_selection)

    assert observation.shape == (CTDE_OBS_SIZE,)
    assert env.observation_space.contains(observation)
    np.testing.assert_array_equal(observation[:OBS_SIZE], public)
    assert public[:36].sum() == 9
    assert observation[OBS_SIZE:].sum() == 36
    env.close()


@pytest.mark.parametrize("env_class", [JassSingleAgentEnv, JassTeamEnv])
def test_gym_terminal_observation_and_mask_fit_spaces(env_class) -> None:
    env = env_class(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="schilten",
        seed=8,
    )
    env.reset()
    total_reward = 0.0
    for _ in range(40):
        action = int(np.flatnonzero(env.get_action_mask())[0])
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            assert terminated
            assert env.get_action_mask().sum() == 0
            assert observation.sum() == 0
            if env_class is JassTeamEnv:
                assert total_reward == (
                    env.env.state.team_points[0] - env.env.state.team_points[1]
                )
            else:
                assert total_reward == env.env.state.team_points[0]
            break
    else:
        pytest.fail("game did not terminate")
    env.close()


def test_team_reward_scaling_and_terminal_bonus() -> None:
    env = JassTeamEnv(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="eicheln",
        seed=14,
        reward_scale=0.001,
        terminal_win_bonus=1.0,
    )
    env.reset()
    total_reward = 0.0
    while True:
        action = int(np.flatnonzero(env.get_action_mask())[0])
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    difference = env.env.state.team_points[0] - env.env.state.team_points[1]
    expected = difference * 0.001 + (1.0 if difference > 0 else -1.0 if difference < 0 else 0.0)
    assert total_reward == pytest.approx(expected)
