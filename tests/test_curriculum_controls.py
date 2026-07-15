from __future__ import annotations

import random

import numpy as np
import pytest

from core.cards import MODE_TRUMP
from env.jass_aec_env import (
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    BIDDING_TRUMP_ACTIONS,
    BIDDING_UNEUFE_ACTION,
    JassAECEnv,
    trump_only_action_mask,
)
from rl.single_agent_env import JassTeamEnv
from rl.train_selfplay import (
    OpponentMixtureSampler,
    TrainConfig,
    _build_env,
    _fork_environment_mismatches,
    _parse_args,
    _sample_opponent_name,
    environment_manifest,
    normalized_opponent_mixture,
)


def test_trump_only_bidding_is_the_environment_legal_space_for_both_bidders() -> None:
    env = JassAECEnv(
        seed=5,
        enable_bidding=True,
        enable_weis=False,
        enable_stock=False,
        trump_only_bidding=True,
    )
    env.reset(seed=5)

    original = env.observe("p0")["action_mask"]
    restricted_copy = trump_only_action_mask(original)
    assert restricted_copy is not original
    assert restricted_copy[BIDDING_OBEABE_ACTION] == 0
    assert restricted_copy[BIDDING_UNEUFE_ACTION] == 0
    assert all(restricted_copy[action] == 1 for action in BIDDING_TRUMP_ACTIONS)
    assert restricted_copy[BIDDING_PUSH_ACTION] == 1

    with pytest.raises(ValueError, match="illegal bidding action"):
        env.step(BIDDING_OBEABE_ACTION)

    env.step(BIDDING_PUSH_ACTION)
    partner = env.agent_selection
    partner_mask = env.observe(partner)["action_mask"]
    assert all(partner_mask[action] == 1 for action in BIDDING_TRUMP_ACTIONS)
    assert partner_mask[BIDDING_OBEABE_ACTION] == 0
    assert partner_mask[BIDDING_UNEUFE_ACTION] == 0
    assert partner_mask[BIDDING_PUSH_ACTION] == 0

    env.step(next(iter(BIDDING_TRUMP_ACTIONS)))
    assert env.mode == MODE_TRUMP
    env.close()


def _play_lowest_team_round(*, normalized: bool) -> tuple[float, int, int]:
    env = JassTeamEnv(
        seed=17,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="schilten",
        reward_scale=1.0,
        terminal_win_bonus=0.0,
        normalize_contract_reward=normalized,
    )
    env.reset(seed=17)
    total_reward = 0.0
    while True:
        action = int(np.flatnonzero(env.get_action_mask())[0])
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    assert env.env.state is not None
    difference = env.env.state.team_points[0] - env.env.state.team_points[1]
    factor = env.env.contract_factor
    env.close()
    return total_reward, difference, factor


def test_contract_normalized_reward_removes_the_active_contract_factor() -> None:
    unnormalized, difference, factor = _play_lowest_team_round(normalized=False)
    normalized, repeated_difference, repeated_factor = _play_lowest_team_round(
        normalized=True
    )

    assert repeated_difference == difference
    assert repeated_factor == factor == 2
    assert unnormalized == pytest.approx(difference)
    assert normalized == pytest.approx(difference / factor)


def test_opponent_mixture_sampling_and_manifest_are_deterministic() -> None:
    raw_mixture = (("strategic", 7.0), ("random", 2.0), ("verardo", 1.0))
    normalized = normalized_opponent_mixture(raw_mixture)
    assert dict(normalized) == pytest.approx(
        {"strategic": 0.7, "random": 0.2, "verardo": 0.1}
    )

    first_rng = random.Random(1234)
    second_rng = random.Random(1234)
    first = [_sample_opponent_name(raw_mixture, first_rng) for _ in range(100)]
    second = [_sample_opponent_name(raw_mixture, second_rng) for _ in range(100)]
    assert first == second
    assert set(first) == {"strategic", "random", "verardo"}

    config = _parse_args(
        [
            "--total-steps",
            "1024",
            "--iterations",
            "2",
            "--profile",
            "verardo-v1",
            "--no-weis",
            "--no-stock",
            "--trump-only-bidding",
            "--normalize-contract-reward",
            "--opponent-mixture",
            "strategic=7,random=2,verardo=1",
        ]
    )
    manifest = environment_manifest(config)
    assert manifest["trump_only_bidding"] is True
    assert manifest["normalize_contract_reward"] is True
    assert manifest["opponent_mixture_source"] == "configured"
    assert manifest["opponent_mixture"] == [
        {"opponent": "strategic", "weight": 0.7},
        {"opponent": "random", "weight": 0.2},
        {"opponent": "verardo", "weight": 0.1},
    ]


def test_curriculum_controls_reject_incompatible_configurations() -> None:
    with pytest.raises(ValueError, match="trump-only bidding"):
        TrainConfig(
            opponent_mixture=(("verardo", 1.0),),
            enable_bidding=True,
        ).validate()
    with pytest.raises(ValueError, match="control-team"):
        TrainConfig(
            control_team=False,
            ctde=False,
            normalize_contract_reward=True,
        ).validate()


def test_exact_verardo_mixture_can_complete_a_training_round() -> None:
    config = TrainConfig(
        seed=19,
        profile_name="verardo-v1",
        enable_weis=False,
        enable_stock=False,
        trump_only_bidding=True,
        opponent_mixture=(("verardo", 1.0),),
    )
    config.validate()
    env = _build_env(
        config,
        opponent_sampler=OpponentMixtureSampler(config.opponent_mixture),
        seed=19,
    )
    env.reset(seed=19)
    for _ in range(30):
        action = int(np.flatnonzero(env.get_action_mask())[0])
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        pytest.fail("matched Verardo training round did not terminate")
    assert env.env.mode == MODE_TRUMP
    env.close()


def test_fork_compatibility_treats_legacy_missing_false_as_false() -> None:
    expected = environment_manifest(TrainConfig())
    legacy = dict(expected)
    legacy.pop("normalize_contract_reward")
    legacy.pop("trump_only_bidding")
    assert _fork_environment_mismatches(legacy, expected) == []

    changed = dict(expected)
    changed["normalize_contract_reward"] = True
    assert _fork_environment_mismatches(changed, expected) == [
        "normalize_contract_reward"
    ]
