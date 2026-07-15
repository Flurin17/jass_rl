import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.cards import MODE_TRUMP, Card
from core.ruleset import STANDARD_RULES_PROFILE
from env.jass_aec_env import (
    ACTION_COUNT,
    BIDDING_PUSH_ACTION,
    OBS_SIZE,
    OBSERVATION_SCHEMA_VERSION,
    JassAECEnv,
)
from rl.advisor import AdvisorState, JassAdvisor, card_code, load_advisor, parse_card
from rl.eval import EvaluationEnvironment
from rl.hybrid_policy import NeuralGuidanceConfig
from rl.run_manifest import MANIFEST_FILENAME, MANIFEST_VERSION, write_manifest
from rl.search_policy import PIMCConfig, policy_implementation_identity


class _Policy:
    def obs_to_tensor(self, observation):
        return np.asarray(observation), False

    def get_distribution(self, tensor, action_masks):
        del tensor
        probabilities = np.asarray(action_masks, dtype=np.float64)
        probabilities /= probabilities.sum()
        return SimpleNamespace(distribution=SimpleNamespace(probs=probabilities.reshape(1, -1)))


class _Model:
    observation_space = SimpleNamespace(shape=(OBS_SIZE,))
    policy = _Policy()


def _initial_state(env: JassAECEnv) -> AdvisorState:
    assert env.state is not None
    return AdvisorState(
        hand=tuple(env.state.hands[0]),
        completed_tricks=(),
        current_trick=(),
        mode=MODE_TRUMP,
        trump_suit="rosen",
        leader=0,
        bidding_enabled=False,
        bid_starter=None,
        bid_chooser=None,
        announcement_enabled=False,
    )


def _relative_state(env: JassAECEnv, actor: int) -> AdvisorState:
    assert env.state is not None
    assert env.bidding is not None and env._chooser is not None

    def relative(player: int) -> int:
        return (player - actor) % 4

    own_team = env.state.team_index(actor)
    decisions = []
    for offset in range(4):
        decision = env._announcement_decisions[(actor + offset) % 4]
        decisions.append("announce" if decision else "pass")
    return AdvisorState(
        hand=tuple(env.state.hands[actor]),
        completed_tricks=tuple(
            tuple((relative(player), card) for player, card in trick.plays)
            for trick in env.state.completed_tricks
        ),
        current_trick=tuple(
            (relative(player), card) for player, card in env.state.trick.plays
        ),
        mode=env.state.mode,
        trump_suit=env.state.trump_suit,
        leader=relative(env.state.leader),
        team_points=(
            env.state.team_points[own_team],
            env.state.team_points[1 - own_team],
        ),
        bidding_enabled=True,
        bid_starter=relative(env.bidding.starter),
        bid_chooser=relative(env._chooser),
        bid_pushed=env.bidding.pushed,
        announcement_enabled=True,
        announcement_status=tuple(decisions),  # type: ignore[arg-type]
    )


def test_public_state_encoding_matches_environment_at_card_boundary() -> None:
    env = JassAECEnv(
        seed=17,
        profile=STANDARD_RULES_PROFILE,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        starter=0,
    )
    env.reset(seed=17)
    try:
        state = _initial_state(env)
        visible = env.observe("p0")

        np.testing.assert_array_equal(
            state.observation(STANDARD_RULES_PROFILE),
            visible["observation"],
        )
        np.testing.assert_array_equal(
            state.action_mask(STANDARD_RULES_PROFILE),
            visible["action_mask"],
        )
    finally:
        env.close()


def test_public_state_matches_every_play_boundary_in_full_round() -> None:
    env = JassAECEnv(
        seed=19,
        profile=STANDARD_RULES_PROFILE,
        enable_bidding=True,
        enable_weis=True,
        enable_stock=True,
        starter=1,
    )
    env.reset(seed=19)
    try:
        env.step(BIDDING_PUSH_ACTION)
        while env.phase != "play":
            mask = env.observe(env.agent_selection)["action_mask"]
            env.step(int(np.flatnonzero(mask)[0]))

        compared = 0
        while env.phase == "play":
            agent = env.agent_selection
            actor = int(agent[1:])
            visible = env.observe(agent)
            state = _relative_state(env, actor)
            np.testing.assert_array_equal(
                state.observation(STANDARD_RULES_PROFILE),
                visible["observation"],
            )
            np.testing.assert_array_equal(
                state.action_mask(STANDARD_RULES_PROFILE),
                visible["action_mask"],
            )
            compared += 1
            env.step(int(np.flatnonzero(visible["action_mask"])[0]))

        assert compared == 36
    finally:
        env.close()


def test_advisor_returns_ranked_legal_actions_with_uncertainty() -> None:
    env = JassAECEnv(
        seed=23,
        profile=STANDARD_RULES_PROFILE,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        starter=0,
    )
    env.reset(seed=23)
    try:
        state = _initial_state(env)
        advisor = JassAdvisor(
            _Model(),
            profile=STANDARD_RULES_PROFILE,
            search_config=PIMCConfig(
                determinizations=2,
                max_rollouts=18,
                rollout_randomness=0.0,
                common_random_numbers=True,
            ),
            guidance_config=NeuralGuidanceConfig(
                max_search_gap_raw_points=0.0,
                min_neural_probability=1.0,
            ),
            seed=5,
        )

        advice = advisor.advise(state)
        mask = state.action_mask(STANDARD_RULES_PROFILE)

        assert mask[advice.selected_action] == 1
        assert advice.selected_card == card_code(parse_card(advice.selected_card))
        assert len(advice.actions) == int(mask.sum())
        assert all(mask[action.action] for action in advice.actions)
        assert all(action.determinizations == 2 for action in advice.actions)
        assert all(action.standard_error is not None for action in advice.actions)
        assert sum(action.neural_probability for action in advice.actions) == pytest.approx(1.0)
        assert advice.manifest_sha256 is None
        assert advice.policy_implementation == policy_implementation_identity()
    finally:
        env.close()


def test_advisor_fallback_is_strict_json() -> None:
    env = JassAECEnv(
        seed=29,
        profile=STANDARD_RULES_PROFILE,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        starter=0,
    )
    env.reset(seed=29)
    try:
        advisor = JassAdvisor(
            _Model(),
            profile=STANDARD_RULES_PROFILE,
            search_config=PIMCConfig(
                determinizations=2,
                max_rollouts=18,
                assignment_node_limit=1,
            ),
        )

        payload = advisor.advise(_initial_state(env)).to_dict()

        json.dumps(payload, allow_nan=False)
        assert all(action["expected_margin"] is None for action in payload["actions"])
    finally:
        env.close()


def test_advisor_enforces_manifested_feature_flags() -> None:
    env = JassAECEnv(
        seed=31,
        profile=STANDARD_RULES_PROFILE,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        starter=0,
    )
    env.reset(seed=31)
    try:
        advisor = JassAdvisor(
            _Model(),
            profile=STANDARD_RULES_PROFILE,
            bidding_enabled=True,
            announcement_enabled=False,
        )
        with pytest.raises(ValueError, match="bidding_enabled"):
            advisor.advise(_initial_state(env))

        trump_only = JassAdvisor(
            _Model(),
            profile=STANDARD_RULES_PROFILE,
            bidding_enabled=True,
            announcement_enabled=False,
            trump_only_bidding=True,
        )
        non_trump = AdvisorState(
            hand=tuple(
                Card("rosen", rank)
                for rank in ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
            ),
            completed_tricks=(),
            current_trick=(),
            mode="obeabe",
            trump_suit=None,
            leader=0,
            bidding_enabled=True,
            bid_starter=0,
            bid_chooser=0,
            announcement_enabled=False,
        )
        with pytest.raises(ValueError, match="trump-only"):
            trump_only.advise(non_trump)
    finally:
        env.close()


def test_state_rejects_duplicates_wrong_turn_and_malformed_cards() -> None:
    payload = {
        "hand": ["rosen:A"] * 9,
        "mode": "trump",
        "trump_suit": "rosen",
        "leader": 0,
    }
    with pytest.raises(ValueError, match="unique"):
        AdvisorState.from_dict(payload)

    with pytest.raises(ValueError, match="next relative player"):
        AdvisorState(
            hand=tuple(Card("rosen", rank) for rank in ("6", "7", "8", "9", "10", "J", "Q", "K")),
            completed_tricks=(),
            current_trick=((0, Card("eicheln", "A")),),
            mode=MODE_TRUMP,
            trump_suit="rosen",
            leader=0,
            bidding_enabled=False,
            bid_starter=None,
            bid_chooser=None,
            announcement_enabled=False,
        ).validate()

    with pytest.raises(ValueError, match="suit:rank"):
        parse_card("ace")


def test_state_rejects_impossible_bidding_and_unresolved_weis() -> None:
    hand = [
        f"rosen:{rank}"
        for rank in ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
    ]
    base = {
        "hand": hand,
        "mode": "trump",
        "trump_suit": "schilten",
        "leader": 0,
        "bid_starter": 0,
        "bid_chooser": 0,
        "announcement_status": ["pass"] * 4,
    }

    with pytest.raises(ValueError, match="bid_chooser"):
        AdvisorState.from_dict({**base, "bid_chooser": 2, "bid_pushed": False})
    with pytest.raises(ValueError, match="first trick leader"):
        AdvisorState.from_dict({**base, "bid_starter": 1, "bid_chooser": 1})
    with pytest.raises(ValueError, match="leader"):
        AdvisorState.from_dict({**base, "leader": 0.0})
    with pytest.raises(ValueError, match="bid_starter"):
        AdvisorState.from_dict({**base, "bid_starter": 0.0})
    with pytest.raises(ValueError, match="resolved"):
        AdvisorState.from_dict(
            {**base, "announcement_status": ["undecided", "pass", "pass", "pass"]}
        )


def test_state_rejects_illegal_known_history_and_impossible_score() -> None:
    impossible_history = AdvisorState(
        hand=tuple(
            Card(suit, rank)
            for suit, rank in (
                ("rosen", "A"),
                ("rosen", "K"),
                ("schellen", "A"),
                ("schellen", "K"),
                ("schilten", "A"),
                ("eicheln", "A"),
                ("eicheln", "K"),
                ("eicheln", "Q"),
            )
        ),
        completed_tricks=(
            (
                (1, Card("rosen", "6")),
                (2, Card("eicheln", "6")),
                (3, Card("eicheln", "7")),
                (0, Card("schilten", "6")),
            ),
        ),
        current_trick=(
            (1, Card("schellen", "6")),
            (2, Card("schellen", "7")),
            (3, Card("schellen", "8")),
        ),
        mode="obeabe",
        trump_suit=None,
        leader=1,
        bidding_enabled=False,
        bid_starter=None,
        bid_chooser=None,
        announcement_enabled=False,
    )
    with pytest.raises(ValueError, match="historical play"):
        impossible_history.action_mask(STANDARD_RULES_PROFILE)

    impossible_score = AdvisorState(
        hand=tuple(
            Card(suit, rank)
            for suit, rank in (
                ("rosen", "K"),
                ("schellen", "A"),
                ("schellen", "K"),
                ("schilten", "A"),
                ("schilten", "K"),
                ("eicheln", "A"),
                ("eicheln", "K"),
                ("eicheln", "Q"),
            )
        ),
        completed_tricks=(
            (
                (0, Card("rosen", "A")),
                (1, Card("rosen", "6")),
                (2, Card("rosen", "7")),
                (3, Card("rosen", "8")),
            ),
        ),
        current_trick=(),
        mode="obeabe",
        trump_suit=None,
        leader=0,
        team_points=(0, 0),
        bidding_enabled=False,
        bid_starter=None,
        bid_chooser=None,
        announcement_enabled=False,
    )
    with pytest.raises(ValueError, match="completed-trick total"):
        impossible_score.observation(STANDARD_RULES_PROFILE)


def test_model_loader_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model_final.zip"
    model_path.write_bytes(b"tampered")
    environment = EvaluationEnvironment(
        control_team=True,
        enable_bidding=True,
        enable_weis=True,
        enable_stock=True,
        mode=None,
        trump_suit=None,
        randomize_starter=True,
        reward_scale=0.004,
        terminal_win_bonus=0.5,
        profile=STANDARD_RULES_PROFILE,
    )
    write_manifest(
        tmp_path / MANIFEST_FILENAME,
        {
            "manifest_version": MANIFEST_VERSION,
            "final_model": model_path.name,
            "final_model_sha256": "0" * 64,
            "checkpoints": [],
            "environment": environment.to_manifest_dict(),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_shape": [OBS_SIZE],
            "action_count": ACTION_COUNT,
        },
    )

    with pytest.raises(ValueError, match="does not match manifest"):
        load_advisor(model_path)
