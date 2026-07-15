import pytest


def _skip_if_missing() -> None:
    try:
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
        import pettingzoo  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

import numpy as np

from core.cards import ALL_CARDS, Card
from env.jass_aec_env import (
    ACTION_COUNT,
    ANNOUNCE_ACTION,
    BIDDING_TRUMP_ACTIONS,
    OBS_CARD_COUNT,
    OBS_HAND_OFFSET,
    OBS_HISTORY_CARDS_OFFSET,
    OBS_HISTORY_PLAYERS_OFFSET,
    OBS_MODE_OFFSET,
    OBS_PLAYS_PER_TRICK,
    OBS_SIZE,
    OBS_TRICK_INDEX_OFFSET,
    PASS_ACTION,
)
from rl.baselines import (
    SeededRandomPolicy,
    StrategicHeuristicPolicy,
    as_aec_policy,
)


def _card_index(suit: str, rank: str) -> int:
    return ALL_CARDS.index(Card(suit, rank))


def _observation(hand: list[Card]) -> np.ndarray:
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    for card in hand:
        observation[OBS_HAND_OFFSET + ALL_CARDS.index(card)] = 1.0
    return observation


def _mask(*actions: int) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[list(actions)] = 1
    return mask


def _encode_current_trick(
    observation: np.ndarray,
    plays: list[tuple[Card, int]],
    trick_index: int = 0,
) -> None:
    observation[OBS_TRICK_INDEX_OFFSET + trick_index] = 1.0
    for play_slot, (card, relative_player) in enumerate(plays):
        flat_slot = trick_index * OBS_PLAYS_PER_TRICK + play_slot
        observation[
            OBS_HISTORY_CARDS_OFFSET + flat_slot * OBS_CARD_COUNT + ALL_CARDS.index(card)
        ] = 1.0
        observation[OBS_HISTORY_PLAYERS_OFFSET + flat_slot * 4 + relative_player] = 1.0


def test_seeded_random_policy_is_reproducible_and_legal() -> None:
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = _mask(2, 9, 17)
    left = SeededRandomPolicy(123)
    right = SeededRandomPolicy(123)

    left_actions = [left(observation, mask, "p0") for _ in range(20)]
    right_actions = [right(observation, mask, "p0") for _ in range(20)]

    assert left_actions == right_actions
    assert set(left_actions) <= {2, 9, 17}
    left.reset(123)
    assert [left(observation, mask, "p0") for _ in range(20)] == left_actions


def test_strategic_policy_bids_a_strong_trump_suit() -> None:
    hand = [
        Card("schilten", "J"),
        Card("schilten", "9"),
        Card("schilten", "A"),
        Card("schilten", "K"),
        Card("schellen", "6"),
        Card("schellen", "8"),
        Card("rosen", "7"),
        Card("rosen", "10"),
        Card("eicheln", "Q"),
    ]
    observation = _observation(hand)
    legal_bids = tuple(BIDDING_TRUMP_ACTIONS) + (40, 41, 42)

    action = StrategicHeuristicPolicy()(observation, _mask(*legal_bids), "p0")

    assert action == next(
        action for action, suit in BIDDING_TRUMP_ACTIONS.items() if suit == "schilten"
    )


def test_strategic_policy_announces_only_when_hand_has_weis() -> None:
    policy = StrategicHeuristicPolicy()
    with_weis = _observation([Card("schellen", rank) for rank in ("6", "7", "8")])
    without_weis = _observation([Card("schellen", "6"), Card("rosen", "8"), Card("eicheln", "10")])
    announce_mask = _mask(ANNOUNCE_ACTION, PASS_ACTION)

    assert policy(with_weis, announce_mask, "p0") == ANNOUNCE_ACTION
    assert policy(without_weis, announce_mask, "p0") == PASS_ACTION


def test_strategic_policy_uses_cheapest_card_that_wins() -> None:
    queen = Card("schellen", "Q")
    ace = Card("schellen", "A")
    observation = _observation([queen, ace])
    observation[OBS_MODE_OFFSET + 1] = 1.0  # Obeabe
    _encode_current_trick(observation, [(Card("schellen", "10"), 3)])
    mask = _mask(ALL_CARDS.index(queen), ALL_CARDS.index(ace))

    action = StrategicHeuristicPolicy()(observation, mask, "p1")

    assert action == ALL_CARDS.index(queen)


def test_strategic_policy_smears_points_into_secured_partner_trick() -> None:
    ten = Card("schellen", "10")
    king = Card("schellen", "K")
    observation = _observation([ten, king])
    observation[OBS_MODE_OFFSET + 1] = 1.0  # Obeabe
    trick = [
        (Card("schellen", "6"), 1),  # p3 is left of acting p2
        (Card("schellen", "A"), 2),  # p0 is partner of acting p2
        (Card("schellen", "7"), 3),  # p1 is right of acting p2
    ]
    _encode_current_trick(observation, trick)
    mask = _mask(ALL_CARDS.index(ten), ALL_CARDS.index(king))

    action = StrategicHeuristicPolicy()(observation, mask, "p2")

    assert action == ALL_CARDS.index(ten)


def test_aec_adapter_does_not_read_private_environment_state() -> None:
    observation = _observation([Card("schellen", rank) for rank in ("6", "7", "8")])
    visible = {
        "observation": observation,
        "action_mask": _mask(ANNOUNCE_ACTION, PASS_ACTION),
    }

    class PublicOnlyEnv:
        @property
        def state(self):
            raise AssertionError("private state was accessed")

        def observe(self, agent: str):
            assert agent == "p0"
            return visible

    action = as_aec_policy(StrategicHeuristicPolicy())(PublicOnlyEnv(), "p0")

    assert action == ANNOUNCE_ACTION


def test_strategic_policy_rejects_empty_action_mask() -> None:
    with pytest.raises(ValueError, match="without a legal action"):
        StrategicHeuristicPolicy()(
            np.zeros(OBS_SIZE, dtype=np.float32),
            np.zeros(ACTION_COUNT, dtype=np.int8),
            "p0",
        )
