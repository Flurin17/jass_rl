import inspect

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

import rl.reference_verardo as reference_verardo
from core.cards import ALL_CARDS, MODE_OBEABE, MODE_UNEUFE, SUITS, Card
from core.ruleset import RulesetConfig
from env.jass_aec_env import (
    ACTION_COUNT,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    BIDDING_TRUMP_ACTIONS,
    BIDDING_UNEUFE_ACTION,
    OBS_CARD_COUNT,
    OBS_HAND_OFFSET,
    OBS_HISTORY_CARDS_OFFSET,
    OBS_HISTORY_PLAYERS_OFFSET,
    OBS_MODE_OFFSET,
    OBS_SIZE,
    OBS_TRICK_INDEX_OFFSET,
    OBS_TRUMP_SUIT_OFFSET,
)
from rl.reference_verardo import (
    VERARDO_QUALIFICATION_GAMES,
    VERARDO_REFERENCE_COMMIT,
    VERARDO_V1_PROFILE,
    TrumpOnlyPolicy,
    VerardoReferencePolicy,
    run_verardo_benchmark,
    verardo_bid_scores,
    verardo_tournament_config,
)


def _card_index(suit: str, rank: str) -> int:
    return ALL_CARDS.index(Card(suit, rank))


def _observation(hand: list[Card], trump_suit: str | None = None) -> np.ndarray:
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    for card in hand:
        observation[OBS_HAND_OFFSET + ALL_CARDS.index(card)] = 1.0
    observation[OBS_TRICK_INDEX_OFFSET] = 1.0
    if trump_suit is not None:
        observation[OBS_MODE_OFFSET] = 1.0
        observation[OBS_TRUMP_SUIT_OFFSET + SUITS.index(trump_suit)] = 1.0
    return observation


def _mask(*actions: int) -> np.ndarray:
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[list(actions)] = 1
    return mask


def _encode_trick(
    observation: np.ndarray,
    plays: list[tuple[Card, int]],
) -> None:
    for play_slot, (card, relative_player) in enumerate(plays):
        flat_slot = play_slot
        observation[
            OBS_HISTORY_CARDS_OFFSET + flat_slot * OBS_CARD_COUNT + ALL_CARDS.index(card)
        ] = 1.0
        observation[OBS_HISTORY_PLAYERS_OFFSET + flat_slot * 4 + relative_player] = 1.0


def _trump_action(suit: str) -> int:
    return next(action for action, value in BIDDING_TRUMP_ACTIONS.items() if value == suit)


def test_verardo_profile_is_face_value_match100_without_announcements() -> None:
    assert VERARDO_V1_PROFILE.allow_weis is False
    assert VERARDO_V1_PROFILE.allow_stock is False
    assert VERARDO_V1_PROFILE.legal_moves.must_overtrump_when_only_trumps is True
    assert VERARDO_V1_PROFILE.match_bonus == 100
    assert set(VERARDO_V1_PROFILE.contract_factors.values()) == {1}
    assert VERARDO_V1_PROFILE.contract_factor(MODE_OBEABE) == 1
    assert VERARDO_V1_PROFILE.contract_factor(MODE_UNEUFE) == 1
    encoded = VERARDO_V1_PROFILE.to_dict()
    assert encoded["legal_moves"]["must_overtrump_when_only_trumps"] is True
    assert RulesetConfig.from_dict(encoded).to_dict() == encoded


def test_reference_identity_and_standalone_defaults_are_pinned() -> None:
    assert VERARDO_REFERENCE_COMMIT == "4ec7c665ce740fa1d792f653875a63f755551e7c"
    assert verardo_tournament_config().episodes == VERARDO_QUALIFICATION_GAMES
    assert (
        inspect.signature(run_verardo_benchmark).parameters["episodes"].default
        == VERARDO_QUALIFICATION_GAMES
    )
    assert reference_verardo._parse_args(["model.zip"]).episodes == VERARDO_QUALIFICATION_GAMES


def test_bid_scores_use_published_weights_and_weak_first_bid_pushes() -> None:
    hand = [
        Card("schellen", "J"),
        Card("schellen", "A"),
        Card("rosen", "9"),
        Card("schilten", "10"),
        Card("eicheln", "6"),
    ]
    observation = _observation(hand)
    scores = verardo_bid_scores(observation)

    assert scores == {
        "schellen": 11.0,
        "rosen": 5.0,
        "schilten": 0.8,
        "eicheln": 0.5,
    }
    bids = tuple(BIDDING_TRUMP_ACTIONS) + (
        BIDDING_OBEABE_ACTION,
        BIDDING_UNEUFE_ACTION,
        BIDDING_PUSH_ACTION,
    )
    assert VerardoReferencePolicy()(observation, _mask(*bids), "p0") == BIDDING_PUSH_ACTION


@pytest.mark.parametrize(
    ("strong_cards", "expected_suit"),
    [
        ([Card("rosen", rank) for rank in ("J", "8", "6")], "rosen"),
        ([Card("eicheln", rank) for rank in ("9", "8", "7", "6")], "eicheln"),
    ],
)
def test_bid_stopper_selects_best_weighted_trump_and_never_no_trump(
    strong_cards: list[Card], expected_suit: str
) -> None:
    observation = _observation(strong_cards)
    bids = tuple(BIDDING_TRUMP_ACTIONS) + (
        BIDDING_OBEABE_ACTION,
        BIDDING_UNEUFE_ACTION,
        BIDDING_PUSH_ACTION,
    )

    action = VerardoReferencePolicy()(observation, _mask(*bids), "p0")

    assert action == _trump_action(expected_suit)


def test_after_partner_push_weak_hand_must_choose_highest_weighted_suit() -> None:
    hand = [Card("schilten", "A"), Card("rosen", "K"), Card("eicheln", "Q")]
    observation = _observation(hand)
    bids = tuple(BIDDING_TRUMP_ACTIONS) + (
        BIDDING_OBEABE_ACTION,
        BIDDING_UNEUFE_ACTION,
    )

    action = VerardoReferencePolicy()(observation, _mask(*bids), "p2")

    assert action == _trump_action("schilten")


def test_lead_draws_with_j_then_nine_and_otherwise_leads_highest_value_nontrump() -> None:
    policy = VerardoReferencePolicy()
    trump = "schilten"
    jack = Card(trump, "J")
    nine = Card(trump, "9")
    ace = Card("schellen", "A")
    ten = Card("rosen", "10")
    observation = _observation([jack, nine, ace, ten], trump)

    assert policy(
        observation,
        _mask(*(_card_index(card.suit, card.rank) for card in (jack, nine, ace, ten))),
        "p0",
    ) == _card_index(trump, "J")

    observation = _observation([ace, ten, Card(trump, "K")], trump)
    assert policy(
        observation,
        _mask(_card_index("schellen", "A"), _card_index("rosen", "10"), _card_index(trump, "K")),
        "p0",
    ) == _card_index("schellen", "A")


def test_on_trump_lead_plays_highest_legal_trump() -> None:
    trump = "rosen"
    low = Card(trump, "7")
    high = Card(trump, "A")
    observation = _observation([low, high], trump)
    _encode_trick(observation, [(Card(trump, "10"), 3)])

    action = VerardoReferencePolicy()(
        observation,
        _mask(_card_index(trump, "7"), _card_index(trump, "A")),
        "p1",
    )

    assert action == _card_index(trump, "A")


def test_steals_ace_or_ten_with_lowest_trump_when_partner_is_not_winning() -> None:
    trump = "eicheln"
    follow = Card("schellen", "K")
    low_trump = Card(trump, "7")
    high_trump = Card(trump, "A")
    observation = _observation([follow, low_trump, high_trump], trump)
    _encode_trick(observation, [(Card("schellen", "10"), 3)])

    action = VerardoReferencePolicy()(
        observation,
        _mask(*(_card_index(card.suit, card.rank) for card in (follow, low_trump, high_trump))),
        "p1",
    )

    assert action == _card_index(trump, "7")


def test_partner_winning_nontrump_lead_uses_highest_follow() -> None:
    trump = "eicheln"
    queen = Card("schellen", "Q")
    king = Card("schellen", "K")
    low_trump = Card(trump, "7")
    observation = _observation([queen, king, low_trump], trump)
    _encode_trick(
        observation,
        [
            (Card("schellen", "10"), 1),
            (Card("schellen", "A"), 2),
            (Card("schellen", "8"), 3),
        ],
    )

    action = VerardoReferencePolicy()(
        observation,
        _mask(*(_card_index(card.suit, card.rank) for card in (queen, king, low_trump))),
        "p0",
    )

    assert action == _card_index("schellen", "K")


def test_when_void_trumps_low_if_losing_but_discards_low_value_if_partner_wins() -> None:
    trump = "schilten"
    low_trump = Card(trump, "6")
    high_trump = Card(trump, "A")
    discard = Card("rosen", "7")
    observation = _observation([low_trump, high_trump, discard], trump)
    _encode_trick(observation, [(Card("schellen", "K"), 3)])
    mask = _mask(*(_card_index(card.suit, card.rank) for card in (low_trump, high_trump, discard)))

    assert VerardoReferencePolicy()(observation, mask, "p1") == _card_index(trump, "6")

    observation = _observation([low_trump, high_trump, discard], trump)
    _encode_trick(
        observation,
        [(Card("schellen", "K"), 1), (Card("schellen", "A"), 2)],
    )
    assert VerardoReferencePolicy()(observation, mask, "p0") == _card_index("rosen", "7")


def test_policy_validates_canonical_contract_and_uses_only_legal_mask() -> None:
    policy = VerardoReferencePolicy()
    with pytest.raises(ValueError, match="canonical v2 observation"):
        policy(np.zeros(10), _mask(0), "p0")
    with pytest.raises(ValueError, match="canonical action mask"):
        policy(np.zeros(OBS_SIZE), np.ones(10), "p0")


def test_trump_only_adapter_hides_no_trump_bids_without_mutating_mask() -> None:
    class RecordingPolicy:
        def __init__(self) -> None:
            self.mask: np.ndarray | None = None

        def __call__(self, observation, action_mask, agent):
            del observation, agent
            self.mask = action_mask
            return int(np.flatnonzero(action_mask)[0])

    policy = RecordingPolicy()
    adapter = TrumpOnlyPolicy(policy)
    original = _mask(
        *BIDDING_TRUMP_ACTIONS,
        BIDDING_OBEABE_ACTION,
        BIDDING_UNEUFE_ACTION,
        BIDDING_PUSH_ACTION,
    )

    assert adapter(_observation([]), original, "p0") in BIDDING_TRUMP_ACTIONS
    assert policy.mask is not None
    assert policy.mask[BIDDING_OBEABE_ACTION] == 0
    assert policy.mask[BIDDING_UNEUFE_ACTION] == 0
    assert original[BIDDING_OBEABE_ACTION] == 1
    assert original[BIDDING_UNEUFE_ACTION] == 1


def test_trump_only_adapter_error_identifies_wrapped_policy() -> None:
    class IgnoringPolicy:
        def __call__(self, observation, action_mask, agent):
            del observation, action_mask, agent
            return BIDDING_OBEABE_ACTION

    mask = _mask(*BIDDING_TRUMP_ACTIONS, BIDDING_OBEABE_ACTION)
    with pytest.raises(
        ValueError,
        match="wrapped policy selected a bid outside the trump-only benchmark",
    ):
        TrumpOnlyPolicy(IgnoringPolicy())(_observation([]), mask, "p0")


def test_paired_benchmark_helper_is_reproducible_and_uses_verardo_profile() -> None:
    config = verardo_tournament_config(episodes=8, seed=73)
    assert config.modes == (None,)
    assert config.enable_bidding is True
    assert config.enable_weis is False
    assert config.enable_stock is False
    assert config.swap_teams is True
    assert config.profile is VERARDO_V1_PROFILE

    first = run_verardo_benchmark(VerardoReferencePolicy(), episodes=8, seed=73)
    second = run_verardo_benchmark(VerardoReferencePolicy(), episodes=8, seed=73)

    assert [result.team_points for result in first.results] == [
        result.team_points for result in second.results
    ]
    assert first.overall.pairs == 4
    assert first.overall.paired_ties == 4
