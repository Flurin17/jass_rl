import pytest

from core.cards import MODE_OBEABE, MODE_TRUMP, RANKS, SUITS, Card
from core.game import play_round
from core.ruleset import LEGACY_UNMULTIPLIED_PROFILE, RulesetConfig
from core.scoring import trick_points
from core.state import GameState, Trick


def _card_sort_key(card: Card) -> tuple[int, int]:
    return (SUITS.index(card.suit), RANKS.index(card.rank))


def _policy_lowest(state, player):
    legal = state.legal_cards_for(player)
    return sorted(legal, key=_card_sort_key)[0]


def test_round_is_deterministic_with_seed() -> None:
    policies = [_policy_lowest] * 4
    result_a = play_round(policies, MODE_TRUMP, trump_suit="rosen", seed=123)
    result_b = play_round(policies, MODE_TRUMP, trump_suit="rosen", seed=123)
    assert result_a.play_log == result_b.play_log
    assert result_a.state.team_points == result_b.state.team_points
    assert result_a.raw_team_points == result_b.raw_team_points


def test_round_completes_with_raw_points_and_contract_factor() -> None:
    policies = [_policy_lowest] * 4
    profile = RulesetConfig()
    result = play_round(
        policies,
        MODE_TRUMP,
        trump_suit="schilten",
        seed=7,
        profile=profile,
    )
    state = result.state

    assert state.is_terminal
    state.validate_terminal()
    assert state.trick_index == 9
    assert state.trick.plays == []
    assert state.leader == state.completed_tricks[-1].winner
    assert all(len(hand) == 0 for hand in state.hands)
    assert len(state.completed_tricks) == 9
    assert sum(result.raw_team_points) == 157
    assert result.contract_factor == 2
    assert result.rules_version == profile.version

    base_points = sum(
        trick_points(trick.cards, state.mode, state.trump_suit, last_trick=False)
        for trick in state.completed_tricks
    )
    assert base_points + 5 == sum(result.raw_team_points)
    match_bonus = profile.match_bonus if result.match_team is not None else 0
    expected_total = 2 * (157 + match_bonus)
    assert sum(state.team_points) == expected_total


def test_legacy_profile_keeps_unmultiplied_round_scores() -> None:
    result = play_round(
        [_policy_lowest] * 4,
        MODE_OBEABE,
        seed=11,
        profile=LEGACY_UNMULTIPLIED_PROFILE,
    )
    assert sum(result.state.team_points) == 157
    assert result.contract_factor == 1


def test_match_bonus_is_applied_before_factor(monkeypatch) -> None:
    class NoShuffleRandom:
        def __init__(self, seed=None) -> None:
            self.seed = seed

        def shuffle(self, deck) -> None:
            return None

    monkeypatch.setattr("core.game.random.Random", NoShuffleRandom)
    result = play_round(
        [_policy_lowest] * 4,
        MODE_TRUMP,
        trump_suit="schellen",
        seed=1,
    )

    assert result.raw_team_points == (157, 0)
    assert result.match_team == 0
    assert result.state.team_points == [514, 0]


@pytest.mark.parametrize(
    ("mode", "trump_suit", "leader"),
    [
        ("invalid", None, 0),
        (MODE_TRUMP, None, 0),
        (MODE_TRUMP, "invalid", 0),
        (MODE_OBEABE, "rosen", 0),
        (MODE_OBEABE, None, -1),
        (MODE_OBEABE, None, 4),
    ],
)
def test_play_round_rejects_invalid_contract_or_leader(
    mode: str, trump_suit: str | None, leader: int
) -> None:
    with pytest.raises(ValueError):
        play_round([_policy_lowest] * 4, mode, trump_suit, leader=leader)


def test_game_state_validates_contract_and_leader() -> None:
    hands = [[], [], [], []]
    with pytest.raises(ValueError):
        GameState(hands=hands, mode=MODE_TRUMP, trump_suit=None)
    with pytest.raises(ValueError):
        GameState(hands=hands, mode=MODE_OBEABE, trump_suit="rosen")
    with pytest.raises(ValueError):
        GameState(hands=hands, mode=MODE_OBEABE, trump_suit=None, leader=4)


def test_completed_trick_and_terminal_round_reject_more_play() -> None:
    trick = Trick(
        plays=[
            (0, Card("schellen", "6")),
            (1, Card("schellen", "7")),
            (2, Card("schellen", "8")),
            (3, Card("schellen", "9")),
        ]
    )
    state = GameState(
        hands=[[Card("rosen", "6")], [], [], []],
        mode=MODE_OBEABE,
        trump_suit=None,
        trick=trick,
    )
    assert state.legal_cards_for(0) == []
    with pytest.raises(ValueError, match="trick is already complete"):
        state.play_card(0, Card("rosen", "6"))

    terminal = play_round(
        [_policy_lowest] * 4,
        MODE_TRUMP,
        trump_suit="eicheln",
        seed=3,
    ).state
    with pytest.raises(ValueError, match="round is already complete"):
        terminal.play_card(terminal.leader, Card("rosen", "6"))
