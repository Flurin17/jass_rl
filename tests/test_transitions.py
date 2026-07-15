import pytest

from core.cards import ALL_CARDS, MODE_OBEABE, Card
from core.rankings import winning_card
from core.ruleset import RulesetConfig
from core.state import GameState, Trick
from core.transitions import (
    ScoreEventKind,
    award_raw_points,
    resolve_completed_trick,
)


def test_completed_trick_emits_raw_and_scored_values_once() -> None:
    profile = RulesetConfig(contract_factors={MODE_OBEABE: 5})
    state = GameState(
        hands=[[], [], [], []],
        mode=MODE_OBEABE,
        trump_suit=None,
        trick=Trick(
            plays=[
                (0, Card("rosen", "A")),
                (1, Card("rosen", "K")),
                (2, Card("rosen", "Q")),
                (3, Card("rosen", "J")),
            ]
        ),
    )

    transition = resolve_completed_trick(state, profile=profile)

    assert transition.trick.winner == 0
    assert transition.trick.points == 20
    assert transition.finalization is None
    assert transition.score_events[0].kind is ScoreEventKind.TRICK
    assert transition.score_events[0].raw_points == 20
    assert transition.score_events[0].scored_points == 100
    assert state.team_points == [100, 0]
    assert state.trick_index == 1
    assert state.trick.plays == []
    assert state.leader == 0

    with pytest.raises(ValueError, match="exactly four plays"):
        resolve_completed_trick(state, profile=profile)
    assert state.team_points == [100, 0]
    assert len(state.completed_tricks) == 1


def test_terminal_transition_applies_match_and_preserves_raw_trick_totals() -> None:
    profile = RulesetConfig()
    state = GameState(hands=[[], [], [], []], mode=MODE_OBEABE, trump_suit=None)
    weis = award_raw_points(
        state,
        0,
        20,
        kind=ScoreEventKind.WEIS,
        profile=profile,
    )
    assert weis.scored_points == 60

    final_transition = None
    for offset in range(0, len(ALL_CARDS), 4):
        cards = list(ALL_CARDS[offset : offset + 4])
        lead = next(
            card
            for card in cards
            if winning_card(
                [card, *(other for other in cards if other != card)],
                card.suit,
                MODE_OBEABE,
            )
            == card
        )
        ordered_cards = [lead, *(card for card in cards if card != lead)]
        state.trick = Trick(plays=list(zip(range(4), ordered_cards, strict=True)))
        final_transition = resolve_completed_trick(state, profile=profile)
        assert final_transition.trick.winner == 0

    assert final_transition is not None
    assert final_transition.is_terminal
    assert final_transition.finalization is not None
    assert final_transition.finalization.raw_team_points == (157, 0)
    assert final_transition.finalization.scored_team_points == (831, 0)
    assert final_transition.finalization.contract_factor == 3
    assert final_transition.finalization.match_team == 0
    assert [event.kind for event in final_transition.score_events] == [
        ScoreEventKind.TRICK,
        ScoreEventKind.MATCH,
    ]
    assert final_transition.score_events[-1].raw_points == 100
    assert final_transition.score_events[-1].scored_points == 300
    assert state.team_points == [831, 0]
    assert state.is_terminal
    state.validate_terminal()

    with pytest.raises(ValueError, match="round is already complete"):
        resolve_completed_trick(state, profile=profile)
    with pytest.raises(ValueError, match="round is already complete"):
        award_raw_points(state, 0, 20, profile=profile)


@pytest.mark.parametrize(
    ("team", "raw_points", "error"),
    [
        (2, 20, "team must be 0 or 1"),
        (0, -1, "raw_points must be a non-negative integer"),
        (False, 20, "team must be 0 or 1"),
        (0, True, "raw_points must be a non-negative integer"),
    ],
)
def test_raw_award_validates_inputs(team: int, raw_points: int, error: str) -> None:
    state = GameState(hands=[[], [], [], []], mode=MODE_OBEABE, trump_suit=None)
    with pytest.raises(ValueError, match=error):
        award_raw_points(state, team, raw_points)
