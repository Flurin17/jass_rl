from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from cli.replay import build_replay, replay_game
from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from core.game import play_round
from core.ruleset import STANDARD_RULES_PROFILE


def _lowest_legal(state, player: int) -> Card:
    return min(state.legal_cards_for(player), key=lambda card: (card.suit, card.rank))


@st.composite
def _contracts(draw: st.DrawFn) -> tuple[str, str | None]:
    mode = draw(st.sampled_from((MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE)))
    trump_suit = draw(st.sampled_from(SUITS)) if mode == MODE_TRUMP else None
    return mode, trump_suit


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), contract=_contracts())
@settings(max_examples=40, deadline=None)
def test_complete_round_invariants_hold_for_generated_deals(
    seed: int, contract: tuple[str, str | None]
) -> None:
    mode, trump_suit = contract
    result = play_round(
        [_lowest_legal] * 4,
        mode=mode,
        trump_suit=trump_suit,
        seed=seed,
        profile=STANDARD_RULES_PROFILE,
    )

    assert result.state.is_terminal
    assert len(result.play_log) == 36
    assert len({card for _, card in result.play_log}) == 36
    assert len(result.state.completed_tricks) == 9
    assert sum(result.raw_team_points) == 157
    expected_raw_total = 157 + (100 if result.match_team is not None else 0)
    assert sum(result.state.team_points) == expected_raw_total * result.contract_factor
    result.state.validate_terminal()


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    leader=st.integers(min_value=0, max_value=3),
    contract=_contracts(),
)
@settings(max_examples=30, deadline=None)
def test_profiled_replay_is_a_lossless_state_transition_log(
    seed: int,
    leader: int,
    contract: tuple[str, str | None],
) -> None:
    mode, trump_suit = contract
    result = play_round(
        [_lowest_legal] * 4,
        mode=mode,
        trump_suit=trump_suit,
        seed=seed,
        leader=leader,
        profile=STANDARD_RULES_PROFILE,
    )

    replay = build_replay(
        result,
        seed=seed,
        mode=mode,
        trump_suit=trump_suit,
        leader=leader,
    )
    replayed = replay_game(replay)

    assert replayed.team_points == result.state.team_points
    assert replayed.completed_tricks == result.state.completed_tricks
    assert replayed.leader == result.state.leader
    assert replayed.is_terminal
