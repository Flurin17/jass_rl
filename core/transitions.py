"""Deterministic state transitions for trick play and round scoring.

``GameState.team_points`` always stores profile-scored points.  Raw values are
kept on ``TrickResult`` and ``ScoreEvent`` objects, and terminal raw trick
totals are exposed by ``RoundFinalization``.  This keeps the state semantic the
same during play and after a round finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .rankings import winning_card
from .ruleset import STANDARD_RULES_PROFILE, RulesetConfig
from .scoring import trick_points
from .state import GameState, Trick, TrickResult


class ScoreEventKind(StrEnum):
    """Source of a raw-to-scored point award."""

    TRICK = "trick"
    MATCH = "match"
    WEIS = "weis"
    STOCK = "stock"
    BONUS = "bonus"


@dataclass(frozen=True)
class ScoreEvent:
    """One point award, before and after the active contract factor."""

    kind: ScoreEventKind
    team: int
    raw_points: int
    scored_points: int
    contract_factor: int


@dataclass(frozen=True)
class RoundFinalization:
    """Terminal scoring facts produced exactly once with the ninth trick.

    ``raw_team_points`` contains raw card/trick points only, preserving the
    established 157-point round invariant. ``scored_team_points`` is the full
    public state total and can additionally include Weis, Stöck, and Match.
    """

    raw_team_points: tuple[int, int]
    scored_team_points: tuple[int, int]
    contract_factor: int
    match_team: int | None


@dataclass(frozen=True)
class TrickTransition:
    """Result and score events emitted by resolving one completed trick."""

    trick: TrickResult
    score_events: tuple[ScoreEvent, ...]
    finalization: RoundFinalization | None = None

    @property
    def is_terminal(self) -> bool:
        return self.finalization is not None


def award_raw_points(
    state: GameState,
    team: int,
    raw_points: int,
    *,
    kind: ScoreEventKind = ScoreEventKind.BONUS,
    profile: RulesetConfig | None = None,
) -> ScoreEvent:
    """Apply one factor-scaled award and return its typed score event."""

    if state.is_terminal:
        raise ValueError("round is already complete")
    if isinstance(team, bool) or not isinstance(team, int) or team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    if (
        isinstance(raw_points, bool)
        or not isinstance(raw_points, int)
        or raw_points < 0
    ):
        raise ValueError("raw_points must be a non-negative integer")
    if not isinstance(kind, ScoreEventKind):
        raise TypeError("kind must be a ScoreEventKind")

    active_profile = profile or STANDARD_RULES_PROFILE
    factor = active_profile.contract_factor(state.mode, state.trump_suit)
    event = ScoreEvent(
        kind=kind,
        team=team,
        raw_points=raw_points,
        scored_points=raw_points * factor,
        contract_factor=factor,
    )
    state.team_points[team] += event.scored_points
    return event


def resolve_completed_trick(
    state: GameState,
    *,
    profile: RulesetConfig | None = None,
) -> TrickTransition:
    """Resolve and commit the state's four-card trick exactly once.

    The transition determines the winner, awards factor-scaled trick points,
    advances the leader/index, and clears the active trick.  Resolving the
    ninth trick also awards Match, validates the terminal state, and returns a
    ``RoundFinalization``.
    """

    if state.is_terminal:
        raise ValueError("round is already complete")
    if len(state.trick.plays) != 4:
        raise ValueError("resolving a trick requires exactly four plays")
    if state.trick_index != len(state.completed_tricks):
        raise ValueError("trick_index must match the completed trick count")
    if state.trick_index >= 9:
        raise ValueError("a round cannot contain more than nine tricks")

    expected_players = [(state.leader + offset) % 4 for offset in range(4)]
    actual_players = [player for player, _ in state.trick.plays]
    if actual_players != expected_players:
        raise ValueError("trick plays must follow the current leader")
    if [result.last_trick for result in state.completed_tricks] != [
        False
    ] * state.trick_index:
        raise ValueError("only the ninth trick may be marked as the last trick")

    if any(len(result.plays) != 4 for result in state.completed_tricks):
        raise ValueError("every completed trick must contain four plays")
    completed_cards = [
        card for result in state.completed_tricks for card in result.cards
    ]
    if len(set(completed_cards)) != len(completed_cards):
        raise ValueError("each card may be played only once per round")

    played_cards = set(completed_cards)
    current_cards = state.trick.cards
    if len(set(current_cards)) != 4 or played_cards.intersection(current_cards):
        raise ValueError("each card may be played only once per round")

    led_suit = state.trick.led_suit
    assert led_suit is not None
    winning = winning_card(
        current_cards,
        led_suit,
        state.mode,
        state.trump_suit,
    )
    winner = next(player for player, card in state.trick.plays if card == winning)
    last_trick = state.trick_index == 8
    raw_points = trick_points(
        current_cards,
        state.mode,
        state.trump_suit,
        last_trick=last_trick,
    )
    result = TrickResult(
        plays=list(state.trick.plays),
        winner=winner,
        points=raw_points,
        last_trick=last_trick,
    )

    active_profile = profile or STANDARD_RULES_PROFILE
    factor = active_profile.contract_factor(state.mode, state.trump_suit)
    raw_totals = _raw_team_points((*state.completed_tricks, result))
    if last_trick:
        if any(state.hands):
            raise ValueError("all hands must be empty after the ninth trick")
        if sum(raw_totals) != 157:
            raise RuntimeError("a completed round must contain 157 raw trick points")

    trick_event = _score_event(
        kind=ScoreEventKind.TRICK,
        team=state.team_index(winner),
        raw_points=raw_points,
        factor=factor,
    )
    state.team_points[trick_event.team] += trick_event.scored_points
    state.completed_tricks.append(result)
    state.leader = winner
    state.trick_index += 1
    state.trick = Trick()

    events = [trick_event]
    finalization: RoundFinalization | None = None
    if last_trick:
        match_team = _match_team(state)
        if match_team is not None and active_profile.match_bonus:
            match_event = _score_event(
                kind=ScoreEventKind.MATCH,
                team=match_team,
                raw_points=active_profile.match_bonus,
                factor=factor,
            )
            state.team_points[match_team] += match_event.scored_points
            events.append(match_event)

        state.validate_terminal()
        finalization = RoundFinalization(
            raw_team_points=raw_totals,
            scored_team_points=(state.team_points[0], state.team_points[1]),
            contract_factor=factor,
            match_team=match_team,
        )

    return TrickTransition(
        trick=result,
        score_events=tuple(events),
        finalization=finalization,
    )


def _score_event(
    *,
    kind: ScoreEventKind,
    team: int,
    raw_points: int,
    factor: int,
) -> ScoreEvent:
    return ScoreEvent(
        kind=kind,
        team=team,
        raw_points=raw_points,
        scored_points=raw_points * factor,
        contract_factor=factor,
    )


def _raw_team_points(results: tuple[TrickResult, ...]) -> tuple[int, int]:
    totals = [0, 0]
    for result in results:
        totals[GameState.team_index(result.winner)] += result.points
    return totals[0], totals[1]


def _match_team(state: GameState) -> int | None:
    tricks_won = [0, 0]
    for result in state.completed_tricks:
        tricks_won[state.team_index(result.winner)] += 1
    return next(
        (team for team, count in enumerate(tricks_won) if count == 9),
        None,
    )
