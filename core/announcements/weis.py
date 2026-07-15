from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..cards import MODE_UNEUFE, Card

SEQUENCE_ORDER = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
SEQUENCE_INDEX = {rank: idx for idx, rank in enumerate(SEQUENCE_ORDER)}

FOUR_KIND_ORDER = ("J", "9", "A", "K", "Q", "10", "8", "7", "6")
FOUR_KIND_INDEX = {rank: idx for idx, rank in enumerate(FOUR_KIND_ORDER)}


@dataclass(frozen=True)
class WeisCombo:
    kind: str  # "sequence" or "four_of_a_kind"
    length: int
    rank: str
    suit: str | None
    points: int


def _sequence_points(length: int) -> int:
    if length == 3:
        return 20
    if length == 4:
        return 50
    if length == 5:
        return 100
    if length == 6:
        return 150
    if length == 7:
        return 200
    if length == 8:
        return 250
    if length == 9:
        return 300
    raise ValueError("invalid sequence length")


def _four_kind_points(rank: str) -> int:
    if rank == "J":
        return 200
    if rank == "9":
        return 150
    # Any other rank counts 100 (including 6/7/8).
    return 100


def _find_sequences(cards: Iterable[Card]) -> list[WeisCombo]:
    by_suit: dict[str, list[Card]] = {}
    for card in cards:
        by_suit.setdefault(card.suit, []).append(card)

    combos: list[WeisCombo] = []
    for suit, suit_cards in by_suit.items():
        ranks = sorted({card.rank for card in suit_cards}, key=lambda r: SEQUENCE_INDEX[r])
        if not ranks:
            continue
        current: list[str] = [ranks[0]]
        for rank in ranks[1:]:
            if SEQUENCE_INDEX[rank] == SEQUENCE_INDEX[current[-1]] + 1:
                current.append(rank)
            else:
                combos.extend(_sequence_combos_from_run(current, suit))
                current = [rank]
        combos.extend(_sequence_combos_from_run(current, suit))
    return combos


def _sequence_combos_from_run(run: list[str], suit: str) -> list[WeisCombo]:
    combos: list[WeisCombo] = []
    if len(run) < 3:
        return combos
    length = len(run)
    top_rank = run[-1]
    points = _sequence_points(length)
    combos.append(
        WeisCombo(
            kind="sequence",
            length=length,
            rank=top_rank,
            suit=suit,
            points=points,
        )
    )
    return combos


def _find_four_of_a_kind(cards: Iterable[Card]) -> list[WeisCombo]:
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1

    combos: list[WeisCombo] = []
    for rank, count in counts.items():
        if count == 4:
            combos.append(
                WeisCombo(
                    kind="four_of_a_kind",
                    length=4,
                    rank=rank,
                    suit=None,
                    points=_four_kind_points(rank),
                )
            )
    return combos


def find_weis(cards: Iterable[Card]) -> list[WeisCombo]:
    # ``Iterable`` includes one-shot generators.  Materialize once so both
    # detectors inspect the same hand.
    hand = list(cards)
    combos = _find_sequences(hand)
    combos.extend(_find_four_of_a_kind(hand))
    return combos


def best_weis(
    combos: Iterable[WeisCombo],
    *,
    mode: str | None = None,
    trump_suit: str | None = None,
) -> WeisCombo | None:
    best: WeisCombo | None = None
    for combo in combos:
        if best is None or compare_weis(combo, best, mode=mode, trump_suit=trump_suit) > 0:
            best = combo
    return best


def compare_weis(
    a: WeisCombo,
    b: WeisCombo,
    *,
    mode: str | None = None,
    trump_suit: str | None = None,
) -> int:
    if a.points != b.points:
        return 1 if a.points > b.points else -1

    # If equal points, compare by type-specific strength.
    if a.kind == b.kind == "sequence":
        if a.length != b.length:
            return 1 if a.length > b.length else -1
        if a.rank != b.rank:
            if mode == MODE_UNEUFE:
                return 1 if SEQUENCE_INDEX[a.rank] < SEQUENCE_INDEX[b.rank] else -1
            return 1 if SEQUENCE_INDEX[a.rank] > SEQUENCE_INDEX[b.rank] else -1
        # If still tied: the sequence in trump wins.
        if trump_suit is not None and a.suit != b.suit:
            if a.suit == trump_suit and b.suit != trump_suit:
                return 1
            if b.suit == trump_suit and a.suit != trump_suit:
                return -1
        return 0

    if a.kind == b.kind == "four_of_a_kind":
        if a.rank != b.rank:
            if mode == MODE_UNEUFE:
                # Equal-scoring fours use the reversed card order in Undenufe.
                return 1 if SEQUENCE_INDEX[a.rank] < SEQUENCE_INDEX[b.rank] else -1
            # Higher four-of-a-kind wins, using J > 9 > A > K > Q > 10 > 8 > 7 > 6.
            return 1 if FOUR_KIND_INDEX[a.rank] < FOUR_KIND_INDEX[b.rank] else -1
        return 0

    # For equal points across different types, the site rule text implies sequence tie-breakers,
    # but this situation only happens at 100 points (sequence of 5 vs four-of-a-kind).
    # Prefer the sequence, then fall back to the other.
    if a.kind != b.kind:
        priority = {"sequence": 2, "four_of_a_kind": 1}
        return 1 if priority[a.kind] > priority[b.kind] else -1

    return 0


def resolve_weis(
    team_a_cards: Iterable[Card],
    team_b_cards: Iterable[Card],
) -> tuple[int, int, int | None, WeisCombo | None]:
    combos_a = find_weis(team_a_cards)
    combos_b = find_weis(team_b_cards)

    best_a = best_weis(combos_a)
    best_b = best_weis(combos_b)

    if best_a is None and best_b is None:
        return 0, 0, None, None
    if best_a is None:
        return 0, sum(combo.points for combo in combos_b), 1, best_b
    if best_b is None:
        return sum(combo.points for combo in combos_a), 0, 0, best_a

    comparison = compare_weis(best_a, best_b)
    if comparison == 0:
        return 0, 0, None, None
    if comparison > 0:
        return sum(combo.points for combo in combos_a), 0, 0, best_a
    return 0, sum(combo.points for combo in combos_b), 1, best_b


def resolve_weis_by_player(
    announced_by_player: dict[int, list[Card]],
    announce_order: list[int],
    *,
    mode: str,
    trump_suit: str | None = None,
) -> tuple[int, int, int | None, WeisCombo | None]:
    """Resolve Weis scoring using per-player announced cards.

    Returns (points_team_a, points_team_b, winner_team_index, best_combo).
    Tie-break for equal best is the earlier announcement in announce_order.
    """

    combos_by_player: dict[int, list[WeisCombo]] = {}
    best_by_player: dict[int, WeisCombo] = {}
    for player in range(4):
        combos = find_weis(announced_by_player.get(player, []))
        combos_by_player[player] = combos
        best = best_weis(combos, mode=mode, trump_suit=trump_suit)
        if best is not None:
            best_by_player[player] = best

    if not best_by_player:
        return 0, 0, None, None

    # Find the globally best Weis combo (by comparison rules).
    best_overall: WeisCombo | None = None
    for combo in best_by_player.values():
        if (
            best_overall is None
            or compare_weis(combo, best_overall, mode=mode, trump_suit=trump_suit) > 0
        ):
            best_overall = combo

    assert best_overall is not None

    # Candidates whose best equals the global best.
    candidates = [
        player
        for player, combo in best_by_player.items()
        if compare_weis(combo, best_overall, mode=mode, trump_suit=trump_suit) == 0
    ]
    if not candidates:
        return 0, 0, None, None

    order_index = {player: idx for idx, player in enumerate(announce_order)}
    winner_player = min(candidates, key=lambda p: order_index.get(p, 10**9))
    winner_team = 0 if winner_player % 2 == 0 else 1

    def _team_points(team_index: int) -> int:
        total = 0
        for player in (0, 2) if team_index == 0 else (1, 3):
            total += sum(combo.points for combo in combos_by_player[player])
        return total

    points_a = _team_points(0) if winner_team == 0 else 0
    points_b = _team_points(1) if winner_team == 1 else 0
    return points_a, points_b, winner_team, best_overall
