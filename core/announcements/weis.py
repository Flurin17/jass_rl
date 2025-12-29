from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..cards import Card

SEQUENCE_ORDER = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")
SEQUENCE_INDEX = {rank: idx for idx, rank in enumerate(SEQUENCE_ORDER)}

FOUR_KIND_ORDER = ("J", "9", "A", "K", "Q", "10", "8", "7", "6")
FOUR_KIND_INDEX = {rank: idx for idx, rank in enumerate(FOUR_KIND_ORDER)}


@dataclass(frozen=True)
class WeisCombo:
    kind: str  # "sequence" or "four_of_a_kind"
    length: int
    rank: str
    suit: Optional[str]
    points: int


def _sequence_points(length: int) -> int:
    if length == 3:
        return 20
    if length == 4:
        return 50
    return 100


def _four_kind_points(rank: str) -> int:
    if rank == "J":
        return 200
    if rank == "9":
        return 150
    if rank in ("A", "K", "Q", "10"):
        return 100
    raise ValueError("invalid rank for four of a kind")


def _find_sequences(cards: Iterable[Card]) -> List[WeisCombo]:
    by_suit: Dict[str, List[Card]] = {}
    for card in cards:
        by_suit.setdefault(card.suit, []).append(card)

    combos: List[WeisCombo] = []
    for suit, suit_cards in by_suit.items():
        ranks = sorted({card.rank for card in suit_cards}, key=lambda r: SEQUENCE_INDEX[r])
        if not ranks:
            continue
        current: List[str] = [ranks[0]]
        for rank in ranks[1:]:
            if SEQUENCE_INDEX[rank] == SEQUENCE_INDEX[current[-1]] + 1:
                current.append(rank)
            else:
                combos.extend(_sequence_combos_from_run(current, suit))
                current = [rank]
        combos.extend(_sequence_combos_from_run(current, suit))
    return combos


def _sequence_combos_from_run(run: List[str], suit: str) -> List[WeisCombo]:
    combos: List[WeisCombo] = []
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


def _find_four_of_a_kind(cards: Iterable[Card]) -> List[WeisCombo]:
    counts: Dict[str, int] = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1

    combos: List[WeisCombo] = []
    for rank, count in counts.items():
        if count == 4 and rank in ("A", "K", "Q", "J", "10", "9"):
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


def find_weis(cards: Iterable[Card]) -> List[WeisCombo]:
    combos = _find_sequences(cards)
    combos.extend(_find_four_of_a_kind(cards))
    return combos


def best_weis(combos: Iterable[WeisCombo]) -> Optional[WeisCombo]:
    best: Optional[WeisCombo] = None
    for combo in combos:
        if best is None or compare_weis(combo, best) > 0:
            best = combo
    return best


def compare_weis(a: WeisCombo, b: WeisCombo) -> int:
    if a.points != b.points:
        return 1 if a.points > b.points else -1
    if a.kind != b.kind:
        priority = {"four_of_a_kind": 2, "sequence": 1}
        return 1 if priority[a.kind] > priority[b.kind] else -1
    if a.kind == "sequence":
        if a.length != b.length:
            return 1 if a.length > b.length else -1
        if a.rank != b.rank:
            return 1 if SEQUENCE_INDEX[a.rank] > SEQUENCE_INDEX[b.rank] else -1
    if a.kind == "four_of_a_kind" and a.rank != b.rank:
        return 1 if FOUR_KIND_INDEX[a.rank] < FOUR_KIND_INDEX[b.rank] else -1
    return 0


def resolve_weis(
    team_a_cards: Iterable[Card],
    team_b_cards: Iterable[Card],
) -> Tuple[int, int, Optional[int], Optional[WeisCombo]]:
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
