from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

# Generic suits/ranks; use a consistent ordering for deck creation.
SUITS: Tuple[str, ...] = ("schellen", "rosen", "schilten", "eicheln")
RANKS: Tuple[str, ...] = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")

MODE_TRUMP = "trump"
MODE_OBEABE = "obeabe"
MODE_UNEUFE = "uneufe"


@dataclass(frozen=True)
class Card:
    suit: str
    rank: str

    def __post_init__(self) -> None:
        if self.suit not in SUITS:
            raise ValueError(f"invalid suit: {self.suit}")
        if self.rank not in RANKS:
            raise ValueError(f"invalid rank: {self.rank}")

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"


def make_deck() -> List[Card]:
    return [Card(suit, rank) for suit in SUITS for rank in RANKS]


def iter_deck() -> Iterable[Card]:
    for suit in SUITS:
        for rank in RANKS:
            yield Card(suit, rank)


ALL_CARDS: Tuple[Card, ...] = tuple(iter_deck())
