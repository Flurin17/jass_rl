from __future__ import annotations

from typing import Iterable, Optional

from .cards import Card, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE

TRUMP_POINTS = {
    "J": 20,  # Buur
    "9": 14,  # Nell
    "A": 11,
    "K": 4,
    "Q": 3,
    "10": 10,
    "8": 0,
    "7": 0,
    "6": 0,
}

NON_TRUMP_POINTS = {
    "A": 11,
    "K": 4,
    "Q": 3,
    "J": 2,
    "10": 10,
    "9": 0,
    "8": 0,
    "7": 0,
    "6": 0,
}

OBEABE_POINTS = {
    "A": 11,
    "K": 4,
    "Q": 3,
    "J": 2,
    "10": 10,
    "9": 0,
    "8": 8,
    "7": 0,
    "6": 0,
}

UNEUFE_POINTS = {
    "A": 0,
    "K": 4,
    "Q": 3,
    "J": 2,
    "10": 10,
    "9": 0,
    "8": 8,
    "7": 0,
    "6": 11,
}


def card_points(card: Card, mode: str, trump_suit: Optional[str] = None) -> int:
    if mode == MODE_TRUMP:
        if trump_suit is None:
            raise ValueError("trump_suit is required for trump mode")
        if card.suit == trump_suit:
            return TRUMP_POINTS[card.rank]
        return NON_TRUMP_POINTS[card.rank]
    if mode == MODE_OBEABE:
        return OBEABE_POINTS[card.rank]
    if mode == MODE_UNEUFE:
        return UNEUFE_POINTS[card.rank]
    raise ValueError(f"unknown mode: {mode}")


def trick_points(
    cards: Iterable[Card],
    mode: str,
    trump_suit: Optional[str] = None,
    last_trick: bool = False,
) -> int:
    total = sum(card_points(card, mode, trump_suit) for card in cards)
    if last_trick:
        total += 5
    return total
