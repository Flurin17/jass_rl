from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from .cards import Card, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE

TRUMP_ORDER = ("J", "9", "A", "K", "Q", "10", "8", "7", "6")
OBEABE_ORDER = ("A", "K", "Q", "J", "10", "9", "8", "7", "6")
UNEUFE_ORDER = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")


def _order_scores(order: Iterable[str]) -> Dict[str, int]:
    order_list = list(order)
    size = len(order_list)
    return {rank: size - idx for idx, rank in enumerate(order_list)}


_TRUMP_SCORES = _order_scores(TRUMP_ORDER)
_OBEABE_SCORES = _order_scores(OBEABE_ORDER)
_UNEUFE_SCORES = _order_scores(UNEUFE_ORDER)


def card_strength(
    card: Card,
    led_suit: str,
    mode: str,
    trump_suit: Optional[str] = None,
) -> Tuple[int, int]:
    if mode == MODE_TRUMP:
        if trump_suit is None:
            raise ValueError("trump_suit is required for trump mode")
        if card.suit == trump_suit:
            return (2, _TRUMP_SCORES[card.rank])
        if card.suit == led_suit:
            return (1, _OBEABE_SCORES[card.rank])
        return (0, _OBEABE_SCORES[card.rank])
    if mode == MODE_OBEABE:
        if card.suit == led_suit:
            return (1, _OBEABE_SCORES[card.rank])
        return (0, _OBEABE_SCORES[card.rank])
    if mode == MODE_UNEUFE:
        if card.suit == led_suit:
            return (1, _UNEUFE_SCORES[card.rank])
        return (0, _UNEUFE_SCORES[card.rank])
    raise ValueError(f"unknown mode: {mode}")


def beats(
    card_a: Card,
    card_b: Card,
    led_suit: str,
    mode: str,
    trump_suit: Optional[str] = None,
) -> bool:
    return card_strength(card_a, led_suit, mode, trump_suit) > card_strength(
        card_b, led_suit, mode, trump_suit
    )


def winning_card(
    cards: Iterable[Card],
    led_suit: str,
    mode: str,
    trump_suit: Optional[str] = None,
) -> Card:
    return max(cards, key=lambda card: card_strength(card, led_suit, mode, trump_suit))
