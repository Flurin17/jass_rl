from __future__ import annotations

from collections.abc import Iterable

from .cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card

TRUMP_ORDER = ("J", "9", "A", "K", "Q", "10", "8", "7", "6")
OBEABE_ORDER = ("A", "K", "Q", "J", "10", "9", "8", "7", "6")
UNEUFE_ORDER = ("6", "7", "8", "9", "10", "J", "Q", "K", "A")


def _order_scores(order: Iterable[str]) -> dict[str, int]:
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
    trump_suit: str | None = None,
) -> tuple[int, int]:
    if led_suit not in SUITS:
        raise ValueError("led_suit must be a valid suit")
    if mode == MODE_TRUMP:
        if trump_suit not in SUITS:
            raise ValueError("trump_suit is required and must be valid for trump mode")
        if card.suit == trump_suit:
            return (2, _TRUMP_SCORES[card.rank])
        if card.suit == led_suit:
            return (1, _OBEABE_SCORES[card.rank])
        return (0, _OBEABE_SCORES[card.rank])
    if mode == MODE_OBEABE:
        if trump_suit is not None:
            raise ValueError("trump_suit must be None for non-trump modes")
        if card.suit == led_suit:
            return (1, _OBEABE_SCORES[card.rank])
        return (0, _OBEABE_SCORES[card.rank])
    if mode == MODE_UNEUFE:
        if trump_suit is not None:
            raise ValueError("trump_suit must be None for non-trump modes")
        if card.suit == led_suit:
            return (1, _UNEUFE_SCORES[card.rank])
        return (0, _UNEUFE_SCORES[card.rank])
    raise ValueError(f"unknown mode: {mode}")


def beats(
    card_a: Card,
    card_b: Card,
    led_suit: str,
    mode: str,
    trump_suit: str | None = None,
) -> bool:
    return card_strength(card_a, led_suit, mode, trump_suit) > card_strength(
        card_b, led_suit, mode, trump_suit
    )


def winning_card(
    cards: Iterable[Card],
    led_suit: str,
    mode: str,
    trump_suit: str | None = None,
) -> Card:
    card_list = list(cards)
    if not card_list:
        raise ValueError("at least one card is required")
    return max(card_list, key=lambda card: card_strength(card, led_suit, mode, trump_suit))
