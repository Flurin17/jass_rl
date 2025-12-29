from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .cards import Card, MODE_TRUMP
from .rankings import beats, winning_card


@dataclass(frozen=True)
class RuleSet:
    must_follow_suit: bool = True
    must_trump: bool = True
    must_overtrump: bool = True
    # If False and partner is currently winning, trumping is not mandatory.
    must_trump_if_partner_winning: bool = False


def _apply_overtrump(
    trumps_in_hand: List[Card],
    trick: List[Card],
    trump_suit: str,
) -> List[Card]:
    trick_trumps = [card for card in trick if card.suit == trump_suit]
    if not trick_trumps:
        return trumps_in_hand
    highest_trump = winning_card(
        trick_trumps, led_suit=trump_suit, mode=MODE_TRUMP, trump_suit=trump_suit
    )
    overtrumps = [
        card
        for card in trumps_in_hand
        if beats(card, highest_trump, led_suit=trump_suit, mode=MODE_TRUMP, trump_suit=trump_suit)
    ]
    return overtrumps if overtrumps else trumps_in_hand


def legal_cards(
    hand: Iterable[Card],
    trick: Iterable[Card],
    mode: str,
    trump_suit: Optional[str] = None,
    partner_is_winning: bool = False,
    ruleset: Optional[RuleSet] = None,
) -> List[Card]:
    rules = ruleset or RuleSet()
    hand_list = list(hand)
    trick_list = list(trick)

    if not hand_list:
        return []
    if not trick_list:
        return hand_list

    led_suit = trick_list[0].suit

    if rules.must_follow_suit:
        suited = [card for card in hand_list if card.suit == led_suit]
        if suited:
            if mode == MODE_TRUMP and led_suit == trump_suit and rules.must_overtrump:
                return _apply_overtrump(suited, trick_list, trump_suit)
            return suited

    if mode == MODE_TRUMP:
        if trump_suit is None:
            raise ValueError("trump_suit is required for trump mode")
        if rules.must_trump and (not partner_is_winning or rules.must_trump_if_partner_winning):
            trumps = [card for card in hand_list if card.suit == trump_suit]
            if trumps:
                if rules.must_overtrump:
                    return _apply_overtrump(trumps, trick_list, trump_suit)
                return trumps

    return hand_list
