from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from .rankings import beats, winning_card


@dataclass(frozen=True)
class RuleSet:
    """Options for card-play obligations.

    The defaults implement the standard Schieber rules documented by Pagat.
    The original fields remain available for callers that use a house rule.
    ``must_overtrump`` applies only after a non-trump lead has already been
    ruffed; a trump lead never carries an overtrumping obligation.
    """

    must_follow_suit: bool = True
    must_trump: bool = False
    must_overtrump: bool = True
    # Relevant only to the optional ``must_trump`` house rule.
    must_trump_if_partner_winning: bool = False
    allow_trump_on_non_trump_lead: bool = True
    allow_puur_withhold: bool = True
    # The standard Swiss profile waives overtrumping when the player's entire
    # hand is trump.  The pinned Verardo engine is stricter: if that hand
    # contains an overtrump, the player must use one; an undertrump is available
    # only when no overtrump exists.
    must_overtrump_when_only_trumps: bool = False


def _validate_contract(mode: str, trump_suit: str | None) -> None:
    if mode not in (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE):
        raise ValueError(f"unknown mode: {mode}")
    if mode == MODE_TRUMP:
        if trump_suit not in SUITS:
            raise ValueError("trump_suit is required and must be a valid suit")
    elif trump_suit is not None:
        raise ValueError("trump_suit must be None for non-trump modes")


def _overtrumps(
    trumps_in_hand: list[Card],
    trick: list[Card],
    trump_suit: str,
) -> list[Card]:
    trick_trumps = [card for card in trick if card.suit == trump_suit]
    if not trick_trumps:
        return trumps_in_hand
    highest_trump = winning_card(
        trick_trumps,
        led_suit=trump_suit,
        mode=MODE_TRUMP,
        trump_suit=trump_suit,
    )
    return [
        card
        for card in trumps_in_hand
        if beats(
            card,
            highest_trump,
            led_suit=trump_suit,
            mode=MODE_TRUMP,
            trump_suit=trump_suit,
        )
    ]


def _legal_after_non_trump_lead(
    hand: list[Card],
    trick: list[Card],
    trump_suit: str,
    partner_is_winning: bool,
    rules: RuleSet,
) -> list[Card]:
    led_suit = trick[0].suit
    follows = [card for card in hand if card.suit == led_suit]
    trumps = [card for card in hand if card.suit == trump_suit]
    non_trumps = [card for card in hand if card.suit != trump_suit]

    # Undertrumping is the one restriction that also applies when a player is
    # void.  Standard Schieber waives it when every remaining card is a trump.
    # The pinned Verardo profile instead keeps any available overtrump
    # mandatory, falling back to all trumps only when no overtrump exists.
    permitted_trumps = trumps
    if rules.must_overtrump and any(card.suit == trump_suit for card in trick):
        only_trumps = len(trumps) == len(hand)
        if not only_trumps or rules.must_overtrump_when_only_trumps:
            overtrumps = _overtrumps(trumps, trick, trump_suit)
            permitted_trumps = overtrumps or (trumps if only_trumps else [])

    if follows and rules.must_follow_suit:
        if not rules.allow_trump_on_non_trump_lead:
            return follows
        permitted = set(follows) | set(permitted_trumps)
        return [card for card in hand if card in permitted]

    trump_is_mandatory = rules.must_trump and (
        not partner_is_winning or rules.must_trump_if_partner_winning
    )
    if trump_is_mandatory and permitted_trumps:
        permitted = set(permitted_trumps)
        return [card for card in hand if card in permitted]

    # A void player may discard freely, except that an undertrump remains
    # illegal.  With must_follow_suit disabled, the same applies to off-suit
    # non-trumps even if the led suit is held.
    permitted = set(non_trumps) | set(permitted_trumps)
    return [card for card in hand if card in permitted]


def legal_cards(
    hand: Iterable[Card],
    trick: Iterable[Card],
    mode: str,
    trump_suit: str | None = None,
    partner_is_winning: bool = False,
    ruleset: RuleSet | None = None,
) -> list[Card]:
    """Return the cards that may legally be played from ``hand``.

    ``trick`` contains the cards already played to the current trick and must
    therefore contain at most three cards.
    """

    _validate_contract(mode, trump_suit)
    rules = ruleset or RuleSet()
    hand_list = list(hand)
    trick_list = list(trick)

    if len(trick_list) > 3:
        raise ValueError("cannot choose a card for a completed trick")
    if not hand_list:
        return []
    if not trick_list:
        return hand_list

    led_suit = trick_list[0].suit

    if mode != MODE_TRUMP:
        if not rules.must_follow_suit:
            return hand_list
        follows = [card for card in hand_list if card.suit == led_suit]
        return follows or hand_list

    assert trump_suit is not None  # established by _validate_contract
    if led_suit != trump_suit:
        return _legal_after_non_trump_lead(
            hand_list,
            trick_list,
            trump_suit,
            partner_is_winning,
            rules,
        )

    if not rules.must_follow_suit:
        return hand_list

    trumps = [card for card in hand_list if card.suit == trump_suit]
    if not trumps:
        return hand_list

    # The Puur (trump J/Under) never has to be played as the player's sole
    # trump.  It remains legal; the player simply gains the option to discard.
    if (
        rules.allow_puur_withhold
        and len(trumps) == 1
        and trumps[0].rank == "J"
        and len(hand_list) > 1
    ):
        return hand_list

    # Unlike a ruff of a non-trump lead, a trump lead has no requirement to
    # beat a trump that is already in the trick.
    return trumps
