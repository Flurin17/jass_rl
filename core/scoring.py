from __future__ import annotations

from collections.abc import Iterable

from .cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, Card
from .ruleset import STANDARD_RULES_PROFILE, RulesetConfig

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


def card_points(card: Card, mode: str, trump_suit: str | None = None) -> int:
    if mode == MODE_TRUMP:
        if trump_suit is None:
            raise ValueError("trump_suit is required for trump mode")
        if card.suit == trump_suit:
            return TRUMP_POINTS[card.rank]
        return NON_TRUMP_POINTS[card.rank]
    if mode == MODE_OBEABE:
        if trump_suit is not None:
            raise ValueError("trump_suit must be None for non-trump modes")
        return OBEABE_POINTS[card.rank]
    if mode == MODE_UNEUFE:
        if trump_suit is not None:
            raise ValueError("trump_suit must be None for non-trump modes")
        return UNEUFE_POINTS[card.rank]
    raise ValueError(f"unknown mode: {mode}")


def trick_points(
    cards: Iterable[Card],
    mode: str,
    trump_suit: str | None = None,
    last_trick: bool = False,
) -> int:
    """Return raw card points, including the unmultiplied last-trick bonus."""

    total = sum(card_points(card, mode, trump_suit) for card in cards)
    if last_trick:
        total += 5
    return total


def round_score(
    raw_trick_points: int,
    mode: str,
    trump_suit: str | None = None,
    *,
    won_all_tricks: bool = False,
    profile: RulesetConfig | None = None,
) -> int:
    """Score one team's raw trick points under a rules profile.

    The Match bonus is added only when that team won all nine tricks, and the
    contract factor is applied after the bonus.
    """

    if (
        not isinstance(raw_trick_points, int)
        or isinstance(raw_trick_points, bool)
        or raw_trick_points < 0
    ):
        raise ValueError("raw_trick_points must be a non-negative integer")
    active_profile = profile or STANDARD_RULES_PROFILE
    factor = active_profile.contract_factor(mode, trump_suit)
    match_points = active_profile.match_bonus if won_all_tricks else 0
    return (raw_trick_points + match_points) * factor
