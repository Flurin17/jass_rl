from typing import Optional

from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, make_deck
from core.scoring import OBEABE_POINTS, TRUMP_POINTS, UNEUFE_POINTS, card_points


def _total_points(mode: str, trump_suit: Optional[str] = None) -> int:
    return sum(card_points(card, mode, trump_suit) for card in make_deck())


def test_total_points_trump() -> None:
    for suit in SUITS:
        assert _total_points(MODE_TRUMP, suit) == 152
        assert _total_points(MODE_TRUMP, suit) + 5 == 157


def test_total_points_obeabe_uneufe() -> None:
    assert _total_points(MODE_OBEABE) == 152
    assert _total_points(MODE_OBEABE) + 5 == 157
    assert _total_points(MODE_UNEUFE) == 152
    assert _total_points(MODE_UNEUFE) + 5 == 157


def test_specific_card_points() -> None:
    trump_suit = SUITS[0]
    from core.cards import Card

    assert card_points(Card(trump_suit, "J"), MODE_TRUMP, trump_suit) == TRUMP_POINTS["J"]
    assert card_points(Card(trump_suit, "9"), MODE_TRUMP, trump_suit) == TRUMP_POINTS["9"]
    assert (
        card_points(Card(trump_suit, "A"), MODE_TRUMP, trump_suit)
        == TRUMP_POINTS["A"]
    )
    other_suit = "rosen" if trump_suit != "rosen" else "schilten"
    assert card_points(Card(other_suit, "A"), MODE_TRUMP, trump_suit) == 11
    assert card_points(Card(other_suit, "8"), MODE_OBEABE) == OBEABE_POINTS["8"]
    assert card_points(Card(other_suit, "6"), MODE_UNEUFE) == UNEUFE_POINTS["6"]
