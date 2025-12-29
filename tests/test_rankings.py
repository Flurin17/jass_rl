from core.cards import Card, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS
from core.rankings import OBEABE_ORDER, TRUMP_ORDER, UNEUFE_ORDER, beats


def test_trump_order() -> None:
    trump_suit = SUITS[0]
    led_suit = trump_suit
    cards = [Card(trump_suit, rank) for rank in TRUMP_ORDER]
    for higher, lower in zip(cards, cards[1:]):
        assert beats(higher, lower, led_suit, MODE_TRUMP, trump_suit)


def test_non_trump_order_in_trump_mode() -> None:
    trump_suit = SUITS[0]
    led_suit = SUITS[1]
    cards = [Card(led_suit, rank) for rank in OBEABE_ORDER]
    for higher, lower in zip(cards, cards[1:]):
        assert beats(higher, lower, led_suit, MODE_TRUMP, trump_suit)


def test_obeabe_order() -> None:
    led_suit = SUITS[0]
    cards = [Card(led_suit, rank) for rank in OBEABE_ORDER]
    for higher, lower in zip(cards, cards[1:]):
        assert beats(higher, lower, led_suit, MODE_OBEABE)


def test_uneufe_order() -> None:
    led_suit = SUITS[0]
    cards = [Card(led_suit, rank) for rank in UNEUFE_ORDER]
    for higher, lower in zip(cards, cards[1:]):
        assert beats(higher, lower, led_suit, MODE_UNEUFE)


def test_trump_beats_non_trump() -> None:
    trump_suit = SUITS[0]
    led_suit = SUITS[1]
    trump_card = Card(trump_suit, "6")
    led_card = Card(led_suit, "A")
    assert beats(trump_card, led_card, led_suit, MODE_TRUMP, trump_suit)


def test_led_suit_beats_off_suit_in_non_trump_modes() -> None:
    led_suit = SUITS[0]
    off_suit = SUITS[1]
    led_card = Card(led_suit, "6")
    off_card = Card(off_suit, "A")
    assert beats(led_card, off_card, led_suit, MODE_OBEABE)
    assert beats(led_card, off_card, led_suit, MODE_UNEUFE)
