import random

from core.cards import Card, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, make_deck
from core.legal_moves import RuleSet, legal_cards


def test_follow_suit_is_enforced() -> None:
    hand = [Card("schilten", "A"), Card("rosen", "6"), Card("schilten", "7")]
    trick = [Card("schilten", "9")]
    for mode in (MODE_OBEABE, MODE_UNEUFE, MODE_TRUMP):
        legal = legal_cards(hand, trick, mode, trump_suit="schellen")
        assert set(legal) == {Card("schilten", "A"), Card("schilten", "7")}


def test_trump_required_when_void() -> None:
    trump_suit = "rosen"
    hand = [Card("rosen", "6"), Card("rosen", "J"), Card("eicheln", "A")]
    trick = [Card("schilten", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit=trump_suit)
    assert set(legal) == {Card("rosen", "6"), Card("rosen", "J")}


def test_partner_winning_can_slough_when_allowed() -> None:
    trump_suit = "rosen"
    hand = [Card("rosen", "6"), Card("eicheln", "A")]
    trick = [Card("schilten", "9")]
    ruleset = RuleSet(must_trump=True, must_trump_if_partner_winning=False)
    legal = legal_cards(
        hand, trick, MODE_TRUMP, trump_suit=trump_suit, partner_is_winning=True, ruleset=ruleset
    )
    assert set(legal) == set(hand)


def test_partner_winning_still_must_trump_when_flagged() -> None:
    trump_suit = "rosen"
    hand = [Card("rosen", "6"), Card("eicheln", "A")]
    trick = [Card("schilten", "9")]
    ruleset = RuleSet(must_trump=True, must_trump_if_partner_winning=True)
    legal = legal_cards(
        hand, trick, MODE_TRUMP, trump_suit=trump_suit, partner_is_winning=True, ruleset=ruleset
    )
    assert set(legal) == {Card("rosen", "6")}


def test_overtrump_required_when_led_trump() -> None:
    trump_suit = "rosen"
    hand = [Card("rosen", "J"), Card("rosen", "7"), Card("schilten", "A")]
    trick = [Card("rosen", "9"), Card("rosen", "6")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit=trump_suit)
    assert set(legal) == {Card("rosen", "J")}


def test_overtrump_required_when_trump_in_trick() -> None:
    trump_suit = "rosen"
    hand = [Card("rosen", "J"), Card("rosen", "7"), Card("eicheln", "A")]
    trick = [Card("schilten", "9"), Card("rosen", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit=trump_suit)
    assert set(legal) == {Card("rosen", "J")}


def test_no_overtrump_available_all_trumps_legal() -> None:
    trump_suit = "rosen"
    hand = [Card("rosen", "6"), Card("rosen", "7"), Card("eicheln", "A")]
    trick = [Card("schilten", "9"), Card("rosen", "J")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit=trump_suit)
    assert set(legal) == {Card("rosen", "6"), Card("rosen", "7")}


def test_randomized_legal_cards_never_empty() -> None:
    rng = random.Random(42)
    deck = make_deck()
    for _ in range(1000):
        hand_size = rng.randint(1, 9)
        trick_size = rng.randint(1, 3)
        sample = rng.sample(deck, hand_size + trick_size)
        hand = sample[:hand_size]
        trick = sample[hand_size:]
        trump_suit = rng.choice(SUITS)
        mode = rng.choice([MODE_OBEABE, MODE_UNEUFE, MODE_TRUMP])
        legal = legal_cards(hand, trick, mode, trump_suit=trump_suit)
        assert legal
        assert set(legal).issubset(set(hand))
