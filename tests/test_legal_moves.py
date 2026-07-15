import random

import pytest

from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card, make_deck
from core.legal_moves import RuleSet, legal_cards


def test_non_trump_contract_requires_following_suit() -> None:
    hand = [Card("schilten", "A"), Card("rosen", "6"), Card("schilten", "7")]
    trick = [Card("schilten", "9")]
    for mode in (MODE_OBEABE, MODE_UNEUFE):
        legal = legal_cards(hand, trick, mode)
        assert set(legal) == {Card("schilten", "A"), Card("schilten", "7")}


def test_trump_contract_allows_following_or_trumping_non_trump_lead() -> None:
    hand = [
        Card("schilten", "A"),
        Card("rosen", "6"),
        Card("schilten", "7"),
        Card("eicheln", "A"),
    ]
    trick = [Card("schilten", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert set(legal) == {
        Card("schilten", "A"),
        Card("schilten", "7"),
        Card("rosen", "6"),
    }


def test_house_rule_can_disable_trumping_while_following() -> None:
    hand = [Card("schilten", "A"), Card("rosen", "6")]
    trick = [Card("schilten", "9")]
    ruleset = RuleSet(allow_trump_on_non_trump_lead=False)
    legal = legal_cards(
        hand,
        trick,
        MODE_TRUMP,
        trump_suit="rosen",
        ruleset=ruleset,
    )
    assert legal == [Card("schilten", "A")]


def test_void_player_may_trump_or_discard() -> None:
    hand = [Card("rosen", "6"), Card("rosen", "J"), Card("eicheln", "A")]
    trick = [Card("schilten", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert legal == hand


def test_partner_winning_can_discard_under_optional_trump_obligation() -> None:
    hand = [Card("rosen", "6"), Card("eicheln", "A")]
    trick = [Card("schilten", "9")]
    ruleset = RuleSet(must_trump=True, must_trump_if_partner_winning=False)
    legal = legal_cards(
        hand,
        trick,
        MODE_TRUMP,
        trump_suit="rosen",
        partner_is_winning=True,
        ruleset=ruleset,
    )
    assert set(legal) == set(hand)


def test_optional_trump_obligation_can_apply_while_partner_wins() -> None:
    hand = [Card("rosen", "6"), Card("eicheln", "A")]
    trick = [Card("schilten", "9")]
    ruleset = RuleSet(must_trump=True, must_trump_if_partner_winning=True)
    legal = legal_cards(
        hand,
        trick,
        MODE_TRUMP,
        trump_suit="rosen",
        partner_is_winning=True,
        ruleset=ruleset,
    )
    assert legal == [Card("rosen", "6")]


def test_trump_lead_requires_trump_but_not_overtrump() -> None:
    hand = [Card("rosen", "J"), Card("rosen", "7"), Card("schilten", "A")]
    trick = [Card("rosen", "9"), Card("rosen", "6")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert set(legal) == {Card("rosen", "J"), Card("rosen", "7")}


def test_undertrump_forbidden_after_non_trump_is_ruffed() -> None:
    hand = [
        Card("schilten", "A"),
        Card("rosen", "J"),
        Card("rosen", "7"),
        Card("eicheln", "A"),
    ]
    trick = [Card("schilten", "9"), Card("rosen", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert set(legal) == {Card("schilten", "A"), Card("rosen", "J")}


def test_void_player_may_discard_but_not_undertrump_after_ruff() -> None:
    hand = [Card("rosen", "6"), Card("rosen", "J"), Card("eicheln", "A")]
    trick = [Card("schilten", "9"), Card("rosen", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert set(legal) == {Card("rosen", "J"), Card("eicheln", "A")}


def test_undertrump_allowed_when_hand_contains_only_trumps() -> None:
    hand = [Card("rosen", "6"), Card("rosen", "7")]
    trick = [Card("schilten", "9"), Card("rosen", "J")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert legal == hand


def test_standard_profile_still_allows_undertrump_when_an_overtrump_is_held() -> None:
    hand = [Card("rosen", "6"), Card("rosen", "J")]
    trick = [Card("schilten", "9"), Card("rosen", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert legal == hand


def test_strict_all_trump_rule_requires_an_available_overtrump() -> None:
    hand = [Card("rosen", "6"), Card("rosen", "J")]
    trick = [Card("schilten", "9"), Card("rosen", "9")]
    legal = legal_cards(
        hand,
        trick,
        MODE_TRUMP,
        trump_suit="rosen",
        ruleset=RuleSet(must_overtrump_when_only_trumps=True),
    )
    assert legal == [Card("rosen", "J")]


def test_strict_all_trump_rule_allows_undertrump_when_none_can_overtrump() -> None:
    hand = [Card("rosen", "6"), Card("rosen", "7")]
    trick = [Card("schilten", "9"), Card("rosen", "J")]
    legal = legal_cards(
        hand,
        trick,
        MODE_TRUMP,
        trump_suit="rosen",
        ruleset=RuleSet(must_overtrump_when_only_trumps=True),
    )
    assert legal == hand


def test_sole_puur_may_be_withheld_when_trump_is_led() -> None:
    hand = [Card("rosen", "J"), Card("eicheln", "A"), Card("schilten", "6")]
    trick = [Card("rosen", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert legal == hand


def test_house_rule_can_disable_puur_exception() -> None:
    hand = [Card("rosen", "J"), Card("eicheln", "A")]
    trick = [Card("rosen", "9")]
    ruleset = RuleSet(allow_puur_withhold=False)
    legal = legal_cards(
        hand,
        trick,
        MODE_TRUMP,
        trump_suit="rosen",
        ruleset=ruleset,
    )
    assert legal == [Card("rosen", "J")]


def test_puur_is_not_exempt_when_another_trump_is_held() -> None:
    hand = [Card("rosen", "J"), Card("rosen", "7"), Card("eicheln", "A")]
    trick = [Card("rosen", "9")]
    legal = legal_cards(hand, trick, MODE_TRUMP, trump_suit="rosen")
    assert set(legal) == {Card("rosen", "J"), Card("rosen", "7")}


@pytest.mark.parametrize(
    ("mode", "trump_suit"),
    [
        ("invalid", None),
        (MODE_TRUMP, None),
        (MODE_TRUMP, "invalid"),
        (MODE_OBEABE, "rosen"),
        (MODE_UNEUFE, "schilten"),
    ],
)
def test_invalid_contract_is_rejected(mode: str, trump_suit: str | None) -> None:
    with pytest.raises(ValueError):
        legal_cards([Card("rosen", "A")], [], mode, trump_suit)


def test_cannot_choose_card_for_completed_trick() -> None:
    trick = [
        Card("rosen", "6"),
        Card("rosen", "7"),
        Card("rosen", "8"),
        Card("rosen", "9"),
    ]
    with pytest.raises(ValueError):
        legal_cards([Card("rosen", "A")], trick, MODE_OBEABE)


def test_randomized_legal_cards_never_empty() -> None:
    rng = random.Random(42)
    deck = make_deck()
    for _ in range(10_000):
        hand_size = rng.randint(1, 9)
        trick_size = rng.randint(1, 3)
        sample = rng.sample(deck, hand_size + trick_size)
        hand = sample[:hand_size]
        trick = sample[hand_size:]
        mode = rng.choice([MODE_OBEABE, MODE_UNEUFE, MODE_TRUMP])
        trump_suit = rng.choice(SUITS) if mode == MODE_TRUMP else None
        legal = legal_cards(hand, trick, mode, trump_suit=trump_suit)
        assert legal
        assert set(legal).issubset(set(hand))
