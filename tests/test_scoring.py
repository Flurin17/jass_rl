import pytest

from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, make_deck
from core.ruleset import (
    LEGACY_UNMULTIPLIED_PROFILE,
    STANDARD_CONTRACT_FACTORS,
    RulesetConfig,
    all_contracts_x1_profile,
)
from core.scoring import (
    OBEABE_POINTS,
    TRUMP_POINTS,
    UNEUFE_POINTS,
    card_points,
    round_score,
)


def _total_points(mode: str, trump_suit: str | None = None) -> int:
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


def test_standard_contract_factors() -> None:
    profile = RulesetConfig()
    assert profile.contract_factor(MODE_TRUMP, "eicheln") == 1
    assert profile.contract_factor(MODE_TRUMP, "rosen") == 1
    assert profile.contract_factor(MODE_TRUMP, "schilten") == 2
    assert profile.contract_factor(MODE_TRUMP, "schellen") == 2
    assert profile.contract_factor(MODE_OBEABE) == 3
    assert profile.contract_factor(MODE_UNEUFE) == 4


def test_contract_factors_are_configurable_and_copied() -> None:
    factors = dict(STANDARD_CONTRACT_FACTORS)
    factors["rosen"] = 5
    profile = RulesetConfig(version="test-profile-v1", contract_factors=factors)
    factors["rosen"] = 9

    assert profile.contract_factor(MODE_TRUMP, "rosen") == 5
    with pytest.raises(TypeError):
        profile.contract_factors["rosen"] = 2  # type: ignore[index]


def test_match_bonus_is_added_before_contract_factor() -> None:
    profile = RulesetConfig()
    assert round_score(157, MODE_TRUMP, "schilten", profile=profile) == 314
    assert (
        round_score(
            157,
            MODE_TRUMP,
            "schilten",
            won_all_tricks=True,
            profile=profile,
        )
        == 514
    )
    assert round_score(157, MODE_OBEABE, won_all_tricks=True, profile=profile) == 771


def test_all_contracts_x1_and_legacy_profiles() -> None:
    x1_profile = all_contracts_x1_profile()
    for suit in SUITS:
        assert x1_profile.contract_factor(MODE_TRUMP, suit) == 1
    assert x1_profile.contract_factor(MODE_OBEABE) == 1
    assert x1_profile.contract_factor(MODE_UNEUFE) == 1
    assert round_score(157, MODE_OBEABE, profile=x1_profile) == 157
    assert (
        round_score(
            157,
            MODE_OBEABE,
            won_all_tricks=True,
            profile=LEGACY_UNMULTIPLIED_PROFILE,
        )
        == 157
    )


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
def test_contract_factor_rejects_invalid_contract(
    mode: str, trump_suit: str | None
) -> None:
    with pytest.raises(ValueError):
        RulesetConfig().contract_factor(mode, trump_suit)


def test_profile_accepts_partial_factor_overrides() -> None:
    profile = RulesetConfig(contract_factors={"rosen": 7})
    assert profile.contract_factor(MODE_TRUMP, "rosen") == 7
    assert profile.contract_factor(MODE_OBEABE) == 3


def test_profile_rejects_unknown_factor_keys() -> None:
    with pytest.raises(ValueError):
        RulesetConfig(contract_factors={"invalid": 1})


def test_profile_json_rejects_string_booleans() -> None:
    with pytest.raises(ValueError, match="booleans"):
        RulesetConfig.from_dict({"allow_stock": "false", "allow_weis": True})
