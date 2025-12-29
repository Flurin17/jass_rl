import pytest

from core.bidding import BiddingAction, BiddingState, run_bidding
from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE


def test_starter_selects_mode() -> None:
    def policy(state: BiddingState, player: int) -> BiddingAction:
        return BiddingAction(mode=MODE_TRUMP, trump_suit="rosen")

    result = run_bidding(policy, starter=1)
    assert result.mode == MODE_TRUMP
    assert result.trump_suit == "rosen"
    assert result.chooser == 1
    assert result.pushed is False


def test_push_then_partner_selects() -> None:
    def policy(state: BiddingState, player: int) -> BiddingAction:
        if not state.pushed:
            return BiddingAction(mode="", push=True)
        return BiddingAction(mode=MODE_OBEABE)

    result = run_bidding(policy, starter=0)
    assert result.mode == MODE_OBEABE
    assert result.trump_suit is None
    assert result.chooser == 2
    assert result.pushed is True


def test_partner_cannot_push() -> None:
    def policy(state: BiddingState, player: int) -> BiddingAction:
        return BiddingAction(mode="", push=True)

    with pytest.raises(ValueError):
        run_bidding(policy, starter=3)


def test_invalid_mode_rejected() -> None:
    def policy(state: BiddingState, player: int) -> BiddingAction:
        return BiddingAction(mode="invalid")

    with pytest.raises(ValueError):
        run_bidding(policy, starter=0)


def test_non_trump_must_not_set_suit() -> None:
    def policy(state: BiddingState, player: int) -> BiddingAction:
        return BiddingAction(mode=MODE_UNEUFE, trump_suit="rosen")

    with pytest.raises(ValueError):
        run_bidding(policy, starter=0)
