from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS


@dataclass(frozen=True)
class BiddingState:
    starter: int
    current_player: int
    pushed: bool


@dataclass(frozen=True)
class BiddingAction:
    mode: str
    trump_suit: Optional[str] = None
    push: bool = False


@dataclass(frozen=True)
class BiddingResult:
    mode: str
    trump_suit: Optional[str]
    chooser: int
    pushed: bool


Policy = Callable[[BiddingState, int], BiddingAction]


def _validate_action(action: BiddingAction) -> None:
    if action.push:
        if action.mode or action.trump_suit is not None:
            raise ValueError("push action must not specify mode or trump_suit")
        return
    if action.mode not in (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE):
        raise ValueError("invalid mode")
    if action.mode == MODE_TRUMP:
        if action.trump_suit not in SUITS:
            raise ValueError("trump_suit required and must be a valid suit")
    else:
        if action.trump_suit is not None:
            raise ValueError("trump_suit must be None for non-trump modes")


def run_bidding(policy: Policy, starter: int = 0) -> BiddingResult:
    state = BiddingState(starter=starter, current_player=starter, pushed=False)
    action = policy(state, state.current_player)
    _validate_action(action)

    if action.push:
        chooser = (starter + 2) % 4
        state = BiddingState(starter=starter, current_player=chooser, pushed=True)
        action = policy(state, state.current_player)
        _validate_action(action)
        if action.push:
            raise ValueError("partner may not push")
        return BiddingResult(
            mode=action.mode,
            trump_suit=action.trump_suit,
            chooser=chooser,
            pushed=True,
        )

    return BiddingResult(
        mode=action.mode,
        trump_suit=action.trump_suit,
        chooser=starter,
        pushed=False,
    )
