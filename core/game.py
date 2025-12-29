from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .cards import Card, MODE_TRUMP, make_deck
from .legal_moves import RuleSet
from .rankings import winning_card
from .scoring import trick_points
from .state import GameState, Trick, TrickResult

Policy = Callable[[GameState, int], Card]
PolicyMap = Union[Sequence[Policy], Dict[int, Policy]]


@dataclass(frozen=True)
class RoundResult:
    state: GameState
    play_log: List[Tuple[int, Card]]


def _get_policy(policy_by_player: PolicyMap, player: int) -> Policy:
    if isinstance(policy_by_player, dict):
        return policy_by_player[player]
    return policy_by_player[player]


def play_round(
    policy_by_player: PolicyMap,
    mode: str,
    trump_suit: Optional[str] = None,
    seed: Optional[int] = None,
    ruleset: Optional[RuleSet] = None,
    leader: int = 0,
) -> RoundResult:
    if mode == MODE_TRUMP and trump_suit is None:
        raise ValueError("trump_suit is required for trump mode")

    rng = random.Random(seed)
    deck = make_deck()
    rng.shuffle(deck)

    hands = [deck[i * 9 : (i + 1) * 9] for i in range(4)]
    state = GameState(hands=hands, mode=mode, trump_suit=trump_suit, leader=leader)

    play_log: List[Tuple[int, Card]] = []

    for trick_index in range(9):
        state.trick_index = trick_index
        state.trick = Trick()
        state.leader = leader

        for _ in range(4):
            player = state.current_player
            policy = _get_policy(policy_by_player, player)
            card = policy(state, player)
            state.play_card(player, card, ruleset=ruleset)
            play_log.append((player, card))

        led_suit = state.trick.led_suit
        cards = state.trick.cards
        winning = winning_card(cards, led_suit, mode, trump_suit)
        winning_player = next(
            player for player, card in state.trick.plays if card == winning
        )
        last_trick = trick_index == 8
        points = trick_points(cards, mode, trump_suit, last_trick=last_trick)
        state.team_points[state.team_index(winning_player)] += points

        state.completed_tricks.append(
            TrickResult(
                plays=list(state.trick.plays),
                winner=winning_player,
                points=points,
                last_trick=last_trick,
            )
        )

        leader = winning_player

    return RoundResult(state=state, play_log=play_log)
