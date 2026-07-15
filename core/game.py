from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .cards import Card, make_deck
from .legal_moves import RuleSet
from .ruleset import STANDARD_RULES_PROFILE, RulesetConfig
from .state import GameState
from .transitions import RoundFinalization, resolve_completed_trick

Policy = Callable[[GameState, int], Card]
PolicyMap = Sequence[Policy] | dict[int, Policy]


@dataclass(frozen=True)
class RoundResult:
    state: GameState
    play_log: list[tuple[int, Card]]
    raw_team_points: tuple[int, int] = (0, 0)
    contract_factor: int = 1
    match_team: int | None = None
    rules_version: str = ""
    rules_profile: RulesetConfig = STANDARD_RULES_PROFILE


def _get_policy(policy_by_player: PolicyMap, player: int) -> Policy:
    if isinstance(policy_by_player, dict):
        return policy_by_player[player]
    return policy_by_player[player]


def play_round(
    policy_by_player: PolicyMap,
    mode: str,
    trump_suit: str | None = None,
    seed: int | None = None,
    ruleset: RuleSet | None = None,
    leader: int = 0,
    profile: RulesetConfig | None = None,
) -> RoundResult:
    active_profile = profile or STANDARD_RULES_PROFILE
    active_profile.contract_factor(mode, trump_suit)  # Validate before dealing.
    active_legal_rules = ruleset or active_profile.legal_moves
    if (
        not isinstance(leader, int)
        or isinstance(leader, bool)
        or not 0 <= leader < 4
    ):
        raise ValueError("leader must be an integer from 0 to 3")

    rng = random.Random(seed)
    deck = make_deck()
    rng.shuffle(deck)

    hands = [deck[i * 9 : (i + 1) * 9] for i in range(4)]
    state = GameState(hands=hands, mode=mode, trump_suit=trump_suit, leader=leader)

    play_log: list[tuple[int, Card]] = []

    finalization: RoundFinalization | None = None
    for _ in range(9):
        for _ in range(4):
            player = state.current_player
            policy = _get_policy(policy_by_player, player)
            card = policy(state, player)
            state.play_card(player, card, ruleset=active_legal_rules)
            play_log.append((player, card))

        transition = resolve_completed_trick(state, profile=active_profile)
        if transition.finalization is not None:
            finalization = transition.finalization

    if finalization is None:  # pragma: no cover - loop invariant guard
        raise RuntimeError("round did not reach terminal finalization")

    return RoundResult(
        state=state,
        play_log=play_log,
        raw_team_points=finalization.raw_team_points,
        contract_factor=finalization.contract_factor,
        match_team=finalization.match_team,
        rules_version=active_profile.version,
        rules_profile=active_profile,
    )
