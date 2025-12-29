from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from core.cards import Card, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, RANKS, SUITS
from core.game import play_round


def _card_sort_key(card: Card) -> tuple[int, int]:
    return (SUITS.index(card.suit), RANKS.index(card.rank))


def _policy_lowest(state, player: int) -> Card:
    legal = state.legal_cards_for(player)
    return sorted(legal, key=_card_sort_key)[0]


def _policy_random(rng: random.Random) -> Callable:
    def _pick(state, player: int) -> Card:
        legal = state.legal_cards_for(player)
        return rng.choice(list(legal))

    return _pick


def _policy_human(state, player: int) -> Card:
    legal = state.legal_cards_for(player)
    print(f"Player {player} hand:")
    for idx, card in enumerate(sorted(state.hands[player], key=_card_sort_key)):
        print(f"  {idx}: {card}")
    print("Legal moves:")
    legal_sorted = sorted(legal, key=_card_sort_key)
    for idx, card in enumerate(legal_sorted):
        print(f"  {idx}: {card}")

    while True:
        choice = input("Choose legal card index: ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        idx = int(choice)
        if 0 <= idx < len(legal_sorted):
            return legal_sorted[idx]
        print("Index out of range.")


@dataclass
class PlayConfig:
    mode: str
    trump_suit: Optional[str]
    seed: Optional[int]
    leader: int
    players: List[str]
    replay_out: Optional[str]


def _build_policies(config: PlayConfig) -> List[Callable]:
    policies: List[Callable] = []
    for idx, kind in enumerate(config.players):
        if kind == "human":
            policies.append(_policy_human)
        elif kind == "random":
            rng = random.Random((config.seed or 0) + idx)
            policies.append(_policy_random(rng))
        elif kind == "low":
            policies.append(_policy_lowest)
        else:
            raise ValueError(f"unknown player type: {kind}")
    return policies


def run(config: PlayConfig) -> None:
    policies = _build_policies(config)
    result = play_round(
        policies,
        mode=config.mode,
        trump_suit=config.trump_suit,
        seed=config.seed,
        leader=config.leader,
    )

    print("Round finished.")
    print(f"Team points: {result.state.team_points}")

    if config.replay_out:
        from cli.replay import build_replay, save_replay

        replay = build_replay(
            result,
            seed=config.seed,
            mode=config.mode,
            trump_suit=config.trump_suit,
            leader=config.leader,
        )
        save_replay(config.replay_out, replay)
        print(f"Replay saved to {config.replay_out}")


def _parse_args() -> PlayConfig:
    parser = argparse.ArgumentParser(description="Play a Jass round (CLI)")
    parser.add_argument("--mode", choices=[MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE], required=True)
    parser.add_argument("--trump-suit", choices=SUITS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--leader", type=int, default=0)
    parser.add_argument(
        "--players",
        default="low,low,low,low",
        help="Comma list for 4 players: low,random,human",
    )
    parser.add_argument("--replay-out")
    args = parser.parse_args()

    players = [p.strip() for p in args.players.split(",") if p.strip()]
    if len(players) != 4:
        raise SystemExit("--players must contain 4 entries")

    seed = args.seed
    if seed is None:
        seed = random.SystemRandom().randint(1, 2**31 - 1)

    return PlayConfig(
        mode=args.mode,
        trump_suit=args.trump_suit,
        seed=seed,
        leader=args.leader,
        players=players,
        replay_out=args.replay_out,
    )


def main() -> None:
    config = _parse_args()
    run(config)


if __name__ == "__main__":
    main()
