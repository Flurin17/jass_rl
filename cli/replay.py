from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from core.cards import Card, MODE_TRUMP, make_deck
from core.rankings import winning_card
from core.scoring import trick_points
from core.state import GameState, Trick, TrickResult
from core.game import RoundResult


@dataclass(frozen=True)
class ReplayData:
    mode: str
    trump_suit: Optional[str]
    leader: int
    seed: int
    play_log: List[Tuple[int, Card]]


def encode_play_log(play_log: Iterable[Tuple[int, Card]]) -> List[List[object]]:
    return [[player, card.suit, card.rank] for player, card in play_log]


def decode_play_log(data: Iterable[Iterable[object]]) -> List[Tuple[int, Card]]:
    decoded: List[Tuple[int, Card]] = []
    for entry in data:
        player, suit, rank = entry
        decoded.append((int(player), Card(str(suit), str(rank))))
    return decoded


def build_replay(
    result: RoundResult,
    seed: int,
    mode: str,
    trump_suit: Optional[str],
    leader: int,
) -> ReplayData:
    return ReplayData(
        mode=mode,
        trump_suit=trump_suit,
        leader=leader,
        seed=seed,
        play_log=list(result.play_log),
    )


def save_replay(path: str | Path, replay: ReplayData) -> None:
    payload = {
        "mode": replay.mode,
        "trump_suit": replay.trump_suit,
        "leader": replay.leader,
        "seed": replay.seed,
        "play_log": encode_play_log(replay.play_log),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_replay(path: str | Path) -> ReplayData:
    payload = json.loads(Path(path).read_text())
    return ReplayData(
        mode=payload["mode"],
        trump_suit=payload.get("trump_suit"),
        leader=int(payload["leader"]),
        seed=int(payload["seed"]),
        play_log=decode_play_log(payload["play_log"]),
    )


def replay_game(replay: ReplayData, strict: bool = True) -> GameState:
    if replay.seed is None:
        raise ValueError("seed is required for replay")

    rng = random.Random(replay.seed)
    deck = make_deck()
    rng.shuffle(deck)

    hands = [deck[i * 9 : (i + 1) * 9] for i in range(4)]
    state = GameState(hands=hands, mode=replay.mode, trump_suit=replay.trump_suit)

    play_iter = iter(replay.play_log)
    leader = replay.leader
    for trick_index in range(9):
        state.trick_index = trick_index
        state.trick = Trick()
        state.leader = leader

        for _ in range(4):
            player, card = next(play_iter)
            if strict and player != state.current_player:
                raise ValueError("replay order mismatch")
            state.play_card(player, card)

        led_suit = state.trick.led_suit
        cards = state.trick.cards
        winning = winning_card(cards, led_suit, replay.mode, replay.trump_suit)
        winning_player = next(p for p, c in state.trick.plays if c == winning)
        last_trick = trick_index == 8
        points = trick_points(cards, replay.mode, replay.trump_suit, last_trick=last_trick)
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

    return state


def run(path: str, strict: bool = True) -> None:
    replay = load_replay(path)
    state = replay_game(replay, strict=strict)

    print("Replay finished.")
    print(f"Team points: {state.team_points}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved Jass game")
    parser.add_argument("path")
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.path, strict=not args.no_strict)


if __name__ == "__main__":
    main()
