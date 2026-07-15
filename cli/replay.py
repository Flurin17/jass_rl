from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.cards import Card, make_deck
from core.game import RoundResult
from core.ruleset import LEGACY_UNMULTIPLIED_PROFILE, RulesetConfig
from core.state import GameState
from core.transitions import resolve_completed_trick


@dataclass(frozen=True)
class ReplayData:
    mode: str
    trump_suit: str | None
    leader: int
    seed: int
    play_log: list[tuple[int, Card]]
    rules_profile: dict[str, object] | None = None


def encode_play_log(play_log: Iterable[tuple[int, Card]]) -> list[list[object]]:
    return [[player, card.suit, card.rank] for player, card in play_log]


def decode_play_log(data: Iterable[Iterable[object]]) -> list[tuple[int, Card]]:
    decoded: list[tuple[int, Card]] = []
    for entry in data:
        player, suit, rank = entry
        decoded.append((int(player), Card(str(suit), str(rank))))
    return decoded


def build_replay(
    result: RoundResult,
    seed: int,
    mode: str,
    trump_suit: str | None,
    leader: int,
) -> ReplayData:
    return ReplayData(
        mode=mode,
        trump_suit=trump_suit,
        leader=leader,
        seed=seed,
        play_log=list(result.play_log),
        rules_profile=result.rules_profile.to_dict(),
    )


def save_replay(path: str | Path, replay: ReplayData) -> None:
    payload = {
        "mode": replay.mode,
        "trump_suit": replay.trump_suit,
        "leader": replay.leader,
        "seed": replay.seed,
        "play_log": encode_play_log(replay.play_log),
        "rules_profile": replay.rules_profile,
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
        rules_profile=payload.get("rules_profile"),
    )


def replay_game(replay: ReplayData, strict: bool = True) -> GameState:
    if replay.seed is None:
        raise ValueError("seed is required for replay")

    profile = (
        RulesetConfig.from_dict(replay.rules_profile)
        if replay.rules_profile is not None
        else LEGACY_UNMULTIPLIED_PROFILE
    )

    rng = random.Random(replay.seed)
    deck = make_deck()
    rng.shuffle(deck)

    hands = [deck[i * 9 : (i + 1) * 9] for i in range(4)]
    state = GameState(
        hands=hands,
        mode=replay.mode,
        trump_suit=replay.trump_suit,
        leader=replay.leader,
    )

    play_iter = iter(replay.play_log)
    for _ in range(9):
        for _ in range(4):
            try:
                logged_player, card = next(play_iter)
            except StopIteration as exc:
                raise ValueError("replay ended before all 36 cards were played") from exc
            if strict and logged_player != state.current_player:
                raise ValueError("replay order mismatch")
            player = logged_player if strict else state.current_player
            state.play_card(player, card, ruleset=profile.legal_moves)

        resolve_completed_trick(state, profile=profile)

    try:
        next(play_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("replay contains entries after the 36th card")

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
