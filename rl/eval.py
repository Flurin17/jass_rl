from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rl.single_agent_env import JassSingleAgentEnv, policy_lowest

try:
    from sb3_contrib import MaskablePPO
except ImportError as exc:  # pragma: no cover
    raise ImportError("sb3-contrib is required for evaluation") from exc


@dataclass
class EvalConfig:
    model_path: Path
    episodes: int
    seed: int
    enable_bidding: bool
    mode: Optional[str]
    trump_suit: Optional[str]


def evaluate(config: EvalConfig) -> dict:
    env = JassSingleAgentEnv(
        seed=config.seed,
        enable_bidding=config.enable_bidding,
        mode=config.mode,
        trump_suit=config.trump_suit,
        opponent_policy=policy_lowest,
    )

    model = MaskablePPO.load(config.model_path)

    wins = 0
    ties = 0
    total_points = 0

    for ep in range(config.episodes):
        obs, _ = env.reset(seed=config.seed + ep)
        done = False
        while not done:
            mask = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        state = env.env.state
        team_a = state.team_points[0]
        team_b = state.team_points[1]
        total_points += team_a
        if team_a > team_b:
            wins += 1
        elif team_a == team_b:
            ties += 1

    return {
        "episodes": config.episodes,
        "wins": wins,
        "ties": ties,
        "win_rate": wins / config.episodes,
        "avg_team_a_points": total_points / config.episodes,
    }


def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a Jass agent")
    parser.add_argument("model_path")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-bidding", action="store_true")
    parser.add_argument("--mode")
    parser.add_argument("--trump-suit")
    args = parser.parse_args()

    return EvalConfig(
        model_path=Path(args.model_path),
        episodes=args.episodes,
        seed=args.seed,
        enable_bidding=args.enable_bidding,
        mode=args.mode,
        trump_suit=args.trump_suit,
    )


def main() -> None:
    config = _parse_args()
    metrics = evaluate(config)
    print(metrics)


if __name__ == "__main__":
    main()
