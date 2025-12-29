from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from rl.single_agent_env import JassSingleAgentEnv, OpponentPolicy, policy_lowest

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except ImportError as exc:  # pragma: no cover
    raise ImportError("sb3-contrib is required for training") from exc


@dataclass
class TrainConfig:
    seed: int
    total_steps: int
    iterations: int
    steps_per_iter: int
    n_steps: int
    batch_size: int
    n_envs: int
    vec_env: str
    device: str
    save_dir: Path
    selfplay: bool
    selfplay_prob: float
    enable_bidding: bool
    mode: Optional[str]
    trump_suit: Optional[str]


class OpponentPool:
    def __init__(self, selfplay_prob: float) -> None:
        self.selfplay_prob = selfplay_prob
        self.checkpoints: List[Path] = []

    def add(self, path: Path) -> None:
        self.checkpoints.append(path)

    def sample_policy(self, rng) -> OpponentPolicy:
        if not self.checkpoints or rng.random() > self.selfplay_prob:
            return policy_lowest
        checkpoint = rng.choice(self.checkpoints)
        model = MaskablePPO.load(checkpoint)

        def _policy(env, agent: str) -> int:
            obs = env.observe(agent)
            action, _ = model.predict(
                obs["observation"], action_masks=obs["action_mask"], deterministic=True
            )
            return int(action)

        return _policy


def _build_env(
    config: TrainConfig,
    opponent_sampler: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> JassSingleAgentEnv:
    env = JassSingleAgentEnv(
        seed=seed if seed is not None else config.seed,
        enable_bidding=config.enable_bidding,
        mode=config.mode,
        trump_suit=config.trump_suit,
        opponent_policy=policy_lowest,
        opponent_sampler=opponent_sampler,
    )
    return env


def train(config: TrainConfig) -> Path:
    config.save_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.save_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    opponent_pool = OpponentPool(config.selfplay_prob) if config.selfplay else None
    opponent_sampler = opponent_pool.sample_policy if opponent_pool else None

    if config.n_envs <= 1:
        env = _build_env(config, opponent_sampler=opponent_sampler)
    else:
        env_fns = [
            lambda rank=rank: _build_env(
                config,
                opponent_sampler=opponent_sampler,
                seed=config.seed + rank,
            )
            for rank in range(config.n_envs)
        ]
        if config.vec_env == "subproc":
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)
        env = VecMonitor(env)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        device=config.device,
    )

    for idx in range(config.iterations):
        model.learn(total_timesteps=config.steps_per_iter)
        checkpoint = run_dir / f"checkpoint_{idx+1}.zip"
        model.save(checkpoint)
        if opponent_pool is not None:
            opponent_pool.add(checkpoint)

    final_path = run_dir / "model_final.zip"
    model.save(final_path)
    return final_path


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train Jass agent (MaskablePPO)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--selfplay", action="store_true")
    parser.add_argument("--selfplay-prob", type=float, default=0.5)
    parser.add_argument("--no-bidding", action="store_true")
    parser.add_argument("--mode")
    parser.add_argument("--trump-suit")
    args = parser.parse_args()

    iterations = max(1, args.iterations)
    steps_per_iter = max(1, args.total_steps // iterations)

    return TrainConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        iterations=iterations,
        steps_per_iter=steps_per_iter,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_envs=max(1, args.n_envs),
        vec_env=args.vec_env,
        device=args.device,
        save_dir=Path(args.save_dir),
        selfplay=args.selfplay,
        selfplay_prob=args.selfplay_prob,
        enable_bidding=not args.no_bidding,
        mode=args.mode,
        trump_suit=args.trump_suit,
    )


def main() -> None:
    config = _parse_args()
    final_path = train(config)
    print(f"Saved model to {final_path}")


if __name__ == "__main__":
    main()
