"""Isolated CPU/MPS MaskablePPO throughput benchmark.

Each device runs in a fresh subprocess.  This avoids carrying PyTorch allocator
state between CPU and MPS runs and lets the OS reclaim unified memory after
each measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from core.cards import MODE_TRUMP
from rl.ctde_policy import CTDEMaskableActorCriticPolicy
from rl.train_selfplay import DEFAULT_CPU_THREADS, TrainConfig, _make_vec_env


@dataclass(frozen=True)
class BenchmarkConfig:
    n_steps: int = 128
    n_envs: int = 1
    batch_size: int = 64
    n_epochs: int = 2
    rollouts: int = 2
    net_arch: tuple[int, ...] = (128, 128)
    seed: int = 0
    cpu_threads: int = DEFAULT_CPU_THREADS

    @property
    def total_timesteps(self) -> int:
        return self.n_steps * self.n_envs * self.rollouts

    def validate(self) -> None:
        values = {
            "n_steps": self.n_steps,
            "n_envs": self.n_envs,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "rollouts": self.rollouts,
            "cpu_threads": self.cpu_threads,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        rollout_size = self.n_steps * self.n_envs
        if self.batch_size > rollout_size or rollout_size % self.batch_size:
            raise ValueError("batch_size must evenly divide n_steps * n_envs")
        if not self.net_arch or any(width <= 0 for width in self.net_arch):
            raise ValueError("net_arch must contain positive widths")


def peak_rss_bytes(raw_peak: int | None = None, system: str | None = None) -> int:
    """Normalize ``ru_maxrss`` (bytes on macOS, KiB on Linux) to bytes."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw_peak is None else raw_peak
    active_system = sys.platform if system is None else system
    return int(peak if active_system == "darwin" else peak * 1024)


def mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except (ImportError, AttributeError):
        return False


class _MemorySampler:
    def __init__(self, device: str) -> None:
        self.device = device
        self.peak_mps_allocated_bytes = 0
        self.peak_mps_driver_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _sample(self) -> None:
        if self.device != "mps":
            return
        import torch

        while not self._stop.wait(0.01):
            self.peak_mps_allocated_bytes = max(
                self.peak_mps_allocated_bytes,
                int(torch.mps.current_allocated_memory()),
            )
            self.peak_mps_driver_bytes = max(
                self.peak_mps_driver_bytes,
                int(torch.mps.driver_allocated_memory()),
            )


def _training_config(config: BenchmarkConfig, device: str) -> TrainConfig:
    return TrainConfig(
        seed=config.seed,
        total_steps=config.total_timesteps,
        iterations=1,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        n_envs=config.n_envs,
        vec_env="dummy",
        device=device,
        cpu_threads=config.cpu_threads,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        net_arch=config.net_arch,
        control_team=True,
        randomize_starter=True,
        reward_scale=0.001,
        terminal_win_bonus=1.0,
        mode=MODE_TRUMP,
        trump_suit="eicheln",
    )


def run_worker(config: BenchmarkConfig, device: str) -> dict[str, object]:
    """Run one measured training job in the current worker process."""

    config.validate()
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if device == "mps" and not mps_available():
        return {"device": device, "status": "skipped", "reason": "MPS unavailable"}

    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    env = _make_vec_env(_training_config(config, device), None)
    sampler = _MemorySampler(device)
    sampler.start()
    try:
        started = time.perf_counter()
        model = MaskablePPO(
            CTDEMaskableActorCriticPolicy,
            env,
            verbose=0,
            seed=config.seed,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=1.0,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.01,
            device=device,
            policy_kwargs={"net_arch": list(config.net_arch)},
        )
        if device == "mps":
            torch.mps.synchronize()
        initialized = time.perf_counter()
        model.learn(total_timesteps=config.total_timesteps)
        if device == "mps":
            torch.mps.synchronize()
        finished = time.perf_counter()
    finally:
        sampler.stop()
        env.close()

    training_seconds = finished - initialized
    result = {
        "device": device,
        "cpu_threads": int(torch.get_num_threads()),
        "algorithm": "MaskablePPO-CTDE",
        "status": "ok",
        "requested_timesteps": config.total_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "initialization_seconds": initialized - started,
        "training_seconds": training_seconds,
        "steps_per_second": model.num_timesteps / training_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_mps_allocated_bytes": sampler.peak_mps_allocated_bytes,
        "peak_mps_driver_bytes": sampler.peak_mps_driver_bytes,
    }
    if device == "mps":
        torch.mps.empty_cache()
    return result


def _worker_command(config: BenchmarkConfig, device: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rl.benchmark_training",
        "--worker",
        "--devices",
        device,
        "--n-steps",
        str(config.n_steps),
        "--n-envs",
        str(config.n_envs),
        "--batch-size",
        str(config.batch_size),
        "--n-epochs",
        str(config.n_epochs),
        "--rollouts",
        str(config.rollouts),
        "--net-arch",
        ",".join(map(str, config.net_arch)),
        "--seed",
        str(config.seed),
        "--cpu-threads",
        str(config.cpu_threads),
    ]


def run_comparison(config: BenchmarkConfig, devices: tuple[str, ...]) -> dict[str, object]:
    """Run requested devices sequentially in isolated child processes."""

    config.validate()
    results: list[dict[str, object]] = []
    for device in devices:
        if device not in {"cpu", "mps"}:
            raise ValueError("devices must contain only cpu or mps")
        if device == "mps" and not mps_available():
            results.append({"device": device, "status": "skipped", "reason": "MPS unavailable"})
            continue
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(config.cpu_threads),
                "MKL_NUM_THREADS": str(config.cpu_threads),
                "PYTHONHASHSEED": str(config.seed),
            }
        )
        completed = subprocess.run(
            _worker_command(config, device),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode:
            results.append(
                {
                    "device": device,
                    "status": "error",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip(),
                }
            )
            continue
        results.append(json.loads(completed.stdout))

    return {
        "benchmark": asdict(config),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "results": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare isolated CPU/MPS training throughput")
    parser.add_argument("--devices", default="cpu,mps")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--net-arch", default="128,128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = BenchmarkConfig(
            n_steps=args.n_steps,
            n_envs=args.n_envs,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            rollouts=args.rollouts,
            net_arch=tuple(int(value) for value in args.net_arch.split(",") if value),
            seed=args.seed,
            cpu_threads=args.cpu_threads,
        )
        config.validate()
        devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
        if not devices:
            raise ValueError("at least one device is required")
        payload = run_worker(config, devices[0]) if args.worker else run_comparison(config, devices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
