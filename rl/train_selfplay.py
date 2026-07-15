from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import os
import random
import re
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np

from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS
from core.ruleset import (
    ALL_CONTRACTS_X1_PROFILE,
    LEGACY_UNMULTIPLIED_PROFILE,
    STANDARD_RULES_PROFILE,
    RulesetConfig,
)
from env.jass_aec_env import ACTION_COUNT, OBS_SIZE, OBSERVATION_SCHEMA_VERSION
from rl.baselines import SeededRandomPolicy, StrategicHeuristicPolicy, as_aec_policy
from rl.ctde_policy import CTDEMaskableActorCriticPolicy
from rl.privileged_observation import (
    CTDE_OBS_SIZE,
    PRIVILEGED_OBSERVATION_SCHEMA_NAME,
    PRIVILEGED_OBSERVATION_SCHEMA_VERSION,
    pad_public_observation,
)
from rl.reference_verardo import (
    VERARDO_PROFILE_VERSION,
    VERARDO_V1_PROFILE,
    VerardoReferencePolicy,
)
from rl.run_manifest import (
    MANIFEST_FILENAME,
    assert_compatible,
    build_manifest,
    git_metadata,
    load_manifest,
    write_manifest,
)
from rl.single_agent_env import (
    JassSingleAgentEnv,
    JassTeamEnv,
    OpponentPolicy,
)

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except ImportError as exc:  # pragma: no cover
    raise ImportError("sb3-contrib is required for training") from exc


PROFILE_BY_NAME = {
    "standard": STANDARD_RULES_PROFILE,
    "all-x1": ALL_CONTRACTS_X1_PROFILE,
    "legacy": LEGACY_UNMULTIPLIED_PROFILE,
    VERARDO_PROFILE_VERSION: VERARDO_V1_PROFILE,
}

DIRECT_OPPONENTS = ("strategic", "random", "verardo")
OpponentMixture = tuple[tuple[str, float], ...]
_DEFAULT_DIRECT_MIXTURE: OpponentMixture = (("strategic", 1.0),)
_LEGACY_SELFPLAY_FALLBACK_MIXTURE: OpponentMixture = (
    ("strategic", 0.65),
    ("random", 0.35),
)

# Eight intra-op workers gave the best mean throughput for the default 512x256
# CTDE policy on the supported 12-core M3 Pro while leaving headroom for the OS.
# Cap the portable default so high-core-count hosts do not oversubscribe this
# comparatively small network; callers can still select any positive value.
DEFAULT_CPU_THREADS = min(8, os.cpu_count() or 1)


def normalized_opponent_mixture(mixture: OpponentMixture) -> OpponentMixture:
    """Validate and normalize a deterministic opponent distribution."""

    if not mixture:
        raise ValueError("opponent_mixture must not be empty")
    seen: set[str] = set()
    validated: list[tuple[str, float]] = []
    for entry in mixture:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("opponent_mixture entries must be (name, weight) pairs")
        name, raw_weight = entry
        if name not in DIRECT_OPPONENTS:
            raise ValueError(
                f"unknown opponent {name!r}; choose from {', '.join(DIRECT_OPPONENTS)}"
            )
        if name in seen:
            raise ValueError(f"duplicate opponent in mixture: {name}")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
            raise ValueError(f"opponent weight for {name} must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"opponent weight for {name} must be positive and finite")
        seen.add(name)
        validated.append((name, weight))
    total = sum(weight for _, weight in validated)
    return tuple((name, weight / total) for name, weight in validated)


def _sample_opponent_name(mixture: OpponentMixture, rng: random.Random) -> str:
    normalized = normalized_opponent_mixture(mixture)
    draw = rng.random()
    cumulative = 0.0
    for name, weight in normalized:
        cumulative += weight
        if draw < cumulative:
            return name
    return normalized[-1][0]  # Float accumulation can end infinitesimally below one.


def _direct_opponent_policy(name: str, rng: random.Random) -> OpponentPolicy:
    if name == "strategic":
        return as_aec_policy(StrategicHeuristicPolicy())
    if name == "random":
        return as_aec_policy(SeededRandomPolicy(rng.randrange(2**63)))
    if name == "verardo":
        return as_aec_policy(VerardoReferencePolicy())
    raise ValueError(f"unknown opponent: {name}")


class OpponentMixtureSampler:
    """Episode-level weighted sampler driven only by the environment RNG."""

    def __init__(self, mixture: OpponentMixture) -> None:
        self.mixture = normalized_opponent_mixture(mixture)

    def __call__(self, rng: random.Random) -> OpponentPolicy:
        return _direct_opponent_policy(_sample_opponent_name(self.mixture, rng), rng)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    total_steps: int = 1_000_000
    iterations: int = 10
    n_steps: int = 1024
    batch_size: int = 512
    n_epochs: int = 8
    n_envs: int = 8
    vec_env: str = "dummy"
    device: str = "auto"
    cpu_threads: int = DEFAULT_CPU_THREADS
    save_dir: Path = Path("models")
    resume: Path | None = None
    initialize_from: Path | None = None
    fork_from: Path | None = None
    allow_environment_fork: bool = False
    allow_cross_profile_warm_start: bool = False
    selfplay: bool = False
    selfplay_prob: float = 0.75
    opponent_pool_size: int = 8
    opponent_mixture: OpponentMixture | None = None
    enable_bidding: bool = True
    enable_weis: bool = True
    enable_stock: bool = True
    trump_only_bidding: bool = False
    net_arch: tuple[int, ...] = (512, 256)
    control_team: bool = True
    ctde: bool = True
    randomize_starter: bool = True
    reward_scale: float = 0.001
    terminal_win_bonus: float = 1.0
    normalize_contract_reward: bool = False
    gamma: float = 1.0
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    mode: str | None = None
    trump_suit: str | None = None
    profile_name: str = "standard"

    @property
    def profile(self) -> RulesetConfig:
        return PROFILE_BY_NAME[self.profile_name]

    def validate(self) -> None:
        integer_fields = {
            "total_steps": self.total_steps,
            "iterations": self.iterations,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "n_envs": self.n_envs,
            "cpu_threads": self.cpu_threads,
            "opponent_pool_size": self.opponent_pool_size,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.iterations > self.total_steps:
            raise ValueError("iterations cannot exceed total_steps")
        if self.vec_env not in {"dummy", "subproc"}:
            raise ValueError("vec_env must be 'dummy' or 'subproc'")
        if self.profile_name not in PROFILE_BY_NAME:
            raise ValueError(f"unknown profile_name: {self.profile_name}")
        if not 0.0 <= self.selfplay_prob <= 1.0:
            raise ValueError("selfplay_prob must be between 0 and 1")
        normalized_mixture = (
            normalized_opponent_mixture(self.opponent_mixture)
            if self.opponent_mixture is not None
            else None
        )
        for name, value in {
            "trump_only_bidding": self.trump_only_bidding,
            "normalize_contract_reward": self.normalize_contract_reward,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if self.trump_only_bidding and not self.enable_bidding:
            raise ValueError("trump-only bidding requires bidding to be enabled")
        if self.normalize_contract_reward and not self.control_team:
            raise ValueError("contract-normalized rewards require --control-team")
        for name, value in {
            "reward_scale": self.reward_scale,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "ent_coef": self.ent_coef,
            "terminal_win_bonus": self.terminal_win_bonus,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite number")
        if self.reward_scale <= 0 or self.learning_rate <= 0:
            raise ValueError("reward_scale and learning_rate must be greater than zero")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")
        if not self.net_arch or any(width <= 0 for width in self.net_arch):
            raise ValueError("net_arch must contain positive layer widths")
        rollout_size = self.n_steps * self.n_envs
        if self.batch_size > rollout_size or rollout_size % self.batch_size:
            raise ValueError("batch_size must evenly divide n_steps * n_envs")
        if self.selfplay and not self.control_team:
            raise ValueError("self-play requires --control-team for coordinated team snapshots")
        if self.ctde and not self.control_team:
            raise ValueError("CTDE requires --control-team")
        starting_sources = sum(
            value is not None
            for value in (self.resume, self.initialize_from, self.fork_from)
        )
        if starting_sources > 1:
            raise ValueError(
                "--resume, --initialize-from, and --fork-from are mutually exclusive"
            )
        if self.selfplay and self.vec_env == "subproc":
            raise ValueError(
                "self-play uses a live opponent pool and therefore requires --vec-env dummy"
            )
        if self.mode not in (None, MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE):
            raise ValueError("mode must be trump, obeabe, uneufe, or omitted")
        # A fixed trump mode with no fixed suit intentionally asks the AEC
        # environment to sample one of the four suits on every reset.  Validate
        # concrete contracts here, while preserving that useful curriculum mode.
        if self.mode and not (self.mode == MODE_TRUMP and self.trump_suit is None):
            self.profile.contract_factor(self.mode, self.trump_suit)
        if self.profile_name == VERARDO_PROFILE_VERSION and (
            self.enable_weis or self.enable_stock
        ):
            raise ValueError("verardo-v1 training requires --no-weis and --no-stock")
        uses_verardo = normalized_mixture is not None and any(
            name == "verardo" for name, _ in normalized_mixture
        )
        verardo_can_play = (
            self.enable_bidding and self.trump_only_bidding
        ) or (
            not self.enable_bidding and self.mode == MODE_TRUMP
        )
        if uses_verardo and not verardo_can_play:
            raise ValueError(
                "the Verardo opponent requires trump-only bidding or a fixed trump game"
            )


def split_steps(total_steps: int, chunks: int) -> tuple[int, ...]:
    """Split an exact requested budget without silently dropping a remainder."""

    if total_steps <= 0 or chunks <= 0 or chunks > total_steps:
        raise ValueError("require total_steps >= chunks > 0")
    quotient, remainder = divmod(total_steps, chunks)
    return tuple(quotient + (index < remainder) for index in range(chunks))


def rollout_chunks(
    current_timesteps: int,
    target_timesteps: int,
    rollout_size: int,
    iterations: int,
) -> tuple[int, ...]:
    """Plan full PPO rollouts that reach or minimally exceed an absolute target."""

    if current_timesteps < 0 or target_timesteps <= current_timesteps:
        raise ValueError("target_timesteps must be greater than current_timesteps >= 0")
    if rollout_size <= 0 or iterations <= 0:
        raise ValueError("rollout_size and iterations must be positive")
    rollout_count = math.ceil((target_timesteps - current_timesteps) / rollout_size)
    counts = split_steps(rollout_count, min(iterations, rollout_count))
    return tuple(count * rollout_size for count in counts)


_RESUME_COMPATIBLE_FIELDS = (
    "seed",
    "n_steps",
    "batch_size",
    "n_epochs",
    "n_envs",
    "cpu_threads",
    "vec_env",
    "selfplay",
    "selfplay_prob",
    "opponent_pool_size",
    "opponent_mixture",
    "net_arch",
    "ctde",
    "gamma",
    "gae_lambda",
    "learning_rate",
    "ent_coef",
)


def assert_resume_training_compatible(
    manifest: dict[str, object], config: TrainConfig
) -> None:
    """Reject resume options that would be ignored or change the saved experiment."""

    recorded = manifest.get("training_config")
    if not isinstance(recorded, dict):
        raise ValueError("run manifest has no training_config mapping")
    mismatches: list[str] = []
    for field in _RESUME_COMPATIBLE_FIELDS:
        expected = getattr(config, field)
        if field == "opponent_mixture" and expected is not None:
            expected = [list(entry) for entry in expected]
        elif isinstance(expected, tuple):
            expected = list(expected)
        if field not in recorded:
            mismatches.append(f"{field} is unrecorded in the original manifest")
        elif recorded.get(field) != expected:
            mismatches.append(f"{field} {recorded.get(field)!r} != {expected!r}")
    if mismatches:
        raise ValueError("Resume training incompatibility: " + "; ".join(mismatches))


class OpponentPool:
    """Bounded historical-policy league shared by in-process environments."""

    def __init__(
        self,
        selfplay_prob: float,
        max_size: int,
        *,
        ctde: bool = False,
        fallback_mixture: OpponentMixture | None = None,
    ) -> None:
        self.selfplay_prob = selfplay_prob
        self.max_size = max_size
        self.ctde = ctde
        self._fallback_sampler = (
            OpponentMixtureSampler(fallback_mixture)
            if fallback_mixture is not None
            else None
        )
        self.checkpoints: list[Path] = []
        self._loaded: dict[Path, MaskablePPO] = {}

    def add(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved in self.checkpoints:
            return
        self.checkpoints.append(resolved)
        while len(self.checkpoints) > self.max_size:
            removed = self.checkpoints.pop(0)
            self._loaded.pop(removed, None)

    def _fallback_policy(self, rng: random.Random) -> OpponentPolicy:
        if self._fallback_sampler is not None:
            return self._fallback_sampler(rng)
        if rng.random() < 0.65:
            return as_aec_policy(StrategicHeuristicPolicy())
        return as_aec_policy(SeededRandomPolicy(rng.randrange(2**31)))

    def sample_policy(self, rng: random.Random) -> OpponentPolicy:
        if (
            not self.checkpoints
            or self.selfplay_prob <= 0.0
            or rng.random() >= self.selfplay_prob
        ):
            return self._fallback_policy(rng)
        checkpoint = rng.choice(self.checkpoints)
        model = self._loaded.get(checkpoint)
        if model is None:
            model = _load_model_preserving_rng(checkpoint, device="cpu")
            self._loaded[checkpoint] = model
        policy_rng = random.Random(rng.randrange(2**63))

        def _policy(env, agent: str) -> int:
            visible = env.observe(agent)
            observation = visible["observation"]
            if self.ctde:
                observation = pad_public_observation(observation)
            action = _predict_preserving_rng(
                model,
                observation,
                visible["action_mask"],
                seed=policy_rng.randrange(2**63),
            )
            return action

        return _policy


def environment_manifest(config: TrainConfig) -> dict[str, object]:
    if config.opponent_mixture is not None:
        mixture = normalized_opponent_mixture(config.opponent_mixture)
        mixture_source = "configured"
    elif config.selfplay:
        mixture = _LEGACY_SELFPLAY_FALLBACK_MIXTURE
        mixture_source = "legacy-selfplay-fallback"
    else:
        mixture = _DEFAULT_DIRECT_MIXTURE
        mixture_source = "default"
    return {
        "control_team": config.control_team,
        "baseline_opponent": (
            "strategic-public-v1"
            if mixture == _DEFAULT_DIRECT_MIXTURE
            else "weighted-public-mixture-v1"
        ),
        "opponent_mixture": [
            {"opponent": name, "weight": weight} for name, weight in mixture
        ],
        "opponent_mixture_source": mixture_source,
        "opponent_mixture_scope": "selfplay-fallback" if config.selfplay else "direct",
        "ctde": config.ctde,
        "actor_observation_shape": [OBS_SIZE],
        "training_observation_shape": [CTDE_OBS_SIZE if config.ctde else OBS_SIZE],
        "privileged_observation_schema": (
            {
                "name": PRIVILEGED_OBSERVATION_SCHEMA_NAME,
                "version": PRIVILEGED_OBSERVATION_SCHEMA_VERSION,
            }
            if config.ctde
            else None
        ),
        "enable_bidding": config.enable_bidding,
        "trump_only_bidding": config.trump_only_bidding,
        "enable_weis": config.enable_weis,
        "enable_stock": config.enable_stock,
        "mode": config.mode,
        "trump_suit": config.trump_suit,
        "randomize_starter": config.randomize_starter,
        "reward_scale": config.reward_scale,
        "terminal_win_bonus": config.terminal_win_bonus,
        "normalize_contract_reward": config.normalize_contract_reward,
        "profile": config.profile.to_dict(),
    }


def _build_env(
    config: TrainConfig,
    opponent_sampler: Callable[[random.Random], OpponentPolicy] | None = None,
    seed: int | None = None,
):
    env_seed = seed if seed is not None else config.seed
    if config.control_team:
        baseline_opponent = as_aec_policy(StrategicHeuristicPolicy())
        return JassTeamEnv(
            seed=env_seed,
            enable_bidding=config.enable_bidding,
            enable_weis=config.enable_weis,
            enable_stock=config.enable_stock,
            trump_only_bidding=config.trump_only_bidding,
            profile=config.profile,
            mode=config.mode,
            trump_suit=config.trump_suit,
            opponent_policy=baseline_opponent,
            opponent_sampler=opponent_sampler,
            randomize_starter=config.randomize_starter,
            reward_scale=config.reward_scale,
            terminal_win_bonus=config.terminal_win_bonus,
            normalize_contract_reward=config.normalize_contract_reward,
            privileged_critic=config.ctde,
        )
    return JassSingleAgentEnv(
        seed=env_seed,
        enable_bidding=config.enable_bidding,
        enable_weis=config.enable_weis,
        enable_stock=config.enable_stock,
        trump_only_bidding=config.trump_only_bidding,
        profile=config.profile,
        mode=config.mode,
        trump_suit=config.trump_suit,
        opponent_policy=as_aec_policy(StrategicHeuristicPolicy()),
        partner_policy=as_aec_policy(StrategicHeuristicPolicy()),
        opponent_sampler=opponent_sampler,
    )


def _make_vec_env(config: TrainConfig, opponent_pool: OpponentPool | None):
    if opponent_pool is not None:
        sampler: Callable[[random.Random], OpponentPolicy] | None = opponent_pool.sample_policy
    elif config.opponent_mixture is not None:
        sampler = OpponentMixtureSampler(config.opponent_mixture)
    else:
        sampler = None
    env_fns = [
        lambda rank=rank: _build_env(config, opponent_sampler=sampler, seed=config.seed + rank)
        for rank in range(config.n_envs)
    ]
    vector = SubprocVecEnv(env_fns) if config.vec_env == "subproc" else DummyVecEnv(env_fns)
    return VecMonitor(vector)


def _resume_model_path(resume: Path) -> tuple[Path, Path]:
    if resume.is_file():
        return resume.parent, resume
    if not resume.is_dir():
        raise FileNotFoundError(f"resume path does not exist: {resume}")
    manifest = load_manifest(resume)
    actual = manifest.get("actual_timesteps", 0)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
        raise ValueError("run manifest actual_timesteps must be a non-negative integer")
    final = resume / "model_final.zip"
    if manifest.get("status") == "trained" and final.exists():
        return resume, final
    recorded_checkpoints = manifest.get("checkpoints", [])
    if not isinstance(recorded_checkpoints, list):
        raise ValueError("run manifest checkpoints must be a list")
    checkpoints = []
    for name in recorded_checkpoints:
        if not isinstance(name, str):
            continue
        checkpoint = resume / name
        timestep = _checkpoint_timestep(checkpoint)
        if checkpoint.is_file() and timestep is not None and timestep <= actual:
            checkpoints.append(checkpoint)
    if not checkpoints:
        if final.exists() and actual == 0:
            return resume, final
        raise FileNotFoundError(f"no checkpoint at or before timestep {actual} in {resume}")
    return resume, max(checkpoints, key=lambda path: _checkpoint_timestep(path) or -1)


def _new_run_dir(save_dir: Path) -> Path:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    candidate = save_dir / stamp
    suffix = 1
    while candidate.exists():
        candidate = save_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _checkpoint_timestep(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _capture_global_rng_state(*, include_accelerators: bool = True):
    import torch

    accelerator_states: dict[str, object] = {}
    if include_accelerators and torch.backends.mps.is_available():
        accelerator_states["mps"] = torch.mps.get_rng_state()
    if include_accelerators and torch.cuda.is_available():
        accelerator_states["cuda"] = torch.cuda.get_rng_state_all()
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state(),
        accelerator_states,
    )


def _restore_global_rng_state(state) -> None:
    import torch

    python_state, numpy_state, torch_state, accelerator_states = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    if "mps" in accelerator_states:
        torch.mps.set_rng_state(accelerator_states["mps"])
    if "cuda" in accelerator_states:
        torch.cuda.set_rng_state_all(accelerator_states["cuda"])


def _load_model_preserving_rng(path: Path, *, device: str):
    state = _capture_global_rng_state()
    try:
        return MaskablePPO.load(path, device=device)
    finally:
        _restore_global_rng_state(state)


def _predict_preserving_rng(
    model,
    observation: np.ndarray,
    action_mask: np.ndarray,
    *,
    seed: int,
) -> int:
    import torch

    state = _capture_global_rng_state(include_accelerators=False)
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(seed)
        torch.random.set_rng_state(cpu_generator.get_state())
        action, _ = model.predict(
            observation,
            action_masks=action_mask,
            deterministic=False,
        )
        encoded = np.asarray(action)
        if encoded.size != 1:
            raise ValueError("opponent model must return exactly one action")
        return int(encoded.reshape(-1)[0])
    finally:
        _restore_global_rng_state(state)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest_model_hash(path: Path, manifest: dict[str, object]) -> None:
    expected: object = None
    if path.name == manifest.get("final_model"):
        expected = manifest.get("final_model_sha256")
    checkpoint_hashes = manifest.get("checkpoint_sha256")
    if expected is None and isinstance(checkpoint_hashes, dict):
        expected = checkpoint_hashes.get(path.name)
    if not isinstance(expected, str):
        raise ValueError(f"run manifest has no SHA-256 for resume artifact {path.name}")
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"resume artifact SHA-256 {actual} does not match manifest {expected}"
        )


def _chunk_seed(base_seed: int, timestep: int) -> int:
    return (
        base_seed * 6_364_136_223_846_793_005
        + timestep * 1_442_695_040_888_963_407
    ) & ((1 << 32) - 1)


def _atomic_model_save(model, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    model.save(temporary)
    temporary.replace(path)


def _copy_actor_warm_start(source, target) -> None:
    source_policy = source.policy
    target_policy = target.policy
    if type(source_policy) is not type(target_policy):
        raise ValueError(
            f"warm-start policy type {type(source_policy).__name__} does not match "
            f"training policy {type(target_policy).__name__}"
        )
    source_shape = tuple(getattr(source.observation_space, "shape", ()))
    target_shape = tuple(getattr(target.observation_space, "shape", ()))
    if source_shape != target_shape:
        raise ValueError(f"warm-start observation shape {source_shape} != {target_shape}")
    if getattr(source.action_space, "n", None) != getattr(target.action_space, "n", None):
        raise ValueError("warm-start action space does not match training")

    module_paths = ("mlp_extractor.policy_net", "action_net")
    for module_path in module_paths:
        source_module = source_policy
        target_module = target_policy
        for component in module_path.split("."):
            source_module = getattr(source_module, component)
            target_module = getattr(target_module, component)
        try:
            target_module.load_state_dict(source_module.state_dict(), strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"warm-start actor architecture differs at {module_path}: {exc}"
            ) from exc


def _load_imitation_warm_start(
    path: Path,
    target,
    *,
    expected_environment: dict[str, object] | None = None,
    allow_cross_profile: bool = False,
) -> dict[str, object]:
    model_path = path.resolve()
    manifest_path = model_path.parent / "imitation_manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            "--initialize-from requires a behavior-cloned model and adjacent "
            "imitation_manifest.json"
        )
    import json

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("model_path") != model_path.name:
        raise ValueError("imitation manifest does not identify a completed warm-start model")
    expected_hash = manifest.get("model_sha256")
    actual_hash = _file_sha256(model_path)
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        raise ValueError("imitation model hash does not match its manifest")
    source_environment = manifest.get("identity", {}).get("dataset_environment")
    comparable_source = None
    comparable_target = None
    if expected_environment is not None:
        comparable_source = (
            {
                key: source_environment.get(key)
                for key in ("enable_bidding", "enable_weis", "enable_stock", "profile")
            }
            if isinstance(source_environment, dict)
            else None
        )
        comparable_target = {
            key: expected_environment[key]
            for key in ("enable_bidding", "enable_weis", "enable_stock", "profile")
        }
        if comparable_source != comparable_target and not allow_cross_profile:
            raise ValueError(
                "imitation dataset rules/profile differ from PPO training; pass the explicit "
                "cross-profile override only for an intentional curriculum"
            )
    source = _load_model_preserving_rng(model_path, device="cpu")
    _copy_actor_warm_start(source, target)
    return {
        "kind": "behavior-clone-actor",
        "model_path": str(model_path),
        "model_sha256": actual_hash,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "dataset_id": manifest.get("identity", {}).get("dataset_id"),
        "epochs_completed": manifest.get("epochs_completed"),
        "dataset_environment": source_environment,
        "cross_profile_override": bool(
            expected_environment is not None
            and comparable_source != comparable_target
            and allow_cross_profile
        ),
    }


_FORK_ENVIRONMENT_COMPATIBILITY_KEYS = (
    "control_team",
    "ctde",
    "actor_observation_shape",
    "training_observation_shape",
    "privileged_observation_schema",
    "enable_bidding",
    "trump_only_bidding",
    "enable_weis",
    "enable_stock",
    "mode",
    "trump_suit",
    "randomize_starter",
    "reward_scale",
    "terminal_win_bonus",
    "normalize_contract_reward",
    "profile",
)


def _fork_environment_mismatches(
    source_environment: dict[str, object],
    expected_environment: dict[str, object],
) -> list[str]:
    """Compare fork semantics while treating absent new booleans as legacy false."""

    false_by_default = {"trump_only_bidding", "normalize_contract_reward"}
    return [
        key
        for key in _FORK_ENVIRONMENT_COMPATIBILITY_KEYS
        if source_environment.get(key, False if key in false_by_default else None)
        != expected_environment.get(key)
    ]


def _load_run_fork(
    path: Path,
    target,
    *,
    expected_environment: dict[str, object],
    allow_environment_change: bool = False,
) -> dict[str, object]:
    model_path = path.resolve()
    manifest = load_manifest(model_path)
    _verify_manifest_model_hash(model_path, manifest)
    source_environment = manifest.get("environment")
    if not isinstance(source_environment, dict):
        raise ValueError("source run manifest has no environment mapping")
    mismatches = _fork_environment_mismatches(source_environment, expected_environment)
    if mismatches and not allow_environment_change:
        raise ValueError(f"forked run environment differs in {mismatches}")
    source = _load_model_preserving_rng(model_path, device="cpu")
    if type(source.policy) is not type(target.policy):
        raise ValueError("forked run policy type does not match target policy")
    try:
        target.policy.load_state_dict(source.policy.state_dict(), strict=True)
    except RuntimeError as exc:
        raise ValueError(f"forked run architecture differs: {exc}") from exc
    manifest_path = model_path.parent / MANIFEST_FILENAME
    return {
        "kind": "ppo-policy-fork",
        "model_path": str(model_path),
        "model_sha256": _file_sha256(model_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "source_timesteps": int(source.num_timesteps),
        "environment_override": bool(mismatches),
        "environment_mismatches": mismatches,
    }


def train(config: TrainConfig) -> Path:
    config.validate()
    workflow_started = time.perf_counter()
    import torch

    # PyTorch otherwise chooses a host-dependent value (six on the supported
    # M3 Pro), making resource use and throughput less reproducible.  Configure
    # it before constructing either the policy or optimizer.
    torch.set_num_threads(config.cpu_threads)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    opponent_pool = (
        OpponentPool(
            config.selfplay_prob,
            config.opponent_pool_size,
            ctde=config.ctde,
            fallback_mixture=config.opponent_mixture,
        )
        if config.selfplay
        else None
    )
    env = _make_vec_env(config, opponent_pool)
    try:
        observation_shape = tuple(env.observation_space.shape)
        environment = environment_manifest(config)
        repo_root = Path(__file__).resolve().parents[1]

        resumed_from: str | None = None
        if config.resume is not None:
            run_dir, model_path = _resume_model_path(config.resume)
            existing = load_manifest(run_dir)
            _verify_manifest_model_hash(model_path, existing)
            original_git = existing.get("git")
            current_git = git_metadata(repo_root)
            if not isinstance(original_git, dict) or original_git.get(
                "working_tree_sha256"
            ) != current_git.get("working_tree_sha256"):
                raise ValueError(
                    "working tree changed since this run started; refusing mixed-lineage resume"
                )
            assert_compatible(
                existing,
                observation_schema_version=OBSERVATION_SCHEMA_VERSION,
                observation_shape=observation_shape,
                action_count=ACTION_COUNT,
                environment=environment,
            )
            assert_resume_training_compatible(existing, config)
            model = _load_model_preserving_rng(model_path, device=config.device)
            if int(model.num_timesteps) != int(existing.get("actual_timesteps", -1)):
                raise ValueError(
                    f"loaded model has {model.num_timesteps} timesteps but manifest records "
                    f"{existing.get('actual_timesteps')}"
                )
            model.set_env(env)
            manifest = existing
            resumed_from = str(model_path)
            recorded_actual = existing.get("actual_timesteps", 0)
            selected_timestep = (
                int(model.num_timesteps)
                if model_path.name == "model_final.zip"
                else _checkpoint_timestep(model_path)
            )
            if selected_timestep != recorded_actual:
                raise ValueError(
                    f"resume artifact has {selected_timestep} timesteps but manifest records "
                    f"{recorded_actual}; refusing to lose or fork progress"
                )
            if opponent_pool is not None:
                for name in existing.get("checkpoints", []):
                    if not isinstance(name, str):
                        continue
                    checkpoint = run_dir / name
                    timestep = _checkpoint_timestep(checkpoint)
                    if (
                        checkpoint.is_file()
                        and timestep is not None
                        and timestep <= model.num_timesteps
                    ):
                        _verify_manifest_model_hash(checkpoint, existing)
                        opponent_pool.add(checkpoint)
            prior_error = manifest.pop("error", None)
            if prior_error is not None:
                manifest["prior_errors"] = [
                    *manifest.get("prior_errors", []),
                    prior_error,
                ]
            manifest["resume_events"] = [
                *manifest.get("resume_events", []),
                {
                    "at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "artifact": str(model_path),
                    "target_timesteps": config.total_steps,
                    "git": current_git,
                },
            ]
        else:
            run_dir = _new_run_dir(config.save_dir)
            policy = CTDEMaskableActorCriticPolicy if config.ctde else "MlpPolicy"
            model = MaskablePPO(
                policy,
                env,
                verbose=1,
                seed=config.seed,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                learning_rate=config.learning_rate,
                ent_coef=config.ent_coef,
                device=config.device,
                policy_kwargs={"net_arch": list(config.net_arch)},
            )
            if config.initialize_from is not None:
                warm_start = _load_imitation_warm_start(
                    config.initialize_from,
                    model,
                    expected_environment=environment,
                    allow_cross_profile=config.allow_cross_profile_warm_start,
                )
            elif config.fork_from is not None:
                warm_start = _load_run_fork(
                    config.fork_from,
                    model,
                    expected_environment=environment,
                    allow_environment_change=config.allow_environment_fork,
                )
            else:
                warm_start = None
            manifest = build_manifest(
                repo_root=repo_root,
                training_config=config,
                environment=environment,
                observation_schema_version=OBSERVATION_SCHEMA_VERSION,
                observation_shape=observation_shape,
                action_count=ACTION_COUNT,
            )
            manifest["warm_start"] = warm_start
            manifest["resume_events"] = []
    except BaseException:
        env.close()
        raise

    try:
        chunks = rollout_chunks(
            int(model.num_timesteps),
            config.total_steps,
            config.n_steps * config.n_envs,
            config.iterations,
        )
    except BaseException:
        env.close()
        raise
    planned_total = int(model.num_timesteps) + sum(chunks)

    manifest.update(
        {
            "status": "training",
            "resumed_from": resumed_from,
            "requested_total_timesteps": config.total_steps,
            "planned_total_timesteps": planned_total,
            "rollout_size": config.n_steps * config.n_envs,
            "actual_timesteps": int(model.num_timesteps),
            "checkpoints": manifest.get("checkpoints", []),
            "checkpoint_sha256": manifest.get("checkpoint_sha256", {}),
            "final_model_stale": (run_dir / "model_final.zip").exists(),
            "current_training_config": dict(vars(config)),
        }
    )
    manifest_path = run_dir / MANIFEST_FILENAME
    write_manifest(manifest_path, manifest)
    durable_timesteps = int(model.num_timesteps)
    invocation_start_timesteps = durable_timesteps
    learning_seconds = 0.0

    def record_resource_usage() -> None:
        elapsed = time.perf_counter() - workflow_started
        trained = int(model.num_timesteps) - invocation_start_timesteps
        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        manifest["resource_usage"] = {
            "device": str(getattr(model, "device", config.device)),
            "cpu_threads": int(torch.get_num_threads()),
            "workflow_seconds": elapsed,
            "learning_seconds": learning_seconds,
            "trained_timesteps_this_invocation": trained,
            "steps_per_learning_second": (
                trained / learning_seconds if learning_seconds > 0.0 else None
            ),
            "peak_rss_bytes": int(raw_peak if sys.platform == "darwin" else raw_peak * 1024),
        }

    try:
        for chunk in chunks:
            before = int(model.num_timesteps)
            seed_for_chunk = _chunk_seed(config.seed, before)
            model.set_random_seed(seed_for_chunk)
            # Every checkpoint chunk starts at a deterministic episode boundary.
            # This makes interruption/resume equivalent to an uninterrupted run
            # with the same chunk plan without serializing private environment state.
            model._last_obs = None
            model._last_episode_starts = None
            learning_started = time.perf_counter()
            try:
                model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            finally:
                learning_seconds += time.perf_counter() - learning_started
            if model.num_timesteps != before + chunk:
                raise RuntimeError(
                    f"MaskablePPO advanced {model.num_timesteps - before} steps; "
                    f"expected one planned chunk of {chunk}"
                )
            checkpoint = run_dir / f"checkpoint_{model.num_timesteps:012d}.zip"
            _atomic_model_save(model, checkpoint)
            if opponent_pool is not None:
                opponent_pool.add(checkpoint)
            manifest["actual_timesteps"] = int(model.num_timesteps)
            manifest["checkpoints"] = [
                *manifest.get("checkpoints", []),
                checkpoint.name,
            ]
            manifest["checkpoint_sha256"] = {
                **manifest.get("checkpoint_sha256", {}),
                checkpoint.name: _file_sha256(checkpoint),
            }
            manifest["latest_checkpoint"] = checkpoint.name
            manifest["last_chunk_seed"] = seed_for_chunk
            record_resource_usage()
            durable_timesteps = int(model.num_timesteps)
            write_manifest(manifest_path, manifest)

        final_path = run_dir / "model_final.zip"
        _atomic_model_save(model, final_path)
        manifest["actual_timesteps"] = int(model.num_timesteps)
        manifest["status"] = "trained"
        manifest["final_model"] = final_path.name
        manifest["final_model_sha256"] = _file_sha256(final_path)
        manifest["final_model_stale"] = False
        record_resource_usage()
        write_manifest(manifest_path, manifest)
        return final_path
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["last_attempted_timesteps"] = int(model.num_timesteps)
        manifest["actual_timesteps"] = durable_timesteps
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        record_resource_usage()
        write_manifest(manifest_path, manifest)
        raise
    finally:
        env.close()


def _parse_opponent_mixture(value: str) -> OpponentMixture:
    entries: list[tuple[str, float]] = []
    for raw_entry in value.split(","):
        name, separator, raw_weight = raw_entry.strip().partition("=")
        if not separator or not name or not raw_weight:
            raise ValueError("use NAME=WEIGHT pairs separated by commas")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"invalid weight for {name!r}: {raw_weight!r}") from exc
        entries.append((name, weight))
    mixture = tuple(entries)
    normalized_opponent_mixture(mixture)
    return mixture


def _parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a manifested MaskablePPO Jass team")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help="PyTorch intra-op workers (default: min(8, available CPUs))",
    )
    parser.add_argument("--save-dir", default="models")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument("--fork-from")
    parser.add_argument("--allow-environment-fork", action="store_true")
    parser.add_argument("--allow-cross-profile-warm-start", action="store_true")
    parser.add_argument("--selfplay", action="store_true")
    parser.add_argument("--selfplay-prob", type=float, default=0.75)
    parser.add_argument("--opponent-pool-size", type=int, default=8)
    parser.add_argument(
        "--opponent-mixture",
        help="episode weights, for example strategic=0.7,random=0.2,verardo=0.1",
    )
    parser.add_argument("--no-bidding", action="store_true")
    parser.add_argument("--trump-only-bidding", action="store_true")
    parser.add_argument("--no-weis", action="store_true")
    parser.add_argument("--no-stock", action="store_true")
    parser.add_argument("--single-seat", action="store_true")
    parser.add_argument("--no-ctde", action="store_true")
    parser.add_argument("--fixed-starter", action="store_true")
    parser.add_argument("--reward-scale", type=float, default=0.001)
    parser.add_argument("--terminal-win-bonus", type=float, default=1.0)
    parser.add_argument("--normalize-contract-reward", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", default="512,256")
    parser.add_argument("--mode", choices=[MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE])
    parser.add_argument("--trump-suit", choices=SUITS)
    parser.add_argument("--profile", choices=sorted(PROFILE_BY_NAME), default="standard")
    args = parser.parse_args(argv)

    try:
        net_arch = tuple(int(value.strip()) for value in args.net_arch.split(",") if value.strip())
    except ValueError as exc:
        raise SystemExit("--net-arch must be comma-separated integers") from exc
    try:
        opponent_mixture = (
            _parse_opponent_mixture(args.opponent_mixture)
            if args.opponent_mixture is not None
            else None
        )
    except ValueError as exc:
        parser.error(f"invalid --opponent-mixture: {exc}")

    config = TrainConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        iterations=args.iterations,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        device=args.device,
        cpu_threads=args.cpu_threads,
        save_dir=Path(args.save_dir),
        resume=Path(args.resume) if args.resume else None,
        initialize_from=(Path(args.initialize_from) if args.initialize_from else None),
        fork_from=Path(args.fork_from) if args.fork_from else None,
        allow_environment_fork=args.allow_environment_fork,
        allow_cross_profile_warm_start=args.allow_cross_profile_warm_start,
        selfplay=args.selfplay,
        selfplay_prob=args.selfplay_prob,
        opponent_pool_size=args.opponent_pool_size,
        opponent_mixture=opponent_mixture,
        enable_bidding=not args.no_bidding,
        trump_only_bidding=args.trump_only_bidding,
        enable_weis=not args.no_weis,
        enable_stock=not args.no_stock,
        net_arch=net_arch,
        control_team=not args.single_seat,
        ctde=not args.no_ctde,
        randomize_starter=not args.fixed_starter,
        reward_scale=args.reward_scale,
        terminal_win_bonus=args.terminal_win_bonus,
        normalize_contract_reward=args.normalize_contract_reward,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        mode=args.mode,
        trump_suit=args.trump_suit,
        profile_name=args.profile,
    )
    try:
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    return config


def main() -> None:
    final_path = train(_parse_args())
    print(f"Saved model and manifest to {final_path}")


if __name__ == "__main__":
    main()
