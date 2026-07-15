"""Resource-bounded behavior cloning for public-information Jass policies.

The collector writes compressed shards containing only the canonical public
observation, the legal-action mask, and the expert's legal action.  Complete
games are the unit of persistence, which makes collection safely resumable
without duplicating or dropping decisions after an interruption.

The trainer streams one shard at a time and optimizes only the actor side of a
MaskablePPO policy with masked cross-entropy.  Both ordinary MaskablePPO and
the public-actor/privileged-critic policy are supported; CTDE inputs receive a
zero private suffix during imitation, exactly as they do at inference.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from core.cards import ALL_CARDS
from core.ruleset import STANDARD_RULES_PROFILE, RulesetConfig
from env.jass_aec_env import (
    ACTION_COUNT,
    OBS_CARD_COUNT,
    OBS_HAND_OFFSET,
    OBS_SIZE,
    OBSERVATION_SCHEMA_NAME,
    OBSERVATION_SCHEMA_VERSION,
    JassAECEnv,
)
from rl.baselines import PublicPolicy, StrategicHeuristicPolicy
from rl.privileged_observation import CTDE_OBS_SIZE, pad_public_observation
from rl.reference_verardo import (
    VERARDO_PROFILE_VERSION,
    VERARDO_V1_PROFILE,
    TrumpOnlyPolicy,
    VerardoReferencePolicy,
)
from rl.run_manifest import git_metadata, runtime_metadata
from rl.search_policy import PIMCConfig, PIMCSearchPolicy

DATASET_FORMAT_VERSION = 2
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
IMITATION_FORMAT_VERSION = 1
IMITATION_MANIFEST_FILENAME = "imitation_manifest.json"
IMITATION_CHECKPOINT_FILENAME = "imitation_checkpoint.pt"
IMITATION_BEST_POLICY_FILENAME = "imitation_best_policy.pt"
IMITATION_MODEL_FILENAME = "model_imitation"

# Float16 exactly represents every binary field and is more than adequate for
# the few bounded score fractions.  At 3,260 bytes per sample, a 4,096-sample
# shard has a raw footprint of roughly 13 MiB before compression.
OBSERVATION_STORAGE_DTYPE = np.dtype(np.float16)
ACTION_STORAGE_DTYPE = np.dtype(np.uint8)
SAMPLE_ID_DTYPE = np.dtype(np.uint64)
DEFAULT_SHARD_SIZE = 4_096
MAX_SHARD_RAW_BYTES = 256 * 1024**2
RESOURCE_SAFE_RAW_DATASET_LIMIT = 16 * 1024**3
MAX_DECISIONS_PER_GAME = 42

ProfileName = Literal["full", "verardo-v1"]


class MaskableModel(Protocol):
    """The small MaskablePPO surface required by the trainer."""

    policy: Any
    observation_space: Any
    action_space: Any
    device: Any

    def save(self, path: str | Path) -> None: ...


@dataclass(frozen=True)
class CollectionConfig:
    """Configuration for deterministic, streaming expert collection."""

    output_dir: Path
    games: int
    seed: int = 0
    profile_name: ProfileName = "full"
    shard_size: int = DEFAULT_SHARD_SIZE
    raw_dataset_limit_bytes: int = RESOURCE_SAFE_RAW_DATASET_LIMIT
    resume: bool = False
    verify_determinism: bool = True
    expert_id: str | None = None

    def validate(self) -> None:
        if isinstance(self.games, bool) or not isinstance(self.games, int) or self.games <= 0:
            raise ValueError("games must be a positive integer")
        if self.profile_name not in ("full", VERARDO_PROFILE_VERSION):
            raise ValueError("profile_name must be 'full' or 'verardo-v1'")
        if (
            isinstance(self.shard_size, bool)
            or not isinstance(self.shard_size, int)
            or self.shard_size <= 0
        ):
            raise ValueError("shard_size must be a positive integer")
        if self.raw_dataset_limit_bytes <= 0:
            raise ValueError("raw_dataset_limit_bytes must be positive")
        if self.raw_dataset_limit_bytes > RESOURCE_SAFE_RAW_DATASET_LIMIT:
            raise ValueError(
                "raw_dataset_limit_bytes may not exceed the 16 GiB local-resource guard"
            )
        maximum_buffered_samples = self.shard_size + MAX_DECISIONS_PER_GAME
        if maximum_buffered_samples * _raw_sample_bytes() > MAX_SHARD_RAW_BYTES:
            raise ValueError("shard_size would exceed the 256 MiB in-memory shard guard")
        projected = self.games * MAX_DECISIONS_PER_GAME * _raw_sample_bytes()
        if projected > self.raw_dataset_limit_bytes:
            raise ValueError(
                f"worst-case raw dataset size {projected} exceeds configured limit "
                f"{self.raw_dataset_limit_bytes}; reduce games or raise the limit within 16 GiB"
            )


@dataclass(frozen=True)
class BehaviorCloneConfig:
    """Actor-only masked cross-entropy training configuration."""

    dataset_dir: Path
    output_dir: Path
    epochs: int = 8
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    validation_fraction: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "auto"
    resume: bool = False

    def validate(self) -> None:
        for name, value in {"epochs": self.epochs, "batch_size": self.batch_size}.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "validation_fraction": self.validation_fraction,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than zero")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between zero and 0.5")


@dataclass(frozen=True)
class DatasetSummary:
    games: int
    samples: int
    shards: int
    raw_bytes: int
    compressed_bytes: int
    dataset_id: str


@dataclass(frozen=True)
class BehaviorCloneReport:
    model_path: Path
    epochs_completed: int
    samples: int
    final_train_loss: float
    final_train_accuracy: float
    final_validation_loss: float
    final_validation_accuracy: float


@dataclass(frozen=True)
class _Sample:
    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    sample_id: int
    agent: int


def _raw_sample_bytes() -> int:
    return (
        OBS_SIZE * OBSERVATION_STORAGE_DTYPE.itemsize
        + ACTION_COUNT * ACTION_STORAGE_DTYPE.itemsize
        + ACTION_STORAGE_DTYPE.itemsize
        + SAMPLE_ID_DTYPE.itemsize
        + ACTION_STORAGE_DTYPE.itemsize
    )


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _policy_identity(policy: PublicPolicy, explicit: str | None) -> dict[str, Any]:
    policy_type = type(policy)
    identity: dict[str, Any] = {
        "id": explicit or f"{policy_type.__module__}.{policy_type.__qualname__}",
        "type": f"{policy_type.__module__}.{policy_type.__qualname__}",
    }
    public_configuration = {
        key: _jsonable(value)
        for key, value in vars(policy).items()
        if not key.startswith("_")
        and (
            dataclasses.is_dataclass(value)
            or isinstance(value, (str, int, float, bool, type(None)))
        )
    }
    if public_configuration:
        identity["configuration"] = public_configuration
    nested = vars(policy).get("policy")
    if callable(nested) and nested is not policy:
        identity["wrapped_policy"] = _policy_identity(nested, None)
    return identity


def _environment_spec(profile_name: ProfileName) -> tuple[RulesetConfig, dict[str, Any]]:
    if profile_name == VERARDO_PROFILE_VERSION:
        profile = VERARDO_V1_PROFILE
        options = {"enable_bidding": True, "enable_weis": False, "enable_stock": False}
    else:
        profile = STANDARD_RULES_PROFILE
        options = {"enable_bidding": True, "enable_weis": True, "enable_stock": True}
    return profile, {
        **options,
        "profile_name": profile_name,
        "profile": profile.to_dict(),
        "starter_rotation": "game_index_mod_4",
    }


def _collection_identity(config: CollectionConfig, policy: PublicPolicy) -> dict[str, Any]:
    _, environment = _environment_spec(config.profile_name)
    return {
        "seed": config.seed,
        "profile_name": config.profile_name,
        "shard_size": config.shard_size,
        "observation_storage_dtype": OBSERVATION_STORAGE_DTYPE.name,
        "compression": "numpy savez_compressed",
        "observation_scope": "canonical_public_only",
        "observation_schema": OBSERVATION_SCHEMA_NAME,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_shape": [OBS_SIZE],
        "action_count": ACTION_COUNT,
        "expert": _policy_identity(policy, config.expert_id),
        "environment": environment,
    }


def _new_dataset_manifest(config: CollectionConfig, policy: PublicPolicy) -> dict[str, Any]:
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "collecting",
        "target_games": config.games,
        "completed_games": 0,
        "samples": 0,
        "raw_bytes": 0,
        "compressed_bytes": 0,
        "dataset_id": None,
        "game_digests": [],
        "identity": _collection_identity(config, policy),
        "shards": [],
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "runtime": runtime_metadata(),
    }


def load_dataset_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / DATASET_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing {DATASET_MANIFEST_FILENAME} in {Path(directory)}")
    payload = json.loads(path.read_text())
    if payload.get("format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"unsupported dataset format {payload.get('format_version')!r}; "
            f"expected {DATASET_FORMAT_VERSION}"
        )
    return payload


def _reset_policy(policy: PublicPolicy, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        try:
            reset(seed)
        except TypeError:
            reset(seed=seed)


def _expected_private_hand(env: JassAECEnv, observer: int) -> Sequence[Any]:
    if env.state is not None:
        return env.state.hands[observer]
    if env._pending_hands is not None:  # noqa: SLF001 - collection privacy audit
        return env._pending_hands[observer]  # noqa: SLF001
    return ()


def _validate_public_sample(
    env: JassAECEnv,
    agent: str,
    observation: np.ndarray,
    action_mask: np.ndarray,
    action: int,
) -> None:
    """Validate legality and the independently checkable privacy boundary."""

    vector = np.asarray(observation)
    mask = np.asarray(action_mask)
    if vector.shape != (OBS_SIZE,):
        raise ValueError(
            f"private-observation leakage risk: expected public shape ({OBS_SIZE},), "
            f"got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError("public observation contains non-finite or out-of-bounds values")
    if mask.shape != (ACTION_COUNT,) or not np.all(np.isin(mask, (0, 1))):
        raise ValueError("action mask must be a binary canonical action vector")
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise ValueError("expert action must be an integer")
    if action < 0 or action >= ACTION_COUNT or mask[int(action)] != 1:
        raise ValueError(f"expert produced illegal action {action} for {agent}")

    observer = int(agent[1:])
    expected = np.zeros(OBS_CARD_COUNT, dtype=np.float32)
    card_index = {card: index for index, card in enumerate(ALL_CARDS)}
    for card in _expected_private_hand(env, observer):
        expected[card_index[card]] = 1.0
    stored_hand = vector[OBS_HAND_OFFSET : OBS_HAND_OFFSET + OBS_CARD_COUNT]
    if not np.array_equal(stored_hand, expected):
        raise ValueError(
            "public hand block is not exactly the acting player's hand; refusing "
            "a dataset that may contain another player's private cards"
        )


def _collect_game(
    policy: PublicPolicy,
    config: CollectionConfig,
    game_index: int,
) -> list[_Sample]:
    profile, environment = _environment_spec(config.profile_name)
    game_seed = config.seed + game_index
    _reset_policy(policy, game_seed)
    env = JassAECEnv(
        seed=game_seed,
        profile=profile,
        enable_bidding=bool(environment["enable_bidding"]),
        enable_weis=bool(environment["enable_weis"]),
        enable_stock=bool(environment["enable_stock"]),
        starter=game_index % 4,
    )
    env.reset(seed=game_seed)
    samples: list[_Sample] = []
    decision = 0
    while env.agents:
        agent = env.agent_selection
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
            continue
        visible = env.observe(agent)
        observation = visible["observation"]
        action_mask = visible["action_mask"]
        expert_action = policy(observation.copy(), action_mask.copy(), agent)
        _validate_public_sample(env, agent, observation, action_mask, expert_action)
        action = int(expert_action)
        if decision >= 256:
            raise RuntimeError("a game exceeded the sample-id decision capacity")
        samples.append(
            _Sample(
                observation=observation.astype(OBSERVATION_STORAGE_DTYPE, copy=True),
                action_mask=action_mask.astype(ACTION_STORAGE_DTYPE, copy=True),
                action=action,
                sample_id=(game_index << 8) | decision,
                agent=int(agent[1:]),
            )
        )
        decision += 1
        env.step(action)
    if not samples:
        raise RuntimeError("completed game produced no expert decisions")
    if len(samples) > MAX_DECISIONS_PER_GAME:
        raise RuntimeError(
            f"game produced {len(samples)} decisions, exceeding the audited maximum "
            f"{MAX_DECISIONS_PER_GAME}"
        )
    return samples


def _samples_digest(samples: Sequence[_Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        _update_sample_digest(
            digest,
            sample.observation,
            sample.action_mask,
            sample.action,
            sample.agent,
            sample.sample_id,
        )
    return digest.hexdigest()


def _update_sample_digest(
    digest: Any,
    observation: np.ndarray,
    action_mask: np.ndarray,
    action: int,
    agent: int,
    sample_id: int,
) -> None:
    digest.update(observation.tobytes())
    digest.update(action_mask.tobytes())
    digest.update(bytes((action, agent)))
    digest.update(sample_id.to_bytes(8, "little"))


def _write_shard(directory: Path, index: int, samples: Sequence[_Sample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot write an empty shard")
    path = directory / f"shard_{index:06d}.npz"
    temporary = path.with_suffix(".npz.tmp")
    observations = np.stack([sample.observation for sample in samples])
    masks = np.stack([sample.action_mask for sample in samples])
    actions = np.asarray([sample.action for sample in samples], dtype=ACTION_STORAGE_DTYPE)
    sample_ids = np.asarray([sample.sample_id for sample in samples], dtype=SAMPLE_ID_DTYPE)
    agents = np.asarray([sample.agent for sample in samples], dtype=ACTION_STORAGE_DTYPE)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            observations=observations,
            action_masks=masks,
            actions=actions,
            sample_ids=sample_ids,
            agents=agents,
        )
    temporary.replace(path)
    raw_bytes = sum(array.nbytes for array in (observations, masks, actions, sample_ids, agents))
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "samples": len(samples),
        "raw_bytes": raw_bytes,
        "compressed_bytes": path.stat().st_size,
        "first_sample_id": int(sample_ids[0]),
        "last_sample_id": int(sample_ids[-1]),
    }


def _dataset_id(manifest: Mapping[str, Any]) -> str:
    canonical = {
        "identity": manifest["identity"],
        "completed_games": manifest["completed_games"],
        "samples": manifest["samples"],
        "game_digests": list(manifest["game_digests"]),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _commit_shard(
    directory: Path,
    manifest: dict[str, Any],
    samples: Sequence[_Sample],
    completed_games: int,
    game_digests: Sequence[str],
) -> None:
    entry = _write_shard(directory, len(manifest["shards"]), samples)
    manifest["shards"].append(entry)
    manifest["completed_games"] = completed_games
    manifest["game_digests"].extend(game_digests)
    manifest["samples"] = sum(int(shard["samples"]) for shard in manifest["shards"])
    manifest["raw_bytes"] = sum(int(shard["raw_bytes"]) for shard in manifest["shards"])
    manifest["compressed_bytes"] = sum(
        int(shard["compressed_bytes"]) for shard in manifest["shards"]
    )
    manifest["dataset_id"] = _dataset_id(manifest)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(directory / DATASET_MANIFEST_FILENAME, manifest)


def _assert_resume_identity(
    manifest: Mapping[str, Any], config: CollectionConfig, policy: PublicPolicy
) -> None:
    expected = _collection_identity(config, policy)
    if manifest.get("identity") != expected:
        raise ValueError("collection resume configuration or expert does not match the dataset")
    completed = int(manifest.get("completed_games", -1))
    if config.games < completed:
        raise ValueError(
            f"target games {config.games} is below already completed games {completed}"
        )
    referenced = {str(entry["path"]) for entry in manifest.get("shards", [])}
    on_disk = {path.name for path in config.output_dir.glob("shard_*.npz")}
    if referenced != on_disk:
        raise ValueError("dataset has missing or orphaned shard files; refusing unsafe resume")


def collect_expert_dataset(
    policy: PublicPolicy,
    config: CollectionConfig,
) -> DatasetSummary:
    """Collect deterministic full-game expert trajectories into compressed shards."""

    config.validate()
    directory = config.output_dir
    manifest_path = directory / DATASET_MANIFEST_FILENAME
    if manifest_path.exists():
        if not config.resume:
            raise FileExistsError(
                f"{manifest_path} already exists; pass resume=True to continue it"
            )
        manifest = load_dataset_manifest(directory)
        _assert_resume_identity(manifest, config, policy)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"output directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        manifest = _new_dataset_manifest(config, policy)
        _atomic_json(manifest_path, manifest)

    start_game = int(manifest["completed_games"])
    manifest["target_games"] = config.games
    manifest["status"] = "collecting"
    manifest.pop("error", None)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(manifest_path, manifest)

    buffer: list[_Sample] = []
    buffered_game_digests: list[str] = []
    try:
        for game_index in range(start_game, config.games):
            samples = _collect_game(policy, config, game_index)
            if config.verify_determinism and game_index == 0:
                repeated = _collect_game(policy, config, game_index)
                if _samples_digest(samples) != _samples_digest(repeated):
                    raise ValueError(
                        "expert/environment replay was not deterministic for the same seed"
                    )
                samples = repeated
            buffer.extend(samples)
            buffered_game_digests.append(_samples_digest(samples))
            # Persist only after a complete game. The small possible overshoot
            # keeps every game atomic for exact resume semantics.
            if len(buffer) >= config.shard_size:
                _commit_shard(
                    directory,
                    manifest,
                    buffer,
                    game_index + 1,
                    buffered_game_digests,
                )
                buffer = []
                buffered_game_digests = []
        if buffer:
            _commit_shard(
                directory,
                manifest,
                buffer,
                config.games,
                buffered_game_digests,
            )
        elif config.games == start_game:
            manifest["completed_games"] = config.games

        manifest["status"] = "complete"
        manifest["target_games"] = config.games
        manifest["dataset_id"] = _dataset_id(manifest)
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
        raise

    return validate_dataset(directory)


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"observations", "action_masks", "actions", "sample_ids", "agents"}
        if set(archive.files) != expected:
            raise ValueError(f"{path.name} has unexpected arrays {sorted(archive.files)}")
        return {name: archive[name] for name in expected}


def _validate_shard_arrays(path: Path, arrays: Mapping[str, np.ndarray]) -> int:
    observations = arrays["observations"]
    masks = arrays["action_masks"]
    actions = arrays["actions"]
    sample_ids = arrays["sample_ids"]
    agents = arrays["agents"]
    count = len(actions)
    if count == 0:
        raise ValueError(f"{path.name} is empty")
    expected_shapes = {
        "observations": (count, OBS_SIZE),
        "action_masks": (count, ACTION_COUNT),
        "actions": (count,),
        "sample_ids": (count,),
        "agents": (count,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{path.name}:{name} shape {arrays[name].shape} != {shape}")
    if observations.dtype != OBSERVATION_STORAGE_DTYPE:
        raise ValueError(f"{path.name} observations must use float16 public storage")
    action_arrays = ("action_masks", "actions", "agents")
    if any(arrays[name].dtype != ACTION_STORAGE_DTYPE for name in action_arrays):
        raise ValueError(f"{path.name} action arrays must use uint8")
    if sample_ids.dtype != SAMPLE_ID_DTYPE:
        raise ValueError(f"{path.name} sample_ids must use uint64")
    invalid_observation = (
        not np.all(np.isfinite(observations))
        or np.any(observations < 0)
        or np.any(observations > 1)
    )
    if invalid_observation:
        raise ValueError(f"{path.name} has invalid public observation values")
    if not np.all(np.isin(masks, (0, 1))) or np.any(masks.sum(axis=1) == 0):
        raise ValueError(f"{path.name} has invalid action masks")
    if np.any(actions >= ACTION_COUNT) or np.any(agents >= 4):
        raise ValueError(f"{path.name} has out-of-range action or agent labels")
    if not np.all(masks[np.arange(count), actions.astype(np.int64)] == 1):
        raise ValueError(f"{path.name} contains an illegal expert label")
    if count > 1 and not np.all(sample_ids[1:] > sample_ids[:-1]):
        raise ValueError(f"{path.name} sample ids are not strictly increasing")
    return count


def validate_dataset(directory: str | Path) -> DatasetSummary:
    """Verify checksums, schemas, privacy width, and every expert label."""

    root = Path(directory)
    manifest = load_dataset_manifest(root)
    identity = manifest.get("identity", {})
    required_identity = {
        "observation_scope": "canonical_public_only",
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_shape": [OBS_SIZE],
        "action_count": ACTION_COUNT,
    }
    for key, expected in required_identity.items():
        if identity.get(key) != expected:
            raise ValueError(
                f"dataset privacy/schema mismatch: identity.{key}={identity.get(key)!r}, "
                f"expected {expected!r}"
            )

    total = 0
    raw_bytes = 0
    compressed_bytes = 0
    previous_id: int | None = None
    previous_game: int | None = None
    previous_decision: int | None = None
    current_game_digest = hashlib.sha256()
    calculated_game_digests: list[str] = []
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset has no shards")
    referenced: set[str] = set()
    for entry in entries:
        path = root / str(entry["path"])
        referenced.add(path.name)
        if not path.is_file():
            raise FileNotFoundError(f"missing dataset shard {path}")
        if _sha256(path) != entry.get("sha256"):
            raise ValueError(f"checksum mismatch for {path.name}")
        arrays = _load_shard(path)
        count = _validate_shard_arrays(path, arrays)
        first_id = int(arrays["sample_ids"][0])
        last_id = int(arrays["sample_ids"][-1])
        if previous_id is not None and first_id <= previous_id:
            raise ValueError("sample ids are duplicated or unordered across shards")
        previous_id = last_id
        for row, encoded_id in enumerate(arrays["sample_ids"]):
            sample_id = int(encoded_id)
            game, decision = divmod(sample_id, 256)
            if previous_game is None:
                if game != 0 or decision != 0:
                    raise ValueError("dataset must start at game zero, decision zero")
            elif game == previous_game:
                if decision != previous_decision + 1:
                    raise ValueError("dataset has a missing or duplicated in-game decision")
            elif game == previous_game + 1:
                if decision != 0:
                    raise ValueError("each collected game must begin at decision zero")
                calculated_game_digests.append(current_game_digest.hexdigest())
                current_game_digest = hashlib.sha256()
            else:
                raise ValueError("dataset has a missing or duplicated game")
            _update_sample_digest(
                current_game_digest,
                arrays["observations"][row],
                arrays["action_masks"][row],
                int(arrays["actions"][row]),
                int(arrays["agents"][row]),
                sample_id,
            )
            previous_game, previous_decision = game, decision
        if count != int(entry["samples"]):
            raise ValueError(f"sample count mismatch for {path.name}")
        if first_id != int(entry["first_sample_id"]) or last_id != int(entry["last_sample_id"]):
            raise ValueError(f"sample id range mismatch for {path.name}")
        shard_raw = sum(array.nbytes for array in arrays.values())
        if shard_raw != int(entry["raw_bytes"]):
            raise ValueError(f"raw byte count mismatch for {path.name}")
        total += count
        raw_bytes += shard_raw
        compressed_bytes += path.stat().st_size

    on_disk = {path.name for path in root.glob("shard_*.npz")}
    if on_disk != referenced:
        raise ValueError("dataset contains missing or untracked shards")
    if total != int(manifest.get("samples", -1)):
        raise ValueError("manifest total sample count does not match shards")
    if raw_bytes != int(manifest.get("raw_bytes", -1)):
        raise ValueError("manifest raw byte count does not match shards")
    if compressed_bytes != int(manifest.get("compressed_bytes", -1)):
        raise ValueError("manifest compressed byte count does not match shards")
    completed_games = int(manifest.get("completed_games", -1))
    if previous_game != completed_games - 1:
        raise ValueError("completed game count does not match encoded sample ids")
    calculated_game_digests.append(current_game_digest.hexdigest())
    if manifest.get("game_digests") != calculated_game_digests:
        raise ValueError("per-game content digests do not match dataset samples")
    expected_id = _dataset_id(manifest)
    if manifest.get("dataset_id") != expected_id:
        raise ValueError("dataset id does not match its identity and shard checksums")

    return DatasetSummary(
        games=completed_games,
        samples=total,
        shards=len(entries),
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        dataset_id=expected_id,
    )


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    mixed = values.astype(np.uint64, copy=True) ^ np.uint64(seed & ((1 << 64) - 1))
    mixed += np.uint64(0x9E3779B97F4A7C15)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return mixed ^ (mixed >> np.uint64(31))


def _validation_mask(sample_ids: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    threshold = int(fraction * (1 << 64))
    if threshold >= 1 << 64:
        return np.ones(sample_ids.shape, dtype=bool)
    return _splitmix64(sample_ids, seed) < np.uint64(threshold)


def _batch_iterator(
    dataset_dir: Path,
    manifest: Mapping[str, Any],
    *,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    epoch: int,
    validation: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    entries = list(manifest["shards"])
    rng = np.random.default_rng(seed + epoch * 1_000_003 + (17 if validation else 0))
    order = np.arange(len(entries))
    if not validation:
        rng.shuffle(order)
    for shard_index in order:
        arrays = _load_shard(dataset_dir / entries[int(shard_index)]["path"])
        is_validation = _validation_mask(
            arrays["sample_ids"], validation_fraction, seed
        )
        indices = np.flatnonzero(is_validation if validation else ~is_validation)
        if not validation:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            yield (
                arrays["observations"][selected].astype(np.float32),
                arrays["action_masks"][selected].astype(bool),
                arrays["actions"][selected].astype(np.int64),
            )


def _model_input(observations: np.ndarray, observation_shape: tuple[int, ...]) -> np.ndarray:
    if observation_shape == (OBS_SIZE,):
        return observations
    if observation_shape == (CTDE_OBS_SIZE,):
        # Exercise the same audited inference transform for the single-item
        # case and use its exact layout for a vectorized batch allocation.
        template = pad_public_observation(observations[0])
        padded = np.zeros((len(observations), template.size), dtype=np.float32)
        padded[:, :OBS_SIZE] = observations
        return padded
    raise ValueError(
        f"model observation shape {observation_shape} is neither public ({OBS_SIZE},) "
        f"nor CTDE ({CTDE_OBS_SIZE},)"
    )


def _actor_parameters(policy: Any) -> list[Any]:
    actor_extractor = getattr(policy, "pi_features_extractor", None)
    critic_extractor = getattr(policy, "vf_features_extractor", None)
    # A shared learned extractor is also part of the critic. Keep it frozen so
    # behavior cloning is strictly actor-only; non-shared actor extractors can
    # be optimized safely.
    feature_module = None if actor_extractor is critic_extractor else actor_extractor
    modules = (
        feature_module,
        getattr(getattr(policy, "mlp_extractor", None), "policy_net", None),
        getattr(policy, "action_net", None),
    )
    parameters: list[Any] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    if not parameters:
        raise ValueError("model policy exposes no trainable actor parameters")
    return parameters


def _resolve_device(model: MaskableModel, requested: str):
    import torch

    if requested == "auto":
        return torch.device(getattr(model, "device", "cpu"))
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _run_batches(
    model: MaskableModel,
    batches: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    device: Any,
    optimizer: Any | None,
    max_grad_norm: float,
    actor_parameters: Sequence[Any],
) -> tuple[float, float, int]:
    import torch

    policy = model.policy
    training = optimizer is not None
    policy.train(training)
    loss_sum = 0.0
    correct = 0
    samples = 0
    observation_shape = tuple(getattr(model.observation_space, "shape", ()))
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for observations, masks, actions in batches:
            model_observations = _model_input(observations, observation_shape)
            obs_tensor = torch.as_tensor(model_observations, dtype=torch.float32, device=device)
            action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
            distribution = policy.get_distribution(obs_tensor, action_masks=masks)
            negative_log_likelihood = -distribution.log_prob(action_tensor).mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                negative_log_likelihood.backward()
                torch.nn.utils.clip_grad_norm_(actor_parameters, max_grad_norm)
                optimizer.step()
            predictions = distribution.distribution.probs.argmax(dim=1)
            count = len(actions)
            loss_sum += float(negative_log_likelihood.detach().cpu()) * count
            correct += int((predictions == action_tensor).sum().detach().cpu())
            samples += count
    if samples == 0:
        raise ValueError("dataset split produced zero samples")
    return loss_sum / samples, correct / samples, samples


def _architecture(model: MaskableModel) -> dict[str, Any]:
    action_count = getattr(model.action_space, "n", None)
    policy = model.policy
    return {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "policy_type": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "observation_shape": list(getattr(model.observation_space, "shape", ())),
        "action_count": int(action_count) if action_count is not None else None,
        "public_actor_size": getattr(policy, "public_observation_size", None),
        "net_arch": _jsonable(getattr(policy, "net_arch", None)),
        "activation": (
            f"{policy.activation_fn.__module__}.{policy.activation_fn.__qualname__}"
            if getattr(policy, "activation_fn", None) is not None
            else None
        ),
        "parameter_shapes": {
            name: list(parameter.shape) for name, parameter in policy.named_parameters()
        },
    }


def _training_identity(
    config: BehaviorCloneConfig,
    dataset: DatasetSummary,
    model: MaskableModel,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_identity = dataset_manifest.get("identity", {})
    return {
        "algorithm": "actor_only_masked_cross_entropy_v1",
        "ctde_private_suffix": "zero_padded",
        "dataset_id": dataset.dataset_id,
        "dataset_profile_name": dataset_identity.get("profile_name"),
        "dataset_environment": dataset_identity.get("environment"),
        "dataset_expert": dataset_identity.get("expert"),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "validation_fraction": config.validation_fraction,
        "max_grad_norm": config.max_grad_norm,
        "seed": config.seed,
        "architecture": _architecture(model),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _load_imitation_manifest(directory: Path) -> dict[str, Any]:
    path = directory / IMITATION_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing {IMITATION_MANIFEST_FILENAME} in {directory}")
    payload = json.loads(path.read_text())
    if payload.get("format_version") != IMITATION_FORMAT_VERSION:
        raise ValueError("unsupported imitation training manifest format")
    return payload


def train_behavior_clone(
    model: MaskableModel,
    config: BehaviorCloneConfig,
) -> BehaviorCloneReport:
    """Warm-start a MaskablePPO/CTDE actor with masked cross-entropy."""

    import torch

    config.validate()
    dataset = validate_dataset(config.dataset_dir)
    dataset_manifest = load_dataset_manifest(config.dataset_dir)
    if getattr(model.action_space, "n", None) != ACTION_COUNT:
        raise ValueError(f"model action space must contain {ACTION_COUNT} actions")
    observation_shape = tuple(getattr(model.observation_space, "shape", ()))
    if observation_shape not in ((OBS_SIZE,), (CTDE_OBS_SIZE,)):
        raise ValueError("model does not use the canonical public or CTDE observation shape")
    if observation_shape == (CTDE_OBS_SIZE,) and getattr(
        model.policy, "public_observation_size", None
    ) != OBS_SIZE:
        raise ValueError("CTDE actor does not declare the canonical public prefix boundary")

    output = config.output_dir
    manifest_path = output / IMITATION_MANIFEST_FILENAME
    checkpoint_path = output / IMITATION_CHECKPOINT_FILENAME
    best_policy_path = output / IMITATION_BEST_POLICY_FILENAME
    identity = _training_identity(config, dataset, model, dataset_manifest)
    if manifest_path.exists():
        if not config.resume:
            raise FileExistsError(
                f"{manifest_path} already exists; pass resume=True to continue training"
            )
        manifest = _load_imitation_manifest(output)
        if manifest.get("identity") != identity:
            raise ValueError("imitation resume dataset, architecture, or hyperparameters differ")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing resume checkpoint {checkpoint_path}")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": IMITATION_FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "status": "initialized",
            "target_epochs": config.epochs,
            "epochs_completed": 0,
            "identity": identity,
            "dataset": dataclasses.asdict(dataset),
            "history": [],
            "git": git_metadata(Path(__file__).resolve().parents[1]),
            "runtime": runtime_metadata(),
        }
        _atomic_json(manifest_path, manifest)

    device = _resolve_device(model, config.device)
    model.policy.to(device)
    model.device = device
    actor_parameters = _actor_parameters(model.policy)
    optimizer = torch.optim.AdamW(
        actor_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    start_epoch = int(manifest["epochs_completed"])
    if config.epochs < start_epoch:
        raise ValueError(
            f"target epochs {config.epochs} is below completed epochs {start_epoch}"
        )
    if config.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("identity") != identity:
            raise ValueError("imitation checkpoint identity does not match manifest")
        model.policy.load_state_dict(checkpoint["policy_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if int(checkpoint["epochs_completed"]) != start_epoch:
            raise ValueError("checkpoint and manifest disagree about completed epochs")

    original_requires_grad = {
        id(parameter): parameter.requires_grad for parameter in model.policy.parameters()
    }
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    for parameter in actor_parameters:
        parameter.requires_grad_(True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    manifest["status"] = "training"
    manifest.pop("error", None)
    manifest["target_epochs"] = config.epochs
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_json(manifest_path, manifest)

    try:
        for epoch in range(start_epoch, config.epochs):
            torch.manual_seed(config.seed + epoch)
            train_metrics = _run_batches(
                model,
                _batch_iterator(
                    config.dataset_dir,
                    dataset_manifest,
                    batch_size=config.batch_size,
                    validation_fraction=config.validation_fraction,
                    seed=config.seed,
                    epoch=epoch,
                    validation=False,
                ),
                device=device,
                optimizer=optimizer,
                max_grad_norm=config.max_grad_norm,
                actor_parameters=actor_parameters,
            )
            validation_metrics = _run_batches(
                model,
                _batch_iterator(
                    config.dataset_dir,
                    dataset_manifest,
                    batch_size=config.batch_size,
                    validation_fraction=config.validation_fraction,
                    seed=config.seed,
                    epoch=epoch,
                    validation=True,
                ),
                device=device,
                optimizer=None,
                max_grad_norm=config.max_grad_norm,
                actor_parameters=actor_parameters,
            )
            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics[0],
                "train_accuracy": train_metrics[1],
                "train_samples": train_metrics[2],
                "validation_loss": validation_metrics[0],
                "validation_accuracy": validation_metrics[1],
                "validation_samples": validation_metrics[2],
            }
            manifest["history"].append(epoch_record)
            manifest["epochs_completed"] = epoch + 1
            if float(validation_metrics[0]) < float(
                manifest.get("best_validation_loss", float("inf"))
            ):
                manifest["best_validation_loss"] = float(validation_metrics[0])
                manifest["best_epoch"] = epoch + 1
                _atomic_torch_save(
                    best_policy_path,
                    {
                        "format_version": IMITATION_FORMAT_VERSION,
                        "identity": identity,
                        "epoch": epoch + 1,
                        "policy_state": model.policy.state_dict(),
                    },
                )
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            _atomic_torch_save(
                checkpoint_path,
                {
                    "format_version": IMITATION_FORMAT_VERSION,
                    "identity": identity,
                    "epochs_completed": epoch + 1,
                    "policy_state": model.policy.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
            )
            _atomic_json(manifest_path, manifest)

        if not best_policy_path.is_file():
            raise RuntimeError("behavior cloning produced no best-policy checkpoint")
        best = torch.load(best_policy_path, map_location=device, weights_only=True)
        if best.get("identity") != identity or int(best.get("epoch", -1)) != int(
            manifest["best_epoch"]
        ):
            raise ValueError("best-policy checkpoint does not match the training manifest")
        model.policy.load_state_dict(best["policy_state"])

        model_path = output / IMITATION_MODEL_FILENAME
        archive_path = model_path.with_suffix(".zip")
        temporary_archive = output / f".{IMITATION_MODEL_FILENAME}.tmp.zip"
        model.save(temporary_archive)
        temporary_archive.replace(archive_path)
        manifest["status"] = "complete"
        manifest["model_path"] = archive_path.name
        manifest["model_sha256"] = _sha256(archive_path)
        manifest["selected_epoch"] = int(manifest["best_epoch"])
        manifest["target_epochs"] = config.epochs
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["updated_at"] = datetime.now(UTC).isoformat()
        _atomic_json(manifest_path, manifest)
        raise
    finally:
        for parameter in model.policy.parameters():
            parameter.requires_grad_(original_requires_grad[id(parameter)])

    final = manifest["history"][int(manifest["best_epoch"]) - 1]
    return BehaviorCloneReport(
        model_path=archive_path,
        epochs_completed=int(manifest["epochs_completed"]),
        samples=dataset.samples,
        final_train_loss=float(final["train_loss"]),
        final_train_accuracy=float(final["train_accuracy"]),
        final_validation_loss=float(final["validation_loss"]),
        final_validation_accuracy=float(final["validation_accuracy"]),
    )


def initialize_behavior_clone_model(
    dataset_dir: str | Path,
    *,
    ctde: bool = True,
    net_arch: Sequence[int] = (512, 256),
    seed: int = 0,
    device: str = "auto",
):
    """Construct an untrained MaskablePPO model compatible with a dataset."""

    import torch
    from sb3_contrib import MaskablePPO

    from rl.ctde_policy import CTDEMaskableActorCriticPolicy
    from rl.single_agent_env import JassTeamEnv

    if not net_arch or any(width <= 0 for width in net_arch):
        raise ValueError("net_arch must contain positive widths")
    manifest = load_dataset_manifest(dataset_dir)
    environment = manifest["identity"]["environment"]
    profile = RulesetConfig.from_dict(environment["profile"])
    env = JassTeamEnv(
        seed=seed,
        profile=profile,
        enable_bidding=bool(environment["enable_bidding"]),
        enable_weis=bool(environment["enable_weis"]),
        enable_stock=bool(environment["enable_stock"]),
        privileged_critic=ctde,
    )
    policy: str | type[Any] = CTDEMaskableActorCriticPolicy if ctde else "MlpPolicy"
    return MaskablePPO(
        policy,
        env,
        seed=seed,
        device=device,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={
            "net_arch": {"pi": list(net_arch), "vf": list(net_arch)},
            "activation_fn": torch.nn.Tanh,
        },
        verbose=0,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and train Jass behavior cloning")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect compressed expert trajectories")
    collect.add_argument("output", type=Path)
    collect.add_argument("--games", type=int, required=True)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--profile", choices=("full", VERARDO_PROFILE_VERSION), default="full")
    collect.add_argument(
        "--expert",
        choices=("strategic", "search", "verardo"),
        default="strategic",
    )
    collect.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    collect.add_argument("--search-determinizations", type=int, default=8)
    collect.add_argument("--search-max-rollouts", type=int, default=72)
    collect.add_argument("--search-time-budget-ms", type=float)
    collect.add_argument("--resume", action="store_true")

    validate = commands.add_parser("validate", help="fully validate a collected dataset")
    validate.add_argument("dataset", type=Path)

    train = commands.add_parser("train", help="initialize and behavior-clone an actor")
    train.add_argument("dataset", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--validation-fraction", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument("--net-arch", type=int, nargs="+", default=[512, 256])
    train.add_argument("--no-ctde", action="store_true")
    train.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "collect":
        if args.expert == "verardo":
            if args.profile != VERARDO_PROFILE_VERSION:
                raise SystemExit("the Verardo expert requires --profile verardo-v1")
            policy: PublicPolicy = VerardoReferencePolicy()
        elif args.expert == "search":
            profile = (
                VERARDO_V1_PROFILE
                if args.profile == VERARDO_PROFILE_VERSION
                else STANDARD_RULES_PROFILE
            )
            search: PublicPolicy = PIMCSearchPolicy(
                seed=args.seed,
                profile=profile,
                config=PIMCConfig(
                    determinizations=args.search_determinizations,
                    max_rollouts=args.search_max_rollouts,
                    time_budget_ms=args.search_time_budget_ms,
                ),
            )
            policy = (
                TrumpOnlyPolicy(search)
                if args.profile == VERARDO_PROFILE_VERSION
                else search
            )
        else:
            strategic: PublicPolicy = StrategicHeuristicPolicy()
            policy = (
                TrumpOnlyPolicy(strategic)
                if args.profile == VERARDO_PROFILE_VERSION
                else strategic
            )
        result = collect_expert_dataset(
            policy,
            CollectionConfig(
                output_dir=args.output,
                games=args.games,
                seed=args.seed,
                profile_name=args.profile,
                shard_size=args.shard_size,
                resume=args.resume,
                expert_id=args.expert,
            ),
        )
    elif args.command == "validate":
        result = validate_dataset(args.dataset)
    else:
        model = initialize_behavior_clone_model(
            args.dataset,
            ctde=not args.no_ctde,
            net_arch=args.net_arch,
            seed=args.seed,
            device=args.device,
        )
        result = train_behavior_clone(
            model,
            BehaviorCloneConfig(
                dataset_dir=args.dataset,
                output_dir=args.output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                device=args.device,
                resume=args.resume,
            ),
        )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


__all__ = [
    "BehaviorCloneConfig",
    "BehaviorCloneReport",
    "CollectionConfig",
    "DatasetSummary",
    "collect_expert_dataset",
    "initialize_behavior_clone_model",
    "load_dataset_manifest",
    "main",
    "train_behavior_clone",
    "validate_dataset",
]


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
