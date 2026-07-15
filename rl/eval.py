"""Manifest-aware, paired tournament evaluation for MaskablePPO models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS
from core.ruleset import (
    ALL_CONTRACTS_X1_PROFILE,
    LEGACY_UNMULTIPLIED_PROFILE,
    STANDARD_RULES_PROFILE,
    RulesetConfig,
)
from env.jass_aec_env import ACTION_COUNT, OBS_SIZE, OBSERVATION_SCHEMA_VERSION
from rl.baselines import PublicPolicy, SeededRandomPolicy, StrategicHeuristicPolicy
from rl.ctde_policy import CTDEMaskableActorCriticPolicy
from rl.privileged_observation import (
    CTDE_OBS_SIZE,
    PRIVILEGED_OBSERVATION_SCHEMA_NAME,
    PRIVILEGED_OBSERVATION_SCHEMA_VERSION,
)
from rl.reference_verardo import (
    TrumpOnlyPolicy,
    VerardoReferencePolicy,
    verardo_tournament_config,
)
from rl.run_manifest import (
    MANIFEST_FILENAME,
    assert_compatible,
    git_metadata,
    load_manifest,
    runtime_metadata,
    write_manifest,
)
from rl.tournament import (
    TournamentConfig,
    TournamentReport,
    maskable_model_policy,
    run_tournament,
)

EVALUATION_REPORT_VERSION = 2
DEFAULT_REPORT_FILENAME = "evaluation_report.json"
OPPONENTS = ("random", "strategic", "verardo-random", "verardo")
VALID_MODES = (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE)

PROFILE_BY_NAME = {
    "standard": STANDARD_RULES_PROFILE,
    "all-x1": ALL_CONTRACTS_X1_PROFILE,
    "legacy": LEGACY_UNMULTIPLIED_PROFILE,
}

# The gates are documented in docs/benchmark_goal.md.  A report keeps every
# component check, rather than reducing an unsuccessful run to one opaque bool.
QUALIFICATION_GATES: Mapping[str, Mapping[str, float | int | None]] = {
    "random": {
        "minimum_games": 4_000,
        "minimum_pairs": 2_000,
        "minimum_win_rate": 0.60,
        "minimum_wilson_low": 0.50,
        "minimum_paired_win_rate": 0.60,
        "minimum_paired_wilson_low": 0.50,
        "minimum_average_point_difference": None,
    },
    "strategic": {
        "minimum_games": 4_000,
        "minimum_pairs": 2_000,
        "minimum_win_rate": 0.55,
        "minimum_wilson_low": 0.50,
        "minimum_paired_win_rate": 0.55,
        "minimum_paired_wilson_low": 0.50,
        "minimum_average_point_difference": None,
    },
    "verardo": {
        "minimum_games": 4_000,
        "minimum_pairs": 2_000,
        "minimum_win_rate": 0.67,
        "minimum_wilson_low": 0.634,
        "minimum_paired_win_rate": 0.67,
        "minimum_paired_wilson_low": 0.634,
        "minimum_average_point_difference": 29.1,
    },
    "verardo-random": {
        "minimum_games": 4_000,
        "minimum_pairs": 2_000,
        "minimum_win_rate": 0.87,
        "minimum_wilson_low": 0.841,
        "minimum_paired_win_rate": 0.87,
        "minimum_paired_wilson_low": 0.841,
        "minimum_average_point_difference": 67.1,
    },
}


@dataclass(frozen=True)
class EvaluationEnvironment:
    """Gameplay configuration reconstructed from a training manifest."""

    control_team: bool
    enable_bidding: bool
    enable_weis: bool
    enable_stock: bool
    mode: str | None
    trump_suit: str | None
    randomize_starter: bool
    reward_scale: float
    terminal_win_bonus: float
    profile: RulesetConfig
    ctde: bool = False
    actor_observation_shape: tuple[int, ...] = (OBS_SIZE,)
    training_observation_shape: tuple[int, ...] = (OBS_SIZE,)
    privileged_observation_schema: tuple[str, int] | None = None

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> EvaluationEnvironment:
        profile_payload = payload.get("profile")
        if not isinstance(profile_payload, Mapping):
            raise ValueError("manifest environment.profile must be a mapping")
        profile = RulesetConfig.from_dict(profile_payload)

        mode = payload.get("mode")
        if mode is not None and mode not in VALID_MODES:
            raise ValueError(f"manifest environment.mode must be one of {VALID_MODES} or null")
        trump_suit = payload.get("trump_suit")
        if trump_suit is not None and trump_suit not in SUITS:
            raise ValueError("manifest environment.trump_suit must be a Swiss suit or null")
        if mode is not None:
            profile.contract_factor(mode, trump_suit)

        ctde = payload.get("ctde", False)
        if not isinstance(ctde, bool):
            raise ValueError("manifest environment.ctde must be a boolean")
        actor_shape = _required_shape(
            payload.get("actor_observation_shape", [OBS_SIZE]),
            "actor_observation_shape",
        )
        training_shape = _required_shape(
            payload.get(
                "training_observation_shape",
                [CTDE_OBS_SIZE if ctde else OBS_SIZE],
            ),
            "training_observation_shape",
        )
        if actor_shape != (OBS_SIZE,):
            raise ValueError("manifest actor observation must use the canonical public shape")
        expected_training_shape = (CTDE_OBS_SIZE,) if ctde else (OBS_SIZE,)
        if training_shape != expected_training_shape:
            raise ValueError(
                f"manifest CTDE/training observation mismatch: {training_shape} != "
                f"{expected_training_shape}"
            )
        privileged_payload = payload.get("privileged_observation_schema")
        privileged_schema: tuple[str, int] | None = None
        if ctde:
            if not isinstance(privileged_payload, Mapping):
                raise ValueError("CTDE manifest requires privileged observation schema metadata")
            name = privileged_payload.get("name")
            version = privileged_payload.get("version")
            if (
                name != PRIVILEGED_OBSERVATION_SCHEMA_NAME
                or version != PRIVILEGED_OBSERVATION_SCHEMA_VERSION
            ):
                raise ValueError("unsupported privileged observation schema")
            privileged_schema = (name, version)
        elif privileged_payload is not None:
            raise ValueError("non-CTDE manifest cannot declare a privileged observation schema")

        environment = cls(
            control_team=_required_bool(payload, "control_team"),
            enable_bidding=_required_bool(payload, "enable_bidding"),
            enable_weis=_required_bool(payload, "enable_weis"),
            enable_stock=_required_bool(payload, "enable_stock"),
            mode=mode,
            trump_suit=trump_suit,
            randomize_starter=_required_bool(payload, "randomize_starter"),
            reward_scale=_required_float(payload, "reward_scale", positive=True),
            terminal_win_bonus=_required_float(
                payload,
                "terminal_win_bonus",
                non_negative=True,
            ),
            profile=profile,
            ctde=ctde,
            actor_observation_shape=actor_shape,
            training_observation_shape=training_shape,
            privileged_observation_schema=privileged_schema,
        )
        return environment

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "control_team": self.control_team,
            "enable_bidding": self.enable_bidding,
            "enable_weis": self.enable_weis,
            "enable_stock": self.enable_stock,
            "mode": self.mode,
            "trump_suit": self.trump_suit,
            "randomize_starter": self.randomize_starter,
            "reward_scale": self.reward_scale,
            "terminal_win_bonus": self.terminal_win_bonus,
            "profile": self.profile.to_dict(),
            "ctde": self.ctde,
            "actor_observation_shape": list(self.actor_observation_shape),
            "training_observation_shape": list(self.training_observation_shape),
            "privileged_observation_schema": (
                {
                    "name": self.privileged_observation_schema[0],
                    "version": self.privileged_observation_schema[1],
                }
                if self.privileged_observation_schema is not None
                else None
            ),
        }

    def tournament_config(self, *, episodes: int, seed: int) -> TournamentConfig:
        if self.enable_bidding:
            modes: tuple[str | None, ...] = (None,)
        elif self.mode is not None:
            modes = (self.mode,)
        else:
            modes = VALID_MODES

        trump_suits = (self.trump_suit,) if self.trump_suit is not None else SUITS
        return TournamentConfig(
            episodes=episodes,
            seed=seed,
            modes=modes,
            trump_suits=trump_suits,
            starters=(0, 1, 2, 3),
            swap_teams=True,
            enable_bidding=self.enable_bidding,
            enable_weis=self.enable_weis,
            enable_stock=self.enable_stock,
            profile=self.profile,
        )


@dataclass(frozen=True)
class EvalConfig:
    model_path: Path
    episodes: int = 4_000
    seed: int = 0
    opponents: tuple[str, ...] = OPPONENTS
    output_path: Path | None = None
    include_results: bool = False
    allow_legacy: bool = False
    device: str = "auto"
    # Explicit fallback settings, used only after --allow-legacy is supplied.
    legacy_enable_bidding: bool = True
    legacy_enable_weis: bool = True
    legacy_enable_stock: bool = True
    legacy_control_team: bool = False
    legacy_mode: str | None = None
    legacy_trump_suit: str | None = None
    legacy_profile_name: str = "standard"

    def validate(self) -> None:
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int):
            raise ValueError("episodes must be an integer")
        if self.episodes <= 0 or self.episodes % 8:
            raise ValueError(
                "episodes must be a positive multiple of 8 for paired four-starter rotation"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not self.opponents:
            raise ValueError("at least one opponent is required")
        if len(set(self.opponents)) != len(self.opponents):
            raise ValueError("opponents must not contain duplicates")
        unknown = sorted(set(self.opponents) - set(OPPONENTS))
        if unknown:
            raise ValueError(f"unknown opponents: {unknown}")
        if self.legacy_profile_name not in PROFILE_BY_NAME:
            raise ValueError(f"unknown legacy profile: {self.legacy_profile_name}")
        if self.output_path is not None and self.output_path.resolve() == self.model_path.resolve():
            raise ValueError("output_path must not overwrite the model")


@dataclass(frozen=True)
class EvaluationContext:
    environment: EvaluationEnvironment
    manifest: Mapping[str, Any] | None
    manifest_path: Path | None
    legacy_reason: str | None
    artifact_hash_validated: bool

    @property
    def is_legacy(self) -> bool:
        return self.manifest is None


ModelLoader = Callable[[Path, str], object]
TournamentRunner = Callable[
    [PublicPolicy, PublicPolicy, TournamentConfig],
    TournamentReport,
]
CandidateBuilder = Callable[[object, EvaluationContext], PublicPolicy]


def evaluate(
    config: EvalConfig,
    *,
    model_loader: ModelLoader | None = None,
    tournament_runner: TournamentRunner | None = None,
    candidate_builder: CandidateBuilder | None = None,
) -> dict[str, Any]:
    """Evaluate a model and atomically persist its reproducible JSON report."""

    config.validate()
    model_path = _resolve_model_path(config.model_path)
    output_path = config.output_path or model_path.parent / DEFAULT_REPORT_FILENAME
    if output_path.resolve() == model_path:
        raise ValueError("output_path must not overwrite the model")
    if output_path.resolve() == _manifest_path(model_path).resolve():
        raise ValueError("output_path must not overwrite the run manifest")
    context = _load_context(model_path, config)

    loader = model_loader or _load_maskable_model
    model = loader(model_path, config.device)
    actor_privacy_validated = _assert_model_spaces(
        model,
        context.environment.training_observation_shape,
        ctde=context.environment.ctde,
    )
    candidate = (
        candidate_builder(model, context)
        if candidate_builder is not None
        else maskable_model_policy(model)
    )
    if not callable(candidate):
        raise TypeError("candidate_builder must return a callable public policy")
    runner = tournament_runner or run_tournament

    opponent_reports: dict[str, dict[str, Any]] = {}
    for opponent in config.opponents:
        report, tournament = _run_opponent(
            opponent,
            candidate,
            context.environment,
            config,
            runner,
        )
        opponent_reports[opponent] = _opponent_payload(
            opponent,
            report,
            tournament,
            include_results=config.include_results,
            provenance_eligible=(
                not context.is_legacy
                and context.artifact_hash_validated
                and actor_privacy_validated
            ),
            protocol_eligible=(
                _is_full_project_environment(context.environment)
                if opponent in {"random", "strategic"}
                else True
            ),
        )

    report_payload: dict[str, Any] = {
        "report_version": EVALUATION_REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": _model_identity(model_path, model),
        "run": _run_identity(model_path, context),
        "evaluation_provenance": {
            "git": git_metadata(Path(__file__).resolve().parents[1]),
            "runtime": runtime_metadata(),
        },
        "compatibility": {
            "manifest_required": not config.allow_legacy,
            "manifest_validated": not context.is_legacy,
            "artifact_hash_validated": context.artifact_hash_validated,
            "actor_privacy_validated": actor_privacy_validated,
            "legacy_override": context.is_legacy,
            "legacy_reason": context.legacy_reason,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "actor_observation_shape": [OBS_SIZE],
            "model_observation_shape": list(context.environment.training_observation_shape),
            "action_count": ACTION_COUNT,
            "device": config.device,
        },
        "evaluation": {
            "seed": config.seed,
            "games_per_opponent": config.episodes,
            "opponents": list(config.opponents),
            "paired_same_deal": True,
            "team_swap": True,
            "starters": [0, 1, 2, 3],
            "manifest_environment": context.environment.to_manifest_dict(),
            "full_project_environment_eligible": _is_full_project_environment(context.environment),
        },
        "opponents": opponent_reports,
        "qualification": _overall_qualification(opponent_reports),
        "report_path": str(output_path.resolve()),
    }
    # Normalize tuples emitted by dataclass/asdict paths so the in-memory value
    # is byte-for-byte representable by the persisted JSON contract.
    json_report = json.loads(json.dumps(report_payload))
    write_manifest(output_path, json_report)
    return json_report


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"manifest environment.{key} must be a boolean")
    return value


def _required_float(
    payload: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"manifest environment.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"manifest environment.{key} must be finite")
    if positive and result <= 0:
        raise ValueError(f"manifest environment.{key} must be positive")
    if non_negative and result < 0:
        raise ValueError(f"manifest environment.{key} must be non-negative")
    return result


def _required_shape(value: Any, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"manifest environment.{key} must be a sequence")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"manifest environment.{key} must contain positive integers")
    return tuple(value)


def _is_full_project_environment(environment: EvaluationEnvironment) -> bool:
    return bool(
        environment.control_team
        and environment.enable_bidding
        and environment.enable_weis
        and environment.enable_stock
        and environment.mode is None
        and environment.trump_suit is None
        and environment.randomize_starter
        and environment.profile.to_dict() == STANDARD_RULES_PROFILE.to_dict()
    )


def _resolve_model_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.exists() and candidate.suffix != ".zip":
        archive = candidate.with_suffix(".zip")
        if archive.exists():
            candidate = archive
    if not candidate.is_file():
        raise FileNotFoundError(f"model does not exist: {path}")
    return candidate.resolve()


def _manifest_path(model_path: Path) -> Path:
    return model_path.parent / MANIFEST_FILENAME


def _load_context(model_path: Path, config: EvalConfig) -> EvaluationContext:
    try:
        manifest = load_manifest(model_path)
        environment_payload = manifest.get("environment")
        if not isinstance(environment_payload, Mapping):
            raise ValueError("run manifest environment must be a mapping")
        environment = EvaluationEnvironment.from_manifest(environment_payload)
        assert_compatible(
            manifest,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION,
            observation_shape=environment.training_observation_shape,
            action_count=ACTION_COUNT,
            environment=environment.to_manifest_dict(),
        )
        artifact_hash_validated = _assert_model_belongs_to_run(model_path, manifest)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        if not config.allow_legacy:
            raise
        return EvaluationContext(
            environment=_legacy_environment(config),
            manifest=None,
            manifest_path=None,
            legacy_reason=f"{type(exc).__name__}: {exc}",
            artifact_hash_validated=False,
        )
    return EvaluationContext(
        environment=environment,
        manifest=manifest,
        manifest_path=_manifest_path(model_path),
        legacy_reason=None,
        artifact_hash_validated=artifact_hash_validated,
    )


def _legacy_environment(config: EvalConfig) -> EvaluationEnvironment:
    mode = config.legacy_mode
    if mode is not None and mode not in VALID_MODES:
        raise ValueError(f"legacy mode must be one of {VALID_MODES} or omitted")
    trump_suit = config.legacy_trump_suit
    if trump_suit is not None and trump_suit not in SUITS:
        raise ValueError("legacy trump_suit must be a Swiss suit or omitted")
    profile = PROFILE_BY_NAME[config.legacy_profile_name]
    if mode is not None:
        profile.contract_factor(mode, trump_suit)
    return EvaluationEnvironment(
        control_team=config.legacy_control_team,
        enable_bidding=config.legacy_enable_bidding,
        enable_weis=config.legacy_enable_weis,
        enable_stock=config.legacy_enable_stock,
        mode=mode,
        trump_suit=trump_suit,
        randomize_starter=False,
        reward_scale=1.0,
        terminal_win_bonus=0.0,
        profile=profile,
        ctde=False,
        actor_observation_shape=(OBS_SIZE,),
        training_observation_shape=(OBS_SIZE,),
        privileged_observation_schema=None,
    )


def _assert_model_belongs_to_run(
    model_path: Path,
    manifest: Mapping[str, Any],
) -> bool:
    known_models: set[str] = set()
    final_model = manifest.get("final_model")
    if isinstance(final_model, str):
        known_models.add(final_model)
    checkpoints = manifest.get("checkpoints", [])
    if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)):
        known_models.update(item for item in checkpoints if isinstance(item, str))
    if not known_models:
        raise ValueError("run manifest does not identify any trained model checkpoints")
    if model_path.name not in known_models:
        raise ValueError(f"model {model_path.name!r} is not listed in the adjacent run manifest")
    expected_hash: str | None = None
    if model_path.name == final_model:
        recorded = manifest.get("final_model_sha256")
        expected_hash = recorded if isinstance(recorded, str) else None
    hashes = manifest.get("checkpoint_sha256")
    if expected_hash is None and isinstance(hashes, Mapping):
        recorded = hashes.get(model_path.name)
        expected_hash = recorded if isinstance(recorded, str) else None
    if expected_hash is None:
        return False
    actual_hash = _sha256(model_path)
    if actual_hash != expected_hash:
        raise ValueError(f"model SHA-256 {actual_hash} does not match manifest {expected_hash}")
    return True


def _load_maskable_model(model_path: Path, device: str) -> object:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - optional CLI dependency
        raise ImportError("sb3-contrib is required for evaluation") from exc
    return MaskablePPO.load(model_path, device=device)


def _assert_model_spaces(
    model: object,
    expected_shape: tuple[int, ...],
    *,
    ctde: bool,
) -> bool:
    observation_shape = tuple(getattr(getattr(model, "observation_space", None), "shape", ()))
    if observation_shape != expected_shape:
        raise ValueError(
            f"model observation shape {observation_shape} is incompatible with "
            f"manifested training shape {expected_shape}"
        )
    action_count = getattr(getattr(model, "action_space", None), "n", None)
    if action_count != ACTION_COUNT:
        raise ValueError(
            f"model action count {action_count!r} is incompatible with canonical "
            f"action count {ACTION_COUNT}"
        )
    if not ctde:
        return observation_shape == (OBS_SIZE,)

    policy = getattr(model, "policy", None)
    if not isinstance(policy, CTDEMaskableActorCriticPolicy):
        raise ValueError("CTDE manifest requires the audited public-actor policy class")
    if getattr(policy, "public_observation_size", None) != OBS_SIZE:
        raise ValueError("CTDE actor public prefix does not match the canonical schema")

    import numpy as np
    import torch

    first = np.zeros((1, CTDE_OBS_SIZE), dtype=np.float32)
    second = first.copy()
    second[:, OBS_SIZE:] = 1.0
    with torch.no_grad():
        first_tensor = policy.obs_to_tensor(first)[0]
        second_tensor = policy.obs_to_tensor(second)[0]
        first_logits = policy.get_distribution(first_tensor).distribution.logits
        second_logits = policy.get_distribution(second_tensor).distribution.logits
    if not torch.equal(first_logits, second_logits):
        raise ValueError("CTDE actor output changes when only private critic inputs change")
    return True


def _run_opponent(
    opponent: str,
    candidate: PublicPolicy,
    environment: EvaluationEnvironment,
    config: EvalConfig,
    runner: TournamentRunner,
) -> tuple[TournamentReport, TournamentConfig]:
    if opponent == "random":
        reference: PublicPolicy = SeededRandomPolicy(config.seed)
        tournament = environment.tournament_config(
            episodes=config.episodes,
            seed=config.seed,
        )
        evaluated_candidate = candidate
    elif opponent == "strategic":
        reference = StrategicHeuristicPolicy()
        tournament = environment.tournament_config(
            episodes=config.episodes,
            seed=config.seed,
        )
        evaluated_candidate = candidate
    elif opponent == "verardo-random":
        reference = TrumpOnlyPolicy(SeededRandomPolicy(config.seed))
        tournament = verardo_tournament_config(
            episodes=config.episodes,
            seed=config.seed,
        )
        evaluated_candidate = TrumpOnlyPolicy(candidate)
    elif opponent == "verardo":
        reference = TrumpOnlyPolicy(VerardoReferencePolicy())
        tournament = verardo_tournament_config(
            episodes=config.episodes,
            seed=config.seed,
        )
        evaluated_candidate = TrumpOnlyPolicy(candidate)
    else:  # pragma: no cover - EvalConfig validation guards this
        raise ValueError(f"unknown opponent: {opponent}")
    tournament.validate()
    return runner(evaluated_candidate, reference, tournament), tournament


def _opponent_payload(
    opponent: str,
    report: TournamentReport,
    tournament: TournamentConfig,
    *,
    include_results: bool,
    provenance_eligible: bool,
    protocol_eligible: bool,
) -> dict[str, Any]:
    overall = report.overall
    return {
        "protocol": (
            "verardo-v1-matched" if opponent in {"verardo-random", "verardo"} else "manifest-exact"
        ),
        "environment": _tournament_environment(tournament),
        "samples": {
            "games": overall.episodes,
            "complete_pairs": overall.pairs,
            "starter_rotations": overall.episodes // 8,
        },
        "metrics": report.to_dict(include_results=include_results),
        "qualification": _qualification(
            opponent,
            report,
            provenance_eligible=provenance_eligible,
            protocol_eligible=protocol_eligible,
        ),
    }


def _tournament_environment(config: TournamentConfig) -> dict[str, Any]:
    return {
        "enable_bidding": config.enable_bidding,
        "enable_weis": config.enable_weis,
        "enable_stock": config.enable_stock,
        "modes": list(config.modes),
        "trump_suits": list(config.trump_suits),
        "starters": list(config.starters),
        "swap_teams": config.swap_teams,
        "profile": (config.profile or STANDARD_RULES_PROFILE).to_dict(),
    }


def _qualification(
    opponent: str,
    report: TournamentReport,
    *,
    provenance_eligible: bool,
    protocol_eligible: bool = True,
) -> dict[str, Any]:
    gate = QUALIFICATION_GATES[opponent]
    overall = report.overall
    minimum_difference = gate["minimum_average_point_difference"]
    checks = {
        "manifest_provenance": provenance_eligible,
        "protocol_environment": protocol_eligible,
        "minimum_games": overall.episodes >= int(gate["minimum_games"]),
        "minimum_pairs": overall.pairs >= int(gate["minimum_pairs"]),
        "complete_pairing": overall.pairs * 2 == overall.episodes,
        "minimum_win_rate": overall.win_rate >= float(gate["minimum_win_rate"]),
        "minimum_wilson_low": overall.win_rate_ci95.low > float(gate["minimum_wilson_low"]),
        "minimum_paired_win_rate": overall.paired_win_rate
        >= float(gate["minimum_paired_win_rate"]),
        "minimum_paired_wilson_low": overall.paired_win_rate_ci95.low
        > float(gate["minimum_paired_wilson_low"]),
        "minimum_average_point_difference": (
            True
            if minimum_difference is None
            else overall.average_point_difference >= float(minimum_difference)
        ),
    }
    performance_checks = {
        key: value
        for key, value in checks.items()
        if key not in {"manifest_provenance", "protocol_environment"}
    }
    return {
        "thresholds": dict(gate),
        "checks": checks,
        "performance_qualified": all(performance_checks.values()),
        "qualified": all(checks.values()),
    }


def _overall_qualification(
    opponent_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool | None]:
    def _passed(name: str) -> bool | None:
        payload = opponent_reports.get(name)
        if payload is None:
            return None
        qualification = payload.get("qualification", {})
        return bool(qualification.get("qualified"))

    random_passed = _passed("random")
    strategic_passed = _passed("strategic")
    full_project = (
        random_passed and strategic_passed
        if random_passed is not None and strategic_passed is not None
        else None
    )
    requested = [_passed(name) for name in opponent_reports]
    verardo_random_passed = _passed("verardo-random")
    verardo_reference_passed = _passed("verardo")
    verardo_project = (
        verardo_random_passed and verardo_reference_passed
        if verardo_random_passed is not None and verardo_reference_passed is not None
        else None
    )
    return {
        "full_project": full_project,
        "verardo_random": verardo_random_passed,
        "verardo_reference": verardo_reference_passed,
        "verardo_project": verardo_project,
        "all_requested": all(value is True for value in requested),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_path: Path, model: object) -> dict[str, Any]:
    return {
        "path": str(model_path),
        "filename": model_path.name,
        "sha256": _sha256(model_path),
        "size_bytes": model_path.stat().st_size,
        "num_timesteps": getattr(model, "num_timesteps", None),
    }


def _run_identity(model_path: Path, context: EvaluationContext) -> dict[str, Any]:
    manifest = context.manifest or {}
    git = manifest.get("git") if isinstance(manifest.get("git"), Mapping) else {}
    return {
        "run_id": model_path.parent.name,
        "directory": str(model_path.parent),
        "manifest_path": (
            str(context.manifest_path) if context.manifest_path is not None else None
        ),
        "manifest_version": manifest.get("manifest_version"),
        "manifest_sha256": (
            _sha256(context.manifest_path) if context.manifest_path is not None else None
        ),
        "created_at": manifest.get("created_at"),
        "status": manifest.get("status"),
        "actual_timesteps": manifest.get("actual_timesteps"),
        "profile_version": context.environment.profile.version,
        "git_commit": git.get("commit"),
        "tracked_diff_sha256": git.get("tracked_diff_sha256"),
    }


def _parse_args(argv: Sequence[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="Run manifest-validated paired Jass tournaments")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--episodes", type=int, default=4_000, help="games per opponent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponents", nargs="+", choices=OPPONENTS, default=list(OPPONENTS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-results", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="explicitly evaluate without a compatible manifest (never formally qualifies)",
    )

    legacy = parser.add_argument_group("legacy fallback environment")
    legacy.add_argument("--no-bidding", action="store_true")
    legacy.add_argument("--no-weis", action="store_true")
    legacy.add_argument("--no-stock", action="store_true")
    legacy.add_argument("--control-team", action="store_true")
    legacy.add_argument("--mode", choices=VALID_MODES)
    legacy.add_argument("--trump-suit", choices=SUITS)
    legacy.add_argument("--profile", choices=tuple(PROFILE_BY_NAME), default="standard")
    args = parser.parse_args(argv)

    return EvalConfig(
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
        opponents=tuple(args.opponents),
        output_path=args.output,
        include_results=bool(args.include_results),
        allow_legacy=bool(args.allow_legacy),
        device=args.device,
        legacy_enable_bidding=not args.no_bidding,
        legacy_enable_weis=not args.no_weis,
        legacy_enable_stock=not args.no_stock,
        legacy_control_team=bool(args.control_team),
        legacy_mode=args.mode,
        legacy_trump_suit=args.trump_suit,
        legacy_profile_name=args.profile,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    try:
        report = evaluate(config)
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateBuilder",
    "DEFAULT_REPORT_FILENAME",
    "EVALUATION_REPORT_VERSION",
    "EvalConfig",
    "EvaluationEnvironment",
    "evaluate",
    "main",
]
