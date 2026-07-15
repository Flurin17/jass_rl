from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

import rl.eval as eval_module
from core.cards import MODE_TRUMP
from core.ruleset import RulesetConfig
from env.jass_aec_env import (
    ACTION_COUNT,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    BIDDING_TRUMP_ACTIONS,
    BIDDING_UNEUFE_ACTION,
    OBS_SIZE,
    OBSERVATION_SCHEMA_VERSION,
)
from rl.baselines import SeededRandomPolicy, StrategicHeuristicPolicy
from rl.ctde_policy import CTDEMaskableActorCriticPolicy
from rl.eval import EvalConfig, EvaluationEnvironment, evaluate
from rl.privileged_observation import (
    CTDE_OBS_SIZE,
    PRIVILEGED_OBSERVATION_SCHEMA_NAME,
    PRIVILEGED_OBSERVATION_SCHEMA_VERSION,
)
from rl.reference_verardo import (
    VERARDO_V1_PROFILE,
    TrumpOnlyPolicy,
    VerardoReferencePolicy,
)
from rl.run_manifest import MANIFEST_FILENAME, MANIFEST_VERSION, write_manifest
from rl.tournament import (
    AggregateMetrics,
    EpisodeResult,
    TournamentReport,
    WilsonInterval,
    build_schedule,
    run_tournament,
    summarize_results,
)


@dataclass
class _Space:
    shape: tuple[int, ...]


@dataclass
class _ActionSpace:
    n: int


class _FakeModel:
    observation_space = _Space((OBS_SIZE,))
    action_space = _ActionSpace(ACTION_COUNT)
    num_timesteps = 1234

    def predict(self, observation, *, action_masks, deterministic):
        del observation, deterministic
        return np.array([np.flatnonzero(action_masks)[0]]), None


class _FakeCTDEModel:
    def __init__(self) -> None:
        self.observation_space = spaces.Box(0.0, 1.0, (CTDE_OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.policy = CTDEMaskableActorCriticPolicy(
            self.observation_space,
            self.action_space,
            lambda _: 3e-4,
            net_arch=[16],
        )
        self.num_timesteps = 1234

    def predict(self, observation, *, action_masks, deterministic):
        assert np.asarray(observation).shape == (CTDE_OBS_SIZE,)
        assert not np.any(np.asarray(observation)[OBS_SIZE:])
        return self.policy.predict(
            observation,
            action_masks=action_masks,
            deterministic=deterministic,
        )


class _NoTrumpPreferringPolicy:
    def __init__(self, preferred_action: int) -> None:
        self.preferred_action = preferred_action
        self.bidding_masks: list[np.ndarray] = []

    def __call__(self, observation, action_mask, agent):
        del observation, agent
        mask = np.asarray(action_mask).reshape(-1)
        bidding_actions = (
            *BIDDING_TRUMP_ACTIONS,
            BIDDING_OBEABE_ACTION,
            BIDDING_UNEUFE_ACTION,
            BIDDING_PUSH_ACTION,
        )
        if any(mask[action] for action in bidding_actions):
            self.bidding_masks.append(mask.copy())
            if mask[self.preferred_action]:
                return self.preferred_action
        return int(np.flatnonzero(mask)[0])


def _model_loader(path: Path, device: str) -> _FakeModel:
    assert path.is_file()
    assert device in {"auto", "cpu"}
    return _FakeModel()


def _winning_runner(candidate, reference, config) -> TournamentReport:
    del candidate, reference
    results = []
    for spec in build_schedule(config):
        points = (120, 37) if spec.candidate_team == 0 else (37, 120)
        results.append(
            EpisodeResult(
                spec=spec,
                mode=spec.mode or MODE_TRUMP,
                trump_suit=spec.trump_suit or "rosen",
                team_points=points,
            )
        )
    return summarize_results(results)


def _environment(**overrides) -> EvaluationEnvironment:
    values = {
        "control_team": True,
        "enable_bidding": False,
        "enable_weis": True,
        "enable_stock": False,
        "mode": MODE_TRUMP,
        "trump_suit": "rosen",
        "randomize_starter": True,
        "reward_scale": 0.001,
        "terminal_win_bonus": 1.0,
        "profile": RulesetConfig(
            version="eval-test-v1",
            allow_stock=False,
            contract_factors={"rosen": 5},
        ),
    }
    values.update(overrides)
    return EvaluationEnvironment(**values)


def _write_run_manifest(
    model_path: Path,
    environment: EvaluationEnvironment,
    *,
    schema_version: int = OBSERVATION_SCHEMA_VERSION,
) -> None:
    write_manifest(
        model_path.parent / MANIFEST_FILENAME,
        {
            "manifest_version": MANIFEST_VERSION,
            "created_at": "2026-07-15T12:00:00+00:00",
            "status": "trained",
            "actual_timesteps": 1234,
            "final_model": model_path.name,
            "checkpoints": [],
            "environment": environment.to_manifest_dict(),
            "observation_schema_version": schema_version,
            "observation_shape": list(environment.training_observation_shape),
            "action_count": ACTION_COUNT,
            "git": {
                "commit": "abc123",
                "tracked_diff_sha256": "def456",
            },
        },
    )


def test_missing_manifest_requires_explicit_legacy_override(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy.zip"
    model_path.write_bytes(b"legacy-model")

    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        evaluate(
            EvalConfig(model_path=model_path, episodes=8, opponents=("random",)),
            model_loader=_model_loader,
            tournament_runner=_winning_runner,
        )

    output_path = tmp_path / "legacy-report.json"
    report = evaluate(
        EvalConfig(
            model_path=model_path,
            episodes=8,
            opponents=("random",),
            output_path=output_path,
            allow_legacy=True,
        ),
        model_loader=_model_loader,
        tournament_runner=_winning_runner,
    )

    assert report["compatibility"]["manifest_validated"] is False
    assert report["compatibility"]["legacy_override"] is True
    assert MANIFEST_FILENAME in report["compatibility"]["legacy_reason"]
    assert report["opponents"]["random"]["qualification"]["qualified"] is False
    assert report["opponents"]["random"]["qualification"]["checks"]["manifest_provenance"] is False
    assert json.loads(output_path.read_text()) == report
    assert not output_path.with_suffix(".json.tmp").exists()


def test_manifest_reconstructs_exact_rules_and_rejects_schema_mismatch(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model_final.zip"
    model_path.write_bytes(b"manifested-model")
    environment = _environment()
    _write_run_manifest(model_path, environment)

    seen_configs = []

    def recording_runner(candidate, reference, config):
        del candidate, reference
        seen_configs.append(config)
        return _winning_runner(None, None, config)

    report = evaluate(
        EvalConfig(
            model_path=model_path,
            episodes=8,
            opponents=("random",),
            output_path=tmp_path / "report.json",
        ),
        model_loader=_model_loader,
        tournament_runner=recording_runner,
    )

    assert len(seen_configs) == 1
    tournament = seen_configs[0]
    assert tournament.enable_bidding is False
    assert tournament.enable_weis is True
    assert tournament.enable_stock is False
    assert tournament.modes == (MODE_TRUMP,)
    assert tournament.trump_suits == ("rosen",)
    assert tournament.profile.to_dict() == environment.profile.to_dict()
    assert report["evaluation"]["manifest_environment"] == environment.to_manifest_dict()
    assert report["run"]["git_commit"] == "abc123"
    assert report["run"]["actual_timesteps"] == 1234

    _write_run_manifest(model_path, environment, schema_version=999)
    with pytest.raises(ValueError, match="observation schema"):
        evaluate(
            EvalConfig(model_path=model_path, episodes=8, opponents=("random",)),
            model_loader=_model_loader,
            tournament_runner=_winning_runner,
        )


def test_evaluate_accepts_a_public_candidate_builder(tmp_path: Path) -> None:
    model_path = tmp_path / "model_final.zip"
    model_path.write_bytes(b"hybrid-model")
    environment = _environment()
    _write_run_manifest(model_path, environment)
    built = []
    seen = []

    def candidate_builder(model, context):
        assert isinstance(model, _FakeModel)
        assert context.environment == environment

        def candidate(observation, action_mask, agent):
            del observation, agent
            return int(np.flatnonzero(action_mask)[-1])

        built.append(candidate)
        return candidate

    def recording_runner(candidate, reference, config):
        del reference
        seen.append(candidate)
        return _winning_runner(None, None, config)

    evaluate(
        EvalConfig(model_path=model_path, episodes=8, opponents=("random",)),
        model_loader=_model_loader,
        tournament_runner=recording_runner,
        candidate_builder=candidate_builder,
    )

    assert seen == built


def test_all_opponents_use_paired_protocols_and_atomic_identity_report(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model_final.zip"
    model_bytes = b"identity-model"
    model_path.write_bytes(model_bytes)
    environment = _environment(
        enable_bidding=True,
        mode=None,
        trump_suit=None,
    )
    _write_run_manifest(model_path, environment)
    output_path = tmp_path / "nested" / "evaluation.json"
    calls = []

    def recording_runner(candidate, reference, config):
        calls.append((candidate, reference, config))
        return _winning_runner(candidate, reference, config)

    report = evaluate(
        EvalConfig(
            model_path=model_path,
            episodes=8,
            opponents=("random", "strategic", "verardo-random", "verardo"),
            output_path=output_path,
            include_results=True,
            device="cpu",
        ),
        model_loader=_model_loader,
        tournament_runner=recording_runner,
    )

    assert isinstance(calls[0][1], SeededRandomPolicy)
    assert isinstance(calls[1][1], StrategicHeuristicPolicy)
    assert isinstance(calls[2][0], TrumpOnlyPolicy)
    assert isinstance(calls[2][1], TrumpOnlyPolicy)
    assert isinstance(calls[2][1].policy, SeededRandomPolicy)
    assert isinstance(calls[3][0], TrumpOnlyPolicy)
    assert isinstance(calls[3][1], TrumpOnlyPolicy)
    assert isinstance(calls[3][1].policy, VerardoReferencePolicy)
    assert calls[0][2].modes == (None,)
    assert calls[0][2].profile.to_dict() == environment.profile.to_dict()
    assert calls[2][2].profile is VERARDO_V1_PROFILE
    assert calls[3][2].profile is VERARDO_V1_PROFILE
    assert calls[3][2].enable_bidding is True
    assert calls[3][2].enable_weis is False
    assert calls[3][2].enable_stock is False

    assert set(report["opponents"]) == {
        "random",
        "strategic",
        "verardo-random",
        "verardo",
    }
    for payload in report["opponents"].values():
        assert payload["samples"] == {
            "games": 8,
            "complete_pairs": 4,
            "starter_rotations": 1,
        }
        assert payload["metrics"]["overall"]["average_point_difference"] == 83.0
        assert payload["metrics"]["overall"]["win_rate_ci95"]["low"] > 0.0
        assert len(payload["metrics"]["results"]) == 8
        assert payload["qualification"]["checks"]["minimum_games"] is False

    assert report["model"]["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert report["model"]["num_timesteps"] == 1234
    assert report["report_version"] == 2
    assert "working_tree_sha256" in report["evaluation_provenance"]["git"]
    assert "platform" in report["evaluation_provenance"]["runtime"]
    assert report["qualification"] == {
        "full_project": False,
        "verardo_random": False,
        "verardo_reference": False,
        "verardo_project": False,
        "all_requested": False,
    }
    assert json.loads(output_path.read_text()) == report
    assert not output_path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("opponent", ("verardo-random", "verardo"))
def test_external_protocol_restricts_both_sides_and_all_games_to_trump(
    opponent: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _NoTrumpPreferringPolicy(BIDDING_OBEABE_ACTION)
    reference = _NoTrumpPreferringPolicy(BIDDING_UNEUFE_ACTION)
    if opponent == "verardo-random":
        monkeypatch.setattr(eval_module, "SeededRandomPolicy", lambda seed: reference)
    else:
        monkeypatch.setattr(eval_module, "VerardoReferencePolicy", lambda: reference)

    report, tournament = eval_module._run_opponent(
        opponent,
        candidate,
        _environment(),
        EvalConfig(model_path=Path("unused.zip"), episodes=8, seed=73),
        run_tournament,
    )

    assert tournament.profile is VERARDO_V1_PROFILE
    for policy in (candidate, reference):
        assert policy.bidding_masks
        assert all(mask[BIDDING_OBEABE_ACTION] == 0 for mask in policy.bidding_masks)
        assert all(mask[BIDDING_UNEUFE_ACTION] == 0 for mask in policy.bidding_masks)
    assert len(report.results) == 8
    assert all(result.mode == MODE_TRUMP for result in report.results)
    assert all(result.trump_suit is not None for result in report.results)


def test_qualification_requires_provenance_samples_wilson_and_difference() -> None:
    from rl.eval import _qualification

    metrics = AggregateMetrics(
        episodes=4_000,
        wins=2_800,
        losses=1_200,
        ties=0,
        win_rate=0.70,
        score_rate=0.70,
        win_rate_ci95=WilsonInterval(0.65, 0.75),
        average_candidate_points=120.0,
        average_reference_points=80.0,
        average_point_difference=40.0,
        pairs=2_000,
        paired_wins=1_400,
        paired_losses=600,
        paired_ties=0,
        paired_win_rate=0.70,
        paired_win_rate_ci95=WilsonInterval(0.64, 0.76),
    )
    tournament = TournamentReport(overall=metrics, by_mode={}, results=())

    qualified = _qualification("verardo", tournament, provenance_eligible=True)
    assert qualified["performance_qualified"] is True
    assert qualified["qualified"] is True

    legacy = _qualification("verardo", tournament, provenance_eligible=False)
    assert legacy["performance_qualified"] is True
    assert legacy["qualified"] is False


def test_ctde_manifest_accepts_private_critic_shape_and_pads_public_inference(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model_final.zip"
    model_path.write_bytes(b"ctde-model")
    environment = _environment(
        ctde=True,
        training_observation_shape=(CTDE_OBS_SIZE,),
        privileged_observation_schema=(
            PRIVILEGED_OBSERVATION_SCHEMA_NAME,
            PRIVILEGED_OBSERVATION_SCHEMA_VERSION,
        ),
    )
    _write_run_manifest(model_path, environment)

    def checking_runner(candidate, reference, config):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        mask[0] = 1
        assert candidate(np.zeros(OBS_SIZE, dtype=np.float32), mask, "p0") == 0
        return _winning_runner(candidate, reference, config)

    report = evaluate(
        EvalConfig(
            model_path=model_path,
            episodes=8,
            opponents=("random",),
        ),
        model_loader=lambda path, device: _FakeCTDEModel(),
        tournament_runner=checking_runner,
    )

    assert report["compatibility"]["actor_observation_shape"] == [OBS_SIZE]
    assert report["compatibility"]["model_observation_shape"] == [CTDE_OBS_SIZE]


def test_ctde_manifest_rejects_an_unaudited_private_input_policy(tmp_path: Path) -> None:
    model_path = tmp_path / "model_final.zip"
    model_path.write_bytes(b"private-actor")
    environment = _environment(
        ctde=True,
        training_observation_shape=(CTDE_OBS_SIZE,),
        privileged_observation_schema=(
            PRIVILEGED_OBSERVATION_SCHEMA_NAME,
            PRIVILEGED_OBSERVATION_SCHEMA_VERSION,
        ),
    )
    _write_run_manifest(model_path, environment)

    private_model = _FakeModel()
    private_model.observation_space = _Space((CTDE_OBS_SIZE,))
    with pytest.raises(ValueError, match="audited public-actor"):
        evaluate(
            EvalConfig(model_path=model_path, episodes=8, opponents=("random",)),
            model_loader=lambda path, device: private_model,
            tournament_runner=_winning_runner,
        )
