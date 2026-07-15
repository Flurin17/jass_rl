import random
from pathlib import Path

import numpy as np
import pytest
import torch

from core.cards import MODE_TRUMP, SUITS
from rl.run_manifest import write_manifest
from rl.train_selfplay import (
    DEFAULT_CPU_THREADS,
    TrainConfig,
    _atomic_model_save,
    _build_env,
    _load_model_preserving_rng,
    _parse_args,
    _predict_preserving_rng,
    _resume_model_path,
    assert_resume_git_compatible,
    assert_resume_training_compatible,
    rollout_chunks,
    split_steps,
)


def test_split_steps_preserves_exact_budget() -> None:
    assert split_steps(10, 3) == (4, 3, 3)
    assert sum(split_steps(1_000_003, 10)) == 1_000_003


def test_rollout_chunks_make_ppo_rounding_explicit() -> None:
    assert rollout_chunks(0, 10, 8, 3) == (8, 8)
    assert rollout_chunks(16, 33, 8, 2) == (16, 8)
    assert sum(rollout_chunks(0, 1_000_000, 8192, 10)) == 1_007_616


def test_resume_rejects_hyperparameters_that_load_would_ignore() -> None:
    config = TrainConfig()
    recorded = {
        field: list(value) if isinstance(value, tuple) else value
        for field, value in vars(config).items()
    }
    assert_resume_training_compatible({"training_config": recorded}, config)
    recorded["n_steps"] = config.n_steps // 2
    with pytest.raises(ValueError, match="n_steps"):
        assert_resume_training_compatible({"training_config": recorded}, config)

    recorded = {
        field: list(value) if isinstance(value, tuple) else value
        for field, value in vars(config).items()
    }
    recorded.pop("cpu_threads")
    with pytest.raises(ValueError, match="cpu_threads is unrecorded"):
        assert_resume_training_compatible({"training_config": recorded}, config)


def test_resume_accepts_exact_git_provenance() -> None:
    original = {"commit": "abc123", "working_tree_sha256": "tree123"}
    current = {"commit": "abc123", "working_tree_sha256": "tree123"}

    assert_resume_git_compatible(original, current)


def test_resume_rejects_different_git_commit_even_when_tree_matches() -> None:
    original = {"commit": "abc123", "working_tree_sha256": "tree123"}
    current = {"commit": "def456", "working_tree_sha256": "tree123"}

    with pytest.raises(
        ValueError,
        match=r"git commit changed.*recorded 'abc123', current 'def456'",
    ):
        assert_resume_git_compatible(original, current)


def test_resume_rejects_different_working_tree_on_same_commit() -> None:
    original = {"commit": "abc123", "working_tree_sha256": "tree123"}
    current = {"commit": "abc123", "working_tree_sha256": "tree456"}

    with pytest.raises(ValueError, match="working tree changed"):
        assert_resume_git_compatible(original, current)


@pytest.mark.parametrize("original", [None, {}, {"working_tree_sha256": "tree123"}])
def test_resume_rejects_missing_git_commit_provenance(original: object) -> None:
    current = {"commit": "abc123", "working_tree_sha256": "tree123"}

    with pytest.raises(ValueError, match="git.*provenance"):
        assert_resume_git_compatible(original, current)


@pytest.mark.parametrize(
    "config,match",
    [
        (TrainConfig(total_steps=0), "total_steps"),
        (TrainConfig(total_steps=4, iterations=5), "iterations"),
        (TrainConfig(selfplay=True, control_team=False), "control-team"),
        (TrainConfig(ctde=True, control_team=False), "CTDE"),
        (
            TrainConfig(resume=Path("run"), initialize_from=Path("warm.zip")),
            "mutually exclusive",
        ),
        (TrainConfig(selfplay=True, vec_env="subproc"), "vec-env dummy"),
        (TrainConfig(n_steps=10, n_envs=1, batch_size=6), "evenly divide"),
        (TrainConfig(cpu_threads=0), "cpu_threads"),
        (TrainConfig(profile_name="missing"), "profile_name"),
    ],
)
def test_invalid_training_configs_fail_fast(config: TrainConfig, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_cli_defaults_fit_m3_local_team_training() -> None:
    config = _parse_args(["--total-steps", "1024", "--iterations", "2"])
    assert config.control_team is True
    assert config.ctde is True
    assert config.randomize_starter is True
    assert config.net_arch == (512, 256)
    assert config.cpu_threads == DEFAULT_CPU_THREADS
    assert config.save_dir == Path("models")
    config.validate()


def test_cli_can_override_cpu_threads() -> None:
    config = _parse_args(
        ["--total-steps", "1024", "--iterations", "2", "--cpu-threads", "12"]
    )

    assert config.cpu_threads == 12


def test_fixed_trump_curriculum_can_randomize_suit_each_round() -> None:
    config = TrainConfig(enable_bidding=False, mode=MODE_TRUMP, trump_suit=None)

    config.validate()
    env = _build_env(config, seed=81)
    try:
        observed = set()
        for seed in range(81, 113):
            env.reset(seed=seed)
            observed.add(env.env.trump_suit)
            assert env.env.mode == MODE_TRUMP
        assert observed == set(SUITS)
    finally:
        env.close()


def test_directory_resume_prefers_latest_checkpoint_when_final_is_stale(
    tmp_path: Path,
) -> None:
    (tmp_path / "model_final.zip").write_bytes(b"stale")
    (tmp_path / "checkpoint_000000000032.zip").write_bytes(b"32")
    latest = tmp_path / "checkpoint_000000000064.zip"
    latest.write_bytes(b"64")
    write_manifest(
        tmp_path / "run_manifest.json",
        {
            "manifest_version": 1,
            "status": "interrupted",
            "actual_timesteps": 64,
            "checkpoints": [
                "checkpoint_000000000032.zip",
                "checkpoint_000000000064.zip",
            ],
        },
    )

    assert _resume_model_path(tmp_path) == (tmp_path, latest)


def test_opponent_load_and_prediction_restore_global_rng(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        def predict(self, observation, *, action_masks, deterministic):
            del observation, action_masks, deterministic
            random.random()
            np.random.random()
            torch.rand(1)
            return np.array([3]), None

    def disruptive_load(path, *, device):
        del path, device
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        return FakeModel()

    monkeypatch.setattr("rl.train_selfplay.MaskablePPO.load", disruptive_load)
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None

    model = _load_model_preserving_rng(tmp_path / "checkpoint.zip", device="cpu")
    assert random.getstate() == python_state
    assert np.random.get_state()[1].tolist() == numpy_state[1].tolist()
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if mps_state is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_state)

    assert _predict_preserving_rng(
        model,
        np.zeros(4, dtype=np.float32),
        np.ones(5, dtype=np.int8),
        seed=123,
    ) == 3
    assert random.getstate() == python_state
    assert np.random.get_state()[1].tolist() == numpy_state[1].tolist()
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if mps_state is not None:
        assert torch.equal(torch.mps.get_rng_state(), mps_state)


def test_model_save_is_atomic(tmp_path: Path) -> None:
    class FakeModel:
        def save(self, path: Path) -> None:
            Path(path).write_bytes(b"model")

    destination = tmp_path / "checkpoint_000000000008.zip"
    _atomic_model_save(FakeModel(), destination)
    assert destination.read_bytes() == b"model"
    assert not list(tmp_path.glob("*.tmp.zip"))
