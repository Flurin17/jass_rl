from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from env.jass_aec_env import ACTION_COUNT, OBS_SIZE
from rl.baselines import StrategicHeuristicPolicy
from rl.imitation import (
    DATASET_MANIFEST_FILENAME,
    IMITATION_MANIFEST_FILENAME,
    BehaviorCloneConfig,
    CollectionConfig,
    collect_expert_dataset,
    initialize_behavior_clone_model,
    train_behavior_clone,
    validate_dataset,
)
from rl.privileged_observation import CTDE_OBS_SIZE
from rl.reference_verardo import TrumpOnlyPolicy
from rl.train_selfplay import _load_imitation_warm_start


def _dataset_arrays(directory: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((directory / DATASET_MANIFEST_FILENAME).read_text())
    combined: dict[str, list[np.ndarray]] = {}
    for entry in manifest["shards"]:
        with np.load(directory / entry["path"], allow_pickle=False) as archive:
            for name in archive.files:
                combined.setdefault(name, []).append(archive[name])
    return {name: np.concatenate(parts) for name, parts in combined.items()}


def test_collection_is_deterministic_public_legal_and_exactly_resumable(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    policy = StrategicHeuristicPolicy()

    direct_summary = collect_expert_dataset(
        policy,
        CollectionConfig(direct, games=3, seed=91, shard_size=50),
    )
    collect_expert_dataset(
        StrategicHeuristicPolicy(),
        CollectionConfig(resumed, games=1, seed=91, shard_size=50),
    )
    resumed_summary = collect_expert_dataset(
        StrategicHeuristicPolicy(),
        CollectionConfig(resumed, games=3, seed=91, shard_size=50, resume=True),
    )

    direct_arrays = _dataset_arrays(direct)
    resumed_arrays = _dataset_arrays(resumed)
    assert direct_summary.games == resumed_summary.games == 3
    assert direct_summary.samples == resumed_summary.samples
    assert direct_summary.dataset_id == resumed_summary.dataset_id
    for name in direct_arrays:
        np.testing.assert_array_equal(direct_arrays[name], resumed_arrays[name])

    observations = direct_arrays["observations"]
    masks = direct_arrays["action_masks"]
    actions = direct_arrays["actions"].astype(np.int64)
    assert observations.shape == (direct_summary.samples, OBS_SIZE)
    assert masks.shape == (direct_summary.samples, ACTION_COUNT)
    assert np.all(masks[np.arange(len(actions)), actions] == 1)
    manifest = json.loads((direct / DATASET_MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "complete"
    assert manifest["identity"]["observation_scope"] == "canonical_public_only"
    assert manifest["identity"]["observation_shape"] == [OBS_SIZE]


def test_collection_rejects_an_illegal_expert_label(tmp_path: Path) -> None:
    class IllegalPolicy:
        def __call__(self, observation, action_mask, agent) -> int:
            del observation, action_mask, agent
            return 0  # The first decision is bidding; card action zero is illegal.

    with pytest.raises(ValueError, match="illegal action"):
        collect_expert_dataset(
            IllegalPolicy(),
            CollectionConfig(tmp_path / "illegal", games=1, verify_determinism=False),
        )
    manifest = json.loads(
        ((tmp_path / "illegal") / DATASET_MANIFEST_FILENAME).read_text()
    )
    assert manifest["status"] == "failed"


def test_ctde_behavior_clone_updates_actor_only_and_resumes(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    collect_expert_dataset(
        TrumpOnlyPolicy(StrategicHeuristicPolicy()),
        CollectionConfig(
            dataset,
            games=6,
            seed=7,
            profile_name="verardo-v1",
            shard_size=80,
            expert_id="strategic-trump-only",
        ),
    )
    model = initialize_behavior_clone_model(
        dataset,
        ctde=True,
        net_arch=(16,),
        seed=11,
        device="cpu",
    )
    assert model.observation_space.shape == (CTDE_OBS_SIZE,)
    actor_before = {
        name: value.detach().clone()
        for name, value in model.policy.action_net.state_dict().items()
    }
    critic_before = {
        f"extractor.{name}": value.detach().clone()
        for name, value in model.policy.mlp_extractor.value_net.state_dict().items()
    }
    critic_before.update(
        {
            f"head.{name}": value.detach().clone()
            for name, value in model.policy.value_net.state_dict().items()
        }
    )

    output = tmp_path / "warmstart"
    first = train_behavior_clone(
        model,
        BehaviorCloneConfig(
            dataset,
            output,
            epochs=1,
            batch_size=64,
            learning_rate=1e-3,
            validation_fraction=0.2,
            seed=13,
            device="cpu",
        ),
    )
    assert first.model_path.is_file()
    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in model.policy.action_net.state_dict().items()
    )
    critic_after = {
        f"extractor.{name}": value
        for name, value in model.policy.mlp_extractor.value_net.state_dict().items()
    }
    critic_after.update(
        {f"head.{name}": value for name, value in model.policy.value_net.state_dict().items()}
    )
    for name, before in critic_before.items():
        torch.testing.assert_close(critic_after[name], before, rtol=0, atol=0)

    resumed = train_behavior_clone(
        model,
        BehaviorCloneConfig(
            dataset,
            output,
            epochs=2,
            batch_size=64,
            learning_rate=1e-3,
            validation_fraction=0.2,
            seed=13,
            device="cpu",
            resume=True,
        ),
    )
    assert resumed.epochs_completed == 2
    training_manifest = json.loads((output / IMITATION_MANIFEST_FILENAME).read_text())
    assert training_manifest["status"] == "complete"
    assert len(training_manifest["history"]) == 2
    assert training_manifest["identity"]["dataset_id"] == validate_dataset(
        dataset
    ).dataset_id

    target = initialize_behavior_clone_model(
        dataset,
        ctde=True,
        net_arch=(16,),
        seed=99,
        device="cpu",
    )
    warm_start = _load_imitation_warm_start(resumed.model_path, target)
    assert warm_start["dataset_id"] == training_manifest["identity"]["dataset_id"]
    for name, value in model.policy.action_net.state_dict().items():
        torch.testing.assert_close(target.policy.action_net.state_dict()[name], value)


def test_plain_maskable_ppo_actor_is_supported(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    collect_expert_dataset(
        TrumpOnlyPolicy(StrategicHeuristicPolicy()),
        CollectionConfig(
            dataset,
            games=3,
            seed=22,
            profile_name="verardo-v1",
            expert_id="strategic-trump-only",
        ),
    )
    model = initialize_behavior_clone_model(
        dataset,
        ctde=False,
        net_arch=(8,),
        seed=22,
        device="cpu",
    )
    assert model.observation_space.shape == (OBS_SIZE,)
    report = train_behavior_clone(
        model,
        BehaviorCloneConfig(
            dataset,
            tmp_path / "plain",
            epochs=1,
            batch_size=64,
            validation_fraction=0.2,
            seed=22,
            device="cpu",
        ),
    )
    assert 0.0 <= report.final_validation_accuracy <= 1.0
    assert report.model_path.is_file()


def test_collection_resource_guards_fail_before_allocation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="16 GiB"):
        CollectionConfig(
            tmp_path,
            games=1,
            raw_dataset_limit_bytes=17 * 1024**3,
        ).validate()
    with pytest.raises(ValueError, match="worst-case"):
        CollectionConfig(
            tmp_path,
            games=10_000_000,
            raw_dataset_limit_bytes=1024,
        ).validate()
