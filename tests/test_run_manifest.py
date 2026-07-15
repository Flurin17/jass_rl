from pathlib import Path

import pytest

from rl.run_manifest import (
    MANIFEST_FILENAME,
    assert_compatible,
    build_manifest,
    hardware_metadata,
    load_manifest,
    write_manifest,
)


def test_manifest_roundtrip_and_compatibility(tmp_path: Path) -> None:
    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[1],
        training_config={"seed": 7, "save_dir": tmp_path},
        environment={"control_team": True, "enable_bidding": False},
        observation_schema_version=3,
        observation_shape=(512,),
        action_count=45,
    )
    write_manifest(tmp_path / MANIFEST_FILENAME, manifest)

    loaded = load_manifest(tmp_path)
    assert loaded["training_config"]["save_dir"] == str(tmp_path)
    assert_compatible(
        loaded,
        observation_schema_version=3,
        observation_shape=(512,),
        action_count=45,
        environment={"control_team": True, "enable_bidding": False},
    )


def test_hardware_metadata_records_auditable_mac_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "hw.model": "Mac15,7",
        "machdep.cpu.brand_string": "Apple M3 Pro",
        "hw.memsize": "38654705664",
    }
    monkeypatch.setattr("rl.run_manifest.platform.system", lambda: "Darwin")
    monkeypatch.setattr("rl.run_manifest._sysctl_value", values.get)

    assert hardware_metadata() == {
        "model_identifier": "Mac15,7",
        "chip": "Apple M3 Pro",
        "physical_memory_bytes": 38_654_705_664,
    }


def test_manifest_rejects_same_shape_but_different_rules(tmp_path: Path) -> None:
    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[1],
        training_config={},
        environment={"enable_weis": True},
        observation_schema_version=1,
        observation_shape=(10,),
        action_count=45,
    )
    write_manifest(tmp_path / MANIFEST_FILENAME, manifest)

    with pytest.raises(ValueError, match="environment.enable_weis"):
        assert_compatible(
            load_manifest(tmp_path),
            observation_schema_version=1,
            observation_shape=(10,),
            action_count=45,
            environment={"enable_weis": False},
        )
