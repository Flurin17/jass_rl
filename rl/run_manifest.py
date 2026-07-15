from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"
TRACKED_PACKAGES = (
    "gymnasium",
    "jass-rl",
    "numpy",
    "pettingzoo",
    "sb3-contrib",
    "stable-baselines3",
    "torch",
)
MAX_PROVENANCE_FILE_BYTES = 16 * 1024**2
MAX_PROVENANCE_TOTAL_BYTES = 128 * 1024**2


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _run_git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _sysctl_value(name: str) -> str | None:
    """Read one non-sensitive macOS hardware fact when sysctl is available."""

    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def hardware_metadata() -> dict[str, Any]:
    """Return portable hardware identity without serial or device identifiers."""

    model_identifier = None
    chip = platform.processor() or None
    physical_memory_bytes = None
    if platform.system() == "Darwin":
        model_identifier = _sysctl_value("hw.model")
        chip = _sysctl_value("machdep.cpu.brand_string") or chip
        raw_memory = _sysctl_value("hw.memsize")
        if raw_memory is not None:
            try:
                physical_memory_bytes = int(raw_memory)
            except ValueError:
                physical_memory_bytes = None
    elif hasattr(os, "sysconf"):
        try:
            physical_memory_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(
                os.sysconf("SC_PHYS_PAGES")
            )
        except (OSError, TypeError, ValueError):
            physical_memory_bytes = None
    return {
        "model_identifier": model_identifier,
        "chip": chip,
        "physical_memory_bytes": physical_memory_bytes,
    }


def git_metadata(repo_root: Path) -> dict[str, Any]:
    status = _run_git(repo_root, "status", "--porcelain=v1")
    diff = _run_git(repo_root, "diff", "--binary", "HEAD")
    untracked_output = _run_git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    untracked_files = sorted(untracked_output.splitlines()) if untracked_output else []
    working_tree_digest = hashlib.sha256((diff or "").encode())
    hashed_untracked: list[str] = []
    skipped_untracked: list[dict[str, Any]] = []
    hashed_bytes = 0
    root = repo_root.resolve()
    for relative in untracked_files:
        path = (repo_root / relative).resolve()
        if not path.is_file() or root not in path.parents:
            continue
        size = path.stat().st_size
        if (
            size > MAX_PROVENANCE_FILE_BYTES
            or hashed_bytes + size > MAX_PROVENANCE_TOTAL_BYTES
        ):
            working_tree_digest.update(relative.encode())
            working_tree_digest.update(f"\0SKIPPED:{size}".encode())
            skipped_untracked.append({"path": relative, "size_bytes": size})
            continue
        working_tree_digest.update(relative.encode())
        working_tree_digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                working_tree_digest.update(chunk)
        hashed_untracked.append(relative)
        hashed_bytes += size
    return {
        "commit": _run_git(repo_root, "rev-parse", "HEAD"),
        "branch": _run_git(repo_root, "branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "tracked_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
        "working_tree_sha256": working_tree_digest.hexdigest(),
        "hashed_untracked_files": hashed_untracked,
        "hashed_untracked_bytes": hashed_bytes,
        "skipped_untracked_files": skipped_untracked,
    }


def runtime_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    accelerator: dict[str, Any] = {"mps_available": False, "mps_built": False}
    try:
        import torch

        accelerator = {
            "mps_available": bool(torch.backends.mps.is_available()),
            "mps_built": bool(torch.backends.mps.is_built()),
        }
    except (ImportError, AttributeError):
        pass

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "hardware": hardware_metadata(),
        "accelerator": accelerator,
        "packages": packages,
    }


def build_manifest(
    *,
    repo_root: Path,
    training_config: Any,
    environment: Mapping[str, Any],
    observation_schema_version: int,
    observation_shape: tuple[int, ...],
    action_count: int,
    resumed_from: str | None = None,
) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "initialized",
        "training_config": _jsonable(training_config),
        "environment": _jsonable(environment),
        "observation_schema_version": observation_schema_version,
        "observation_shape": list(observation_shape),
        "action_count": action_count,
        "resumed_from": resumed_from,
        "git": git_metadata(repo_root),
        "runtime": runtime_metadata(),
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_manifest(model_or_directory: str | Path) -> dict[str, Any]:
    candidate = Path(model_or_directory)
    if candidate.is_dir():
        path = candidate / MANIFEST_FILENAME
    else:
        path = candidate.parent / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_FILENAME} next to {candidate}. This is a legacy or incomplete run."
        )
    payload = json.loads(path.read_text())
    if payload.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported run manifest version {payload.get('manifest_version')!r}; "
            f"expected {MANIFEST_VERSION}."
        )
    return payload


def assert_compatible(
    manifest: Mapping[str, Any],
    *,
    observation_schema_version: int,
    observation_shape: tuple[int, ...],
    action_count: int,
    environment: Mapping[str, Any],
) -> None:
    mismatches: list[str] = []
    if manifest.get("observation_schema_version") != observation_schema_version:
        mismatches.append(
            "observation schema "
            f"{manifest.get('observation_schema_version')} != {observation_schema_version}"
        )
    if tuple(manifest.get("observation_shape", ())) != tuple(observation_shape):
        mismatches.append(
            f"observation shape {manifest.get('observation_shape')} != {list(observation_shape)}"
        )
    if manifest.get("action_count") != action_count:
        mismatches.append(f"action count {manifest.get('action_count')} != {action_count}")

    recorded_environment = manifest.get("environment", {})
    for key, value in environment.items():
        if recorded_environment.get(key) != _jsonable(value):
            mismatches.append(
                f"environment.{key} {recorded_environment.get(key)!r} != {_jsonable(value)!r}"
            )

    if mismatches:
        raise ValueError("Model/environment incompatibility: " + "; ".join(mismatches))
