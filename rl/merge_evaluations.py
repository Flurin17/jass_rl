"""Strictly merge disjoint paired qualification shards into one report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl.eval import _overall_qualification, _qualification
from rl.run_manifest import git_metadata, runtime_metadata, write_manifest
from rl.tournament import AggregateMetrics, TournamentReport, WilsonInterval, wilson_interval

MERGED_REPORT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same(label: str, values: Sequence[Any]) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"qualification shards disagree on {label}")
    return first


def _weighted(rows: Sequence[Mapping[str, Any]], key: str, weight: str) -> float:
    denominator = sum(int(row[weight]) for row in rows)
    if denominator == 0:
        return 0.0
    return sum(float(row[key]) * int(row[weight]) for row in rows) / denominator


def merge_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge sufficient aggregate statistics without episode-level payloads."""

    if not rows:
        raise ValueError("at least one metric row is required")
    episodes = sum(int(row["episodes"]) for row in rows)
    pairs = sum(int(row["pairs"]) for row in rows)
    wins = sum(int(row["wins"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    ties = sum(int(row["ties"]) for row in rows)
    paired_wins = sum(int(row["paired_wins"]) for row in rows)
    paired_losses = sum(int(row["paired_losses"]) for row in rows)
    paired_ties = sum(int(row["paired_ties"]) for row in rows)
    if wins + losses + ties != episodes:
        raise ValueError("game counts do not sum to episodes")
    if paired_wins + paired_losses + paired_ties != pairs:
        raise ValueError("paired counts do not sum to pairs")
    candidate_matches = sum(int(row.get("candidate_matches", 0)) for row in rows)
    reference_matches = sum(int(row.get("reference_matches", 0)) for row in rows)
    candidate_decisions = sum(int(row.get("candidate_decisions", 0)) for row in rows)
    candidate_seconds = sum(float(row.get("candidate_inference_seconds", 0.0)) for row in rows)
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / episodes,
        "score_rate": (wins + 0.5 * ties) / episodes,
        "win_rate_ci95": vars(wilson_interval(wins, episodes)),
        "average_candidate_points": _weighted(rows, "average_candidate_points", "episodes"),
        "average_reference_points": _weighted(rows, "average_reference_points", "episodes"),
        "average_point_difference": _weighted(rows, "average_point_difference", "episodes"),
        "pairs": pairs,
        "paired_wins": paired_wins,
        "paired_losses": paired_losses,
        "paired_ties": paired_ties,
        "paired_win_rate": paired_wins / pairs if pairs else 0.0,
        "paired_win_rate_ci95": vars(wilson_interval(paired_wins, pairs)),
        "candidate_matches": candidate_matches,
        "reference_matches": reference_matches,
        "match_rate": candidate_matches / episodes,
        "candidate_decisions": candidate_decisions,
        "candidate_inference_seconds": candidate_seconds,
        "mean_candidate_inference_ms": (
            candidate_seconds * 1000.0 / candidate_decisions if candidate_decisions else 0.0
        ),
    }


def _merge_groups(
    reports: Sequence[Mapping[str, Any]],
    opponent: str,
    group: str,
) -> dict[str, Any]:
    keys = sorted(
        {key for report in reports for key in report["opponents"][opponent]["metrics"][group]}
    )
    return {
        key: merge_metrics(
            [
                report["opponents"][opponent]["metrics"][group][key]
                for report in reports
                if key in report["opponents"][opponent]["metrics"][group]
            ]
        )
        for key in keys
    }


def _aggregate_object(payload: Mapping[str, Any]) -> AggregateMetrics:
    values = dict(payload)
    values["win_rate_ci95"] = WilsonInterval(**values["win_rate_ci95"])
    values["paired_win_rate_ci95"] = WilsonInterval(**values["paired_win_rate_ci95"])
    allowed = {field.name for field in fields(AggregateMetrics)}
    return AggregateMetrics(**{key: value for key, value in values.items() if key in allowed})


def _candidate_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(payload))
    identity.pop("stats", None)
    identity.pop("last_decision", None)
    return identity


def _merge_candidate_stats(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats = [report["candidate_policy"]["stats"] for report in reports]
    keys = sorted({key for row in stats for key in row})
    merged: dict[str, Any] = {}
    for key in keys:
        values = [row.get(key, 0) for row in stats]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"candidate statistic {key} is not numeric")
        merged[key] = sum(values)
    return merged


def merge_reports(
    paths: Sequence[Path],
    *,
    output: Path,
    required_games: int = 4_000,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two shard reports are required")
    reports = [json.loads(path.read_text()) for path in paths]
    _same("report version", [report.get("report_version") for report in reports])
    opponents = [tuple(report.get("opponents", {})) for report in reports]
    opponent_tuple = _same("opponent set", opponents)
    if len(opponent_tuple) != 1:
        raise ValueError("each shard must contain exactly one opponent")
    opponent = opponent_tuple[0]

    projections = []
    for report in reports:
        candidate = report.get("candidate_policy")
        if not isinstance(candidate, Mapping):
            raise ValueError("every shard must describe candidate_policy")
        evaluation_provenance = report.get("evaluation_provenance")
        evaluation_git = (
            evaluation_provenance.get("git")
            if isinstance(evaluation_provenance, Mapping)
            else None
        )
        if not isinstance(evaluation_git, Mapping) or evaluation_git.get("dirty") is not False:
            raise ValueError(
                "every shard evaluation_provenance.git.dirty must be exactly false"
            )
        payload = report["opponents"][opponent]
        checks = payload["qualification"]["checks"]
        if not checks.get("manifest_provenance") or not checks.get("protocol_environment"):
            raise ValueError("every shard must pass provenance and protocol checks")
        projections.append(
            {
                "model": report["model"],
                "run": report["run"],
                "compatibility": report["compatibility"],
                "evaluation_provenance": evaluation_provenance,
                "candidate": _candidate_identity(candidate),
                "environment": payload["environment"],
                "protocol": payload["protocol"],
                "evaluation": {
                    key: value
                    for key, value in report["evaluation"].items()
                    if key not in {"seed", "games_per_opponent"}
                },
            }
        )
    shared = _same("model, policy, environment, and provenance", projections)

    seed_ranges = []
    used_deal_seeds: set[int] = set()
    for path, report in zip(paths, reports, strict=True):
        seed = report["evaluation"]["seed"]
        overall = report["opponents"][opponent]["metrics"]["overall"]
        games = int(overall["episodes"])
        pairs = int(overall["pairs"])
        if games != pairs * 2 or games % 8:
            raise ValueError(f"shard {path} is not complete paired starter rotations")
        deal_seeds = set(range(seed, seed + pairs))
        if used_deal_seeds.intersection(deal_seeds):
            raise ValueError("shard deal-seed ranges overlap")
        used_deal_seeds.update(deal_seeds)
        seed_ranges.append(
            {
                "report": str(path.resolve()),
                "sha256": _sha256(path),
                "seed_first": seed,
                "seed_last": seed + pairs - 1,
                "games": games,
                "pairs": pairs,
            }
        )

    overall = merge_metrics(
        [report["opponents"][opponent]["metrics"]["overall"] for report in reports]
    )
    if overall["episodes"] != required_games:
        raise ValueError(
            f"merged report has {overall['episodes']} games; expected {required_games}"
        )
    tournament_report = TournamentReport(
        overall=_aggregate_object(overall),
        by_mode={},
        results=(),
    )
    qualification = _qualification(
        opponent,
        tournament_report,
        provenance_eligible=True,
        protocol_eligible=True,
    )
    first_payload = reports[0]["opponents"][opponent]
    opponent_payload = {
        "protocol": first_payload["protocol"],
        "environment": first_payload["environment"],
        "samples": {
            "games": overall["episodes"],
            "complete_pairs": overall["pairs"],
            "starter_rotations": sum(
                int(report["opponents"][opponent]["samples"]["starter_rotations"])
                for report in reports
            ),
            "independent_deal_seeds": len(used_deal_seeds),
        },
        "metrics": {
            "overall": overall,
            "by_mode": _merge_groups(reports, opponent, "by_mode"),
            "by_contract": _merge_groups(reports, opponent, "by_contract"),
            "team_swap_supported": True,
            "team_swap_method": "direct AEC callback assignment",
        },
        "qualification": qualification,
    }
    candidate = copy.deepcopy(reports[0]["candidate_policy"])
    candidate["stats"] = _merge_candidate_stats(reports)
    candidate["last_decision"] = reports[-1]["candidate_policy"].get("last_decision")
    repo_root = Path(__file__).resolve().parents[1]
    merged: dict[str, Any] = {
        "report_version": reports[0]["report_version"],
        "merged_report_version": MERGED_REPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": reports[0]["model"],
        "run": reports[0]["run"],
        "evaluation_provenance": {
            "git": git_metadata(repo_root),
            "runtime": runtime_metadata(),
        },
        "compatibility": reports[0]["compatibility"],
        "evaluation": {
            **shared["evaluation"],
            "seed": None,
            "games_per_opponent": overall["episodes"],
            "sharded": True,
            "shards": seed_ranges,
        },
        "candidate_policy": candidate,
        "opponents": {opponent: opponent_payload},
        "qualification": _overall_qualification({opponent: opponent_payload}),
        "report_path": str(output.resolve()),
    }
    normalized = json.loads(json.dumps(merged))
    write_manifest(output, normalized)
    return normalized


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint qualification shards")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-games", type=int, default=4_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = merge_reports(
            args.reports,
            output=args.output,
            required_games=args.required_games,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MERGED_REPORT_VERSION", "main", "merge_metrics", "merge_reports"]
