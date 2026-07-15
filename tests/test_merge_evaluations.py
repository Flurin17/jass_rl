import json
from pathlib import Path

import pytest

from rl.merge_evaluations import merge_metrics, merge_reports
from rl.tournament import wilson_interval


def _metric(*, wins: int, paired_wins: int, point_difference: float) -> dict:
    episodes = 8
    pairs = 4
    losses = episodes - wins
    paired_losses = pairs - paired_wins
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "ties": 0,
        "win_rate": wins / episodes,
        "score_rate": wins / episodes,
        "win_rate_ci95": vars(wilson_interval(wins, episodes)),
        "average_candidate_points": 80.0 + point_difference / 2,
        "average_reference_points": 80.0 - point_difference / 2,
        "average_point_difference": point_difference,
        "pairs": pairs,
        "paired_wins": paired_wins,
        "paired_losses": paired_losses,
        "paired_ties": 0,
        "paired_win_rate": paired_wins / pairs,
        "paired_win_rate_ci95": vars(wilson_interval(paired_wins, pairs)),
        "candidate_matches": 1,
        "reference_matches": 0,
        "match_rate": 1 / episodes,
        "candidate_decisions": 100,
        "candidate_inference_seconds": 1.0,
        "mean_candidate_inference_ms": 10.0,
    }


def _report(seed: int, metric: dict) -> dict:
    return {
        "report_version": 2,
        "model": {"sha256": "model", "path": "/model.zip"},
        "run": {"manifest_sha256": "manifest"},
        "evaluation_provenance": {
            "git": {
                "commit": "abc",
                "working_tree_sha256": "tree",
                "dirty": False,
            },
            "runtime": {"python": "test"},
        },
        "compatibility": {
            "manifest_validated": True,
            "artifact_hash_validated": True,
            "actor_privacy_validated": True,
        },
        "evaluation": {
            "seed": seed,
            "games_per_opponent": 8,
            "opponents": ["verardo"],
            "paired_same_deal": True,
            "team_swap": True,
            "starters": [0, 1, 2, 3],
            "manifest_environment": {"profile": "test"},
            "full_project_environment_eligible": False,
        },
        "candidate_policy": {
            "type": "NeuralGuidedPIMCPolicy",
            "model_sha256": "model",
            "search_config": {"determinizations": 24},
            "guidance_config": {"max_search_gap_raw_points": 3.0},
            "search_profile": {"version": "verardo-v1"},
            "stats": {"decisions": 100, "overrides": 10},
            "last_decision": None,
        },
        "opponents": {
            "verardo": {
                "protocol": "verardo-v1-matched",
                "environment": {"profile": {"version": "verardo-v1"}},
                "samples": {
                    "games": 8,
                    "complete_pairs": 4,
                    "starter_rotations": 1,
                },
                "metrics": {
                    "overall": metric,
                    "by_mode": {"trump": metric},
                    "by_contract": {"trump:rosen": metric},
                    "team_swap_supported": True,
                    "team_swap_method": "direct AEC callback assignment",
                },
                "qualification": {
                    "checks": {
                        "manifest_provenance": True,
                        "protocol_environment": True,
                    }
                },
            }
        },
    }


def test_merge_metrics_recomputes_counts_averages_and_intervals() -> None:
    merged = merge_metrics(
        [
            _metric(wins=6, paired_wins=3, point_difference=20.0),
            _metric(wins=5, paired_wins=4, point_difference=40.0),
        ]
    )

    assert merged["episodes"] == 16
    assert merged["wins"] == 11
    assert merged["win_rate"] == 11 / 16
    assert merged["pairs"] == 8
    assert merged["paired_wins"] == 7
    assert merged["average_point_difference"] == 30.0
    assert merged["candidate_decisions"] == 200
    assert merged["candidate_inference_seconds"] == 2.0
    assert merged["mean_candidate_inference_ms"] == 10.0


def test_merge_reports_requires_disjoint_complete_compatible_shards(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(_report(100, _metric(wins=6, paired_wins=3, point_difference=20.0))))
    right.write_text(
        json.dumps(_report(104, _metric(wins=5, paired_wins=4, point_difference=40.0)))
    )
    output = tmp_path / "merged.json"

    report = merge_reports([left, right], output=output, required_games=16)

    overall = report["opponents"]["verardo"]["metrics"]["overall"]
    assert overall["episodes"] == 16
    assert overall["wins"] == 11
    assert report["opponents"]["verardo"]["samples"]["independent_deal_seeds"] == 8
    assert report["candidate_policy"]["stats"] == {
        "decisions": 200,
        "overrides": 20,
    }
    assert len(report["evaluation"]["shards"]) == 2
    assert json.loads(output.read_text()) == report

    overlapping = _report(102, _metric(wins=5, paired_wins=4, point_difference=40.0))
    right.write_text(json.dumps(overlapping))
    with pytest.raises(ValueError, match="overlap"):
        merge_reports([left, right], output=output, required_games=16)


def test_merge_reports_rejects_dirty_shards_even_when_they_agree(tmp_path: Path) -> None:
    paths = []
    for name, seed in (("left", 100), ("right", 104)):
        report = _report(seed, _metric(wins=6, paired_wins=3, point_difference=20.0))
        report["evaluation_provenance"]["git"]["dirty"] = True
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths.append(path)

    with pytest.raises(ValueError, match=r"git\.dirty must be exactly false"):
        merge_reports(paths, output=tmp_path / "merged.json", required_games=16)


def test_merge_reports_rejects_shards_with_missing_dirty_flag(tmp_path: Path) -> None:
    paths = []
    for name, seed in (("left", 100), ("right", 104)):
        report = _report(seed, _metric(wins=6, paired_wins=3, point_difference=20.0))
        del report["evaluation_provenance"]["git"]["dirty"]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report))
        paths.append(path)

    with pytest.raises(ValueError, match=r"git\.dirty must be exactly false"):
        merge_reports(paths, output=tmp_path / "merged.json", required_games=16)
