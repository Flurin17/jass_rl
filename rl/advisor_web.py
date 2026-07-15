"""Local-only web UI for entering a public game state and requesting advice."""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from rl.advisor import AdvisorState, JassAdvisor, load_advisor
from rl.eval import QUALIFICATION_GATES
from rl.hybrid_policy import NeuralGuidanceConfig
from rl.search_policy import PIMCConfig

MAX_REQUEST_BYTES = 1024 * 1024
UI_PATH = Path(__file__).with_name("static") / "advisor.html"
LOGGER = logging.getLogger(__name__)


def _validate_qualification_gate(
    path: Path,
    opponent: str,
    report: dict[str, Any],
    qualification: dict[str, Any],
) -> None:
    gate = dict(QUALIFICATION_GATES.get(opponent, {}))
    if not gate or qualification.get("thresholds") != gate:
        raise ValueError(f"qualification report {path} has unexpected gate thresholds")
    overall = report.get("metrics", {}).get("overall", {})
    try:
        episodes = overall["episodes"]
        pairs = overall["pairs"]
        win_rate = overall["win_rate"]
        win_rate_low = overall["win_rate_ci95"]["low"]
        paired_win_rate = overall["paired_win_rate"]
        paired_win_rate_low = overall["paired_win_rate_ci95"]["low"]
        point_difference = overall["average_point_difference"]
        minimum_difference = gate["minimum_average_point_difference"]
        recorded_checks = qualification["checks"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"qualification report {path} has incomplete gate metrics") from exc
    try:
        expected_checks = {
            "manifest_provenance": True,
            "protocol_environment": True,
            "minimum_games": episodes >= gate["minimum_games"],
            "minimum_pairs": pairs >= gate["minimum_pairs"],
            "complete_pairing": pairs * 2 == episodes,
            "minimum_win_rate": win_rate >= gate["minimum_win_rate"],
            "minimum_wilson_low": win_rate_low > gate["minimum_wilson_low"],
            "minimum_paired_win_rate": paired_win_rate
            >= gate["minimum_paired_win_rate"],
            "minimum_paired_wilson_low": paired_win_rate_low
            > gate["minimum_paired_wilson_low"],
            "minimum_average_point_difference": (
                True
                if minimum_difference is None
                else point_difference >= minimum_difference
            ),
        }
    except TypeError as exc:
        raise ValueError(f"qualification report {path} has non-numeric gate metrics") from exc
    if recorded_checks != expected_checks:
        raise ValueError(f"qualification report {path} has inconsistent gate checks")
    performance_checks = {
        key: passed
        for key, passed in expected_checks.items()
        if key not in {"manifest_provenance", "protocol_environment"}
    }
    if qualification.get("performance_qualified") is not all(
        performance_checks.values()
    ) or qualification.get("qualified") is not all(expected_checks.values()):
        raise ValueError(f"qualification report {path} has inconsistent qualification status")
    if not all(expected_checks.values()):
        raise ValueError(f"qualification report {path} did not pass every formal gate")


def example_state(advisor: JassAdvisor | None = None) -> dict[str, Any]:
    bidding_enabled = advisor.bidding_enabled if advisor is not None else True
    announcement_enabled = (
        advisor.announcement_enabled if advisor is not None else True
    )
    if bidding_enabled is None:
        bidding_enabled = True
    if announcement_enabled is None:
        announcement_enabled = True
    mode = advisor.fixed_mode if advisor is not None else None
    if mode is None:
        mode = "trump"
    trump_suit = advisor.fixed_trump_suit if advisor is not None else None
    if mode == "trump" and trump_suit is None:
        trump_suit = "schilten"
    return {
        "hand": [
            "schellen:A",
            "schellen:10",
            "rosen:K",
            "rosen:8",
            "schilten:J",
            "schilten:9",
            "eicheln:A",
            "eicheln:Q",
            "eicheln:6",
        ],
        "completed_tricks": [],
        "current_trick": [],
        "mode": mode,
        "trump_suit": trump_suit,
        "leader": 0,
        "team_points": [0, 0],
        "bidding_enabled": bidding_enabled,
        "bid_starter": 0 if bidding_enabled else None,
        "bid_chooser": 0 if bidding_enabled else None,
        "bid_pushed": False,
        "announcement_enabled": announcement_enabled,
        "announcement_status": ["pass", "pass", "pass", "pass"],
    }


def qualification_summary(
    paths: list[Path],
    *,
    advisor: JassAdvisor,
) -> list[dict[str, Any]]:
    """Load reports only when they describe this exact advisor policy."""

    expected_model_sha = advisor.policy.model_sha256
    if not expected_model_sha:
        raise ValueError("qualification evidence requires a hashed advisor model")
    if not advisor.manifest_sha256:
        raise ValueError("qualification evidence requires a hashed run manifest")
    expected_profile = advisor.profile.to_dict()
    expected_search = asdict(advisor.search.config)
    expected_guidance = asdict(advisor.policy.config)
    summaries = []
    for path in paths:
        payload = json.loads(path.read_text())
        candidate = payload.get("candidate_policy")
        model = payload.get("model")
        run = payload.get("run")
        compatibility = payload.get("compatibility")
        if (
            not isinstance(candidate, dict)
            or not isinstance(model, dict)
            or not isinstance(run, dict)
        ):
            raise ValueError(f"qualification report {path} lacks policy identity")
        mismatches = []
        if model.get("sha256") != expected_model_sha:
            mismatches.append("model SHA-256")
        if candidate.get("model_sha256") != expected_model_sha:
            mismatches.append("candidate model SHA-256")
        if candidate.get("type") != "NeuralGuidedPIMCPolicy":
            mismatches.append("candidate policy type")
        if candidate.get("implementation") != advisor.implementation_identity:
            mismatches.append("policy implementation identity")
        if candidate.get("search_profile") != expected_profile:
            mismatches.append("rules profile")
        if candidate.get("search_config") != expected_search:
            mismatches.append("search configuration")
        if candidate.get("guidance_config") != expected_guidance:
            mismatches.append("neural-guidance configuration")
        if candidate.get("public_information_only") is not True:
            mismatches.append("public-information boundary")
        if run.get("manifest_sha256") != advisor.manifest_sha256:
            mismatches.append("run manifest SHA-256")
        required_compatibility = (
            "actor_privacy_validated",
            "artifact_hash_validated",
            "manifest_validated",
        )
        if (
            not isinstance(compatibility, dict)
            or any(
                compatibility.get(key) is not True
                for key in required_compatibility
            )
            or compatibility.get("observation_schema_version") != 2
            or compatibility.get("action_count") != 45
        ):
            mismatches.append("artifact/privacy compatibility")
        if mismatches:
            raise ValueError(
                f"qualification report {path} does not match the loaded advisor: "
                + ", ".join(mismatches)
            )
        provenance = payload.get("evaluation_provenance", {}).get("git", {})
        if payload.get("report_version") != 2 or provenance.get("dirty") is not False:
            raise ValueError(
                f"qualification report {path} must have clean report-v2 provenance"
            )
        opponents = payload.get("opponents", {})
        if len(opponents) != 1:
            raise ValueError(f"qualification report {path} must contain one opponent")
        opponent, report = next(iter(opponents.items()))
        expected_protocol = (
            "verardo-v1-matched"
            if opponent in {"verardo", "verardo-random"}
            else "manifest-exact"
        )
        evaluation = payload.get("evaluation", {})
        report_environment = report.get("environment", {})
        manifest_environment = evaluation.get("manifest_environment", {})
        if (
            report.get("protocol") != expected_protocol
            or evaluation.get("paired_same_deal") is not True
            or evaluation.get("team_swap") is not True
            or evaluation.get("starters") != [0, 1, 2, 3]
            or report_environment.get("profile") != advisor.profile.to_dict()
            or report_environment.get("enable_bidding")
            != advisor.bidding_enabled
            or report_environment.get("enable_weis")
            != advisor.announcement_enabled
            or report_environment.get("enable_stock") != advisor.stock_enabled
            or manifest_environment.get("profile") != advisor.profile.to_dict()
            or manifest_environment.get("enable_bidding")
            != advisor.bidding_enabled
            or manifest_environment.get("enable_weis")
            != advisor.announcement_enabled
            or manifest_environment.get("enable_stock") != advisor.stock_enabled
        ):
            raise ValueError(
                f"qualification report {path} has incompatible evaluation protocol"
            )
        overall = report["metrics"]["overall"]
        qualification = report["qualification"]
        _validate_qualification_gate(path, opponent, report, qualification)
        summaries.append(
            {
                "opponent": opponent,
                "games": overall["episodes"],
                "pairs": overall["pairs"],
                "win_rate": overall["win_rate"],
                "win_rate_low": overall["win_rate_ci95"]["low"],
                "paired_win_rate": overall["paired_win_rate"],
                "paired_win_rate_low": overall["paired_win_rate_ci95"]["low"],
                "point_difference": overall["average_point_difference"],
                "thresholds": qualification["thresholds"],
                "checks": qualification["checks"],
                "qualified": qualification["qualified"],
                "policy_matched": True,
                "report": path.name,
            }
        )
    return summaries


def make_handler(
    advisor: JassAdvisor,
    *,
    reports: list[dict[str, Any]],
    ui_path: Path = UI_PATH,
) -> type[BaseHTTPRequestHandler]:
    if not ui_path.is_file():
        raise FileNotFoundError(ui_path)
    lock = threading.Lock()
    if any(report.get("policy_matched") is not True for report in reports):
        raise ValueError("qualification evidence must be bound to the loaded advisor")
    status = {
        "model_sha256": advisor.policy.model_sha256,
        "manifest_sha256": advisor.manifest_sha256,
        "policy_implementation": advisor.implementation_identity,
        "profile": advisor.profile.version,
        "search": {
            "determinizations": advisor.search.config.determinizations,
            "max_rollouts": advisor.search.config.max_rollouts,
            "opponent_rollout_policy": advisor.search.config.opponent_rollout_policy,
        },
        "guidance": asdict(advisor.policy.config),
        "environment": {
            "bidding_enabled": advisor.bidding_enabled,
            "announcement_enabled": advisor.announcement_enabled,
            "stock_enabled": advisor.stock_enabled,
            "trump_only_bidding": advisor.trump_only_bidding,
            "fixed_mode": advisor.fixed_mode,
            "fixed_trump_suit": advisor.fixed_trump_suit,
        },
        "qualification": reports,
    }

    class AdvisorHandler(BaseHTTPRequestHandler):
        server_version = "JassAdvisor/1"

        def _json(self, payload: Any, status_code: int = 200) -> None:
            encoded = json.dumps(
                payload,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/":
                encoded = ui_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if self.path == "/api/status":
                self._json(status)
                return
            if self.path == "/api/example":
                self._json(example_state(advisor))
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/api/advice":
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                raw_length = self.headers.get("Content-Length")
                length = int(raw_length) if raw_length is not None else 0
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                payload = json.loads(self.rfile.read(length))
                state = AdvisorState.from_dict(payload)
                with lock:
                    advice = advisor.advise(state).to_dict()
                self._json(advice)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception:
                LOGGER.exception("unexpected advisor failure")
                self._json(
                    {"error": "advisor inference failed"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    return AdvisorHandler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local Jass advisor")
    parser.add_argument("model", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=24)
    parser.add_argument("--max-rollouts", type=int, default=216)
    parser.add_argument(
        "--opponent-rollout-policy",
        choices=("generic", "random", "verardo"),
        default="generic",
    )
    parser.add_argument("--infer-opponent-bids", action="store_true")
    parser.add_argument("--terminal-win-bonus", type=float, default=0.0)
    parser.add_argument("--max-search-gap", type=float, default=3.0)
    parser.add_argument("--min-neural-probability", type=float, default=0.5)
    parser.add_argument("--report", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        advisor = load_advisor(
            args.model,
            device=args.device,
            seed=args.seed,
            search_config=PIMCConfig(
                determinizations=args.determinizations,
                max_rollouts=args.max_rollouts,
                common_random_numbers=True,
                opponent_rollout_policy=args.opponent_rollout_policy,
                infer_opponent_bids=args.infer_opponent_bids,
                terminal_win_bonus=args.terminal_win_bonus,
            ),
            guidance_config=NeuralGuidanceConfig(
                max_search_gap_raw_points=args.max_search_gap,
                min_neural_probability=args.min_neural_probability,
            ),
        )
        reports = qualification_summary(args.report, advisor=advisor)
        handler = make_handler(advisor, reports=reports)
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except (FileNotFoundError, ImportError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Jass advisor listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "MAX_REQUEST_BYTES",
    "UI_PATH",
    "example_state",
    "main",
    "make_handler",
    "qualification_summary",
]
