import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.ruleset import STANDARD_RULES_PROFILE
from rl.advisor import AdvisorState
from rl.advisor_web import UI_PATH, example_state, make_handler, qualification_summary
from rl.eval import QUALIFICATION_GATES
from rl.hybrid_policy import NeuralGuidanceConfig
from rl.search_policy import PIMCConfig, policy_implementation_identity


class _Advice:
    def to_dict(self):
        return {"selected_action": 1, "selected_card": "schellen:7", "actions": []}


class _Advisor:
    profile = STANDARD_RULES_PROFILE
    policy = SimpleNamespace(
        model_sha256="abc",
        config=NeuralGuidanceConfig(),
    )
    search = SimpleNamespace(
        config=PIMCConfig(
            determinizations=2,
            max_rollouts=18,
            common_random_numbers=True,
        )
    )
    bidding_enabled = True
    announcement_enabled = True
    stock_enabled = True
    trump_only_bidding = False
    fixed_mode = None
    fixed_trump_suit = None
    manifest_sha256 = "def"
    implementation_identity = policy_implementation_identity()

    def advise(self, state):
        assert isinstance(state, AdvisorState)
        return _Advice()


def test_example_and_packaged_ui_are_valid() -> None:
    AdvisorState.from_dict(example_state())
    html = UI_PATH.read_text()
    assert 'id="deck"' in html
    assert 'id="ranking"' in html
    assert "/api/advice" in html
    assert "http://" not in html and "https://" not in html


def test_qualification_summary_extracts_gate_evidence(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "report_version": 2,
                "evaluation_provenance": {"git": {"dirty": False}},
                "model": {"sha256": "abc"},
                "run": {"manifest_sha256": "def"},
                "candidate_policy": {
                    "type": "NeuralGuidedPIMCPolicy",
                    "implementation": _Advisor.implementation_identity,
                    "model_sha256": "abc",
                    "public_information_only": True,
                    "search_profile": STANDARD_RULES_PROFILE.to_dict(),
                    "search_config": asdict(_Advisor.search.config),
                    "guidance_config": asdict(_Advisor.policy.config),
                },
                "compatibility": {
                    "actor_privacy_validated": True,
                    "artifact_hash_validated": True,
                    "manifest_validated": True,
                    "observation_schema_version": 2,
                    "action_count": 45,
                },
                "evaluation": {
                    "paired_same_deal": True,
                    "team_swap": True,
                    "starters": [0, 1, 2, 3],
                    "manifest_environment": {
                        "profile": STANDARD_RULES_PROFILE.to_dict(),
                        "enable_bidding": True,
                        "enable_weis": True,
                        "enable_stock": True,
                    },
                },
                "opponents": {
                    "random": {
                        "protocol": "manifest-exact",
                        "environment": {
                            "profile": STANDARD_RULES_PROFILE.to_dict(),
                            "enable_bidding": True,
                            "enable_weis": True,
                            "enable_stock": True,
                        },
                        "metrics": {
                            "overall": {
                                "episodes": 4000,
                                "pairs": 2000,
                                "win_rate": 0.8,
                                "win_rate_ci95": {"low": 0.78},
                                "paired_win_rate": 0.9,
                                "paired_win_rate_ci95": {"low": 0.88},
                                "average_point_difference": 42.0,
                            }
                        },
                        "qualification": {
                            "thresholds": dict(QUALIFICATION_GATES["random"]),
                            "checks": {
                                "manifest_provenance": True,
                                "protocol_environment": True,
                                "minimum_games": True,
                                "minimum_pairs": True,
                                "complete_pairing": True,
                                "minimum_win_rate": True,
                                "minimum_wilson_low": True,
                                "minimum_paired_win_rate": True,
                                "minimum_paired_wilson_low": True,
                                "minimum_average_point_difference": True,
                            },
                            "performance_qualified": True,
                            "qualified": True,
                        },
                    }
                }
            }
        )
    )

    summary = qualification_summary([path], advisor=_Advisor())  # type: ignore[arg-type]

    assert summary[0]["opponent"] == "random"
    assert summary[0]["qualified"] is True
    assert summary[0]["games"] == 4000
    assert summary[0]["policy_matched"] is True
    assert summary[0]["report"] == "report.json"

    tampered = json.loads(path.read_text())
    tampered["candidate_policy"]["implementation"]["sha256"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="policy implementation identity"):
        qualification_summary([path], advisor=_Advisor())  # type: ignore[arg-type]

    tampered = json.loads(path.read_text())
    tampered["candidate_policy"]["implementation"] = _Advisor.implementation_identity
    tampered["opponents"]["random"]["metrics"]["overall"]["win_rate"] = 0.1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="inconsistent gate checks"):
        qualification_summary([path], advisor=_Advisor())  # type: ignore[arg-type]


def test_qualification_summary_rejects_a_different_model(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "run": {"manifest_sha256": "def"},
                "model": {"sha256": "different"},
                "candidate_policy": {
                    "type": "NeuralGuidedPIMCPolicy",
                    "implementation": _Advisor.implementation_identity,
                    "model_sha256": "different",
                    "public_information_only": True,
                    "search_profile": STANDARD_RULES_PROFILE.to_dict(),
                    "search_config": asdict(_Advisor.search.config),
                    "guidance_config": asdict(_Advisor.policy.config),
                },
                "compatibility": {
                    "actor_privacy_validated": True,
                    "artifact_hash_validated": True,
                    "manifest_validated": True,
                    "observation_schema_version": 2,
                    "action_count": 45,
                },
                "opponents": {},
            }
        )
    )

    with pytest.raises(ValueError, match="model SHA-256"):
        qualification_summary([path], advisor=_Advisor())  # type: ignore[arg-type]


def test_handler_rejects_unbound_qualification_evidence() -> None:
    with pytest.raises(ValueError, match="bound"):
        make_handler(_Advisor(), reports=[{"qualified": True}])  # type: ignore[arg-type]


def test_local_server_exposes_status_example_and_advice() -> None:
    from http.server import ThreadingHTTPServer

    handler = make_handler(_Advisor(), reports=[])  # type: ignore[arg-type]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/status") as response:
            status = json.load(response)
        assert status["model_sha256"] == "abc"
        assert status["environment"] == {
            "bidding_enabled": True,
            "announcement_enabled": True,
            "stock_enabled": True,
            "trump_only_bidding": False,
            "fixed_mode": None,
            "fixed_trump_suit": None,
        }

        request = urllib.request.Request(
            base + "/api/advice",
            data=json.dumps(example_state()).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            advice = json.load(response)
        assert advice["selected_card"] == "schellen:7"

        bad = urllib.request.Request(
            base + "/api/advice",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(bad)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:
            pytest.fail("malformed advisor state unexpectedly succeeded")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
