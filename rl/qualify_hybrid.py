"""Manifest-aware qualification CLI for the neural-guided PIMC candidate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from rl.eval import EvalConfig, EvaluationContext, evaluate
from rl.hybrid_policy import NeuralGuidanceConfig, NeuralGuidedPIMCPolicy
from rl.reference_verardo import VERARDO_V1_PROFILE
from rl.run_manifest import write_manifest
from rl.search_policy import (
    PIMCConfig,
    PIMCSearchPolicy,
    policy_implementation_identity,
)

FULL_OPPONENTS = ("random", "strategic")
EXTERNAL_OPPONENTS = ("verardo-random", "verardo")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify a manifest-backed model with public neural-guided PIMC"
    )
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--benchmark", choices=("full", "external"), required=True)
    parser.add_argument("--opponents", nargs="+")
    parser.add_argument("--episodes", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-results", action="store_true")
    parser.add_argument("--determinizations", type=int, default=24)
    parser.add_argument("--max-rollouts", type=int, default=216)
    parser.add_argument("--max-rollout-plies", type=int, default=36)
    parser.add_argument("--assignment-node-limit", type=int, default=8_000)
    parser.add_argument("--rollout-randomness", type=float, default=0.04)
    parser.add_argument("--terminal-win-bonus", type=float, default=0.0)
    parser.add_argument(
        "--common-random-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--opponent-rollout-policy",
        choices=("generic", "random", "verardo"),
    )
    parser.add_argument("--infer-opponent-bids", action="store_true")
    parser.add_argument("--prune-bid-constraints", action="store_true")
    parser.add_argument("--max-search-gap", type=float, default=3.0)
    parser.add_argument("--min-neural-probability", type=float, default=0.5)
    parser.add_argument("--neural-bidding", action="store_true")
    return parser.parse_args(argv)


def _resolved_opponents(benchmark: str, requested: list[str] | None) -> tuple[str, ...]:
    allowed = FULL_OPPONENTS if benchmark == "full" else EXTERNAL_OPPONENTS
    opponents = tuple(requested) if requested else allowed
    unknown = sorted(set(opponents) - set(allowed))
    if unknown:
        raise ValueError(f"{benchmark} benchmark does not support opponents {unknown}")
    if len(set(opponents)) != len(opponents):
        raise ValueError("opponents must not contain duplicates")
    return opponents


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        opponents = _resolved_opponents(args.benchmark, args.opponents)
        rollout_policy = args.opponent_rollout_policy or (
            "verardo" if args.benchmark == "external" else "generic"
        )
        if args.infer_opponent_bids and "verardo-random" in opponents:
            raise ValueError(
                "bid inference assumes the deterministic reference bidder; run the "
                "external random and reference opponents as separate reports"
            )
        search_config = PIMCConfig(
            determinizations=args.determinizations,
            max_rollouts=args.max_rollouts,
            max_rollout_plies=args.max_rollout_plies,
            assignment_node_limit=args.assignment_node_limit,
            rollout_randomness=args.rollout_randomness,
            terminal_win_bonus=args.terminal_win_bonus,
            common_random_numbers=args.common_random_numbers,
            opponent_rollout_policy=rollout_policy,
            infer_opponent_bids=args.infer_opponent_bids,
            prune_bid_constraints=args.prune_bid_constraints,
        )
        guidance_config = NeuralGuidanceConfig(
            max_search_gap_raw_points=args.max_search_gap,
            min_neural_probability=args.min_neural_probability,
            use_neural_bidding=args.neural_bidding,
        )
        search_config.validate()
        guidance_config.validate()
        output = args.output or (
            args.model_path.parent
            / (f"hybrid_evaluation_{args.benchmark}_{'-'.join(opponents)}_seed{args.seed}.json")
        )
        holder: dict[str, NeuralGuidedPIMCPolicy] = {}

        def build_candidate(model: object, context: EvaluationContext):
            profile = (
                VERARDO_V1_PROFILE if args.benchmark == "external" else context.environment.profile
            )
            candidate = NeuralGuidedPIMCPolicy(
                model,
                PIMCSearchPolicy(
                    seed=args.seed,
                    config=search_config,
                    profile=profile,
                ),
                config=guidance_config,
                model_path=args.model_path,
            )
            holder["candidate"] = candidate
            return candidate

        report = evaluate(
            EvalConfig(
                model_path=args.model_path,
                episodes=args.episodes,
                seed=args.seed,
                opponents=opponents,
                output_path=output,
                include_results=args.include_results,
                device=args.device,
            ),
            candidate_builder=build_candidate,
        )
        candidate = holder["candidate"]
        report["candidate_policy"] = {
            "type": "NeuralGuidedPIMCPolicy",
            "implementation": policy_implementation_identity(),
            "public_information_only": True,
            "model_sha256": candidate.model_sha256,
            "search_config": asdict(search_config),
            "guidance_config": asdict(guidance_config),
            "search_profile": candidate.search.profile.to_dict(),
            "stats": asdict(candidate.stats),
            "last_decision": (
                asdict(candidate.last_decision) if candidate.last_decision is not None else None
            ),
        }
        write_manifest(output, report)
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EXTERNAL_OPPONENTS", "FULL_OPPONENTS", "main"]
