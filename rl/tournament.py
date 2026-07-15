"""Paired, seat-rotated tournament evaluation for public-information policies.

Unlike :class:`rl.single_agent_env.JassTeamEnv`, the runner assigns callbacks
directly to both AEC teams.  This makes a real team swap possible: on the second
game of a pair, the candidate and reference callbacks exchange Team A/Team B
while the deal seed, cards, starter, mode, and trump stay unchanged.
"""

from __future__ import annotations

import math
import operator
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from core.cards import MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS
from core.ruleset import RulesetConfig
from env.jass_aec_env import OBS_SIZE, JassAECEnv
from rl.baselines import PublicPolicy
from rl.privileged_observation import CTDE_OBS_SIZE, pad_public_observation

VALID_MODES = (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE)


@dataclass(frozen=True)
class TournamentConfig:
    """Configuration for a balanced tournament.

    ``episodes`` is the total number of games, not the number of deal pairs.
    It must contain complete starter rotations and, when ``swap_teams`` is true,
    both sides of every same-deal pair.
    """

    episodes: int
    seed: int = 0
    modes: tuple[str | None, ...] = VALID_MODES
    trump_suits: tuple[str, ...] = SUITS
    starters: tuple[int, ...] = (0, 1, 2, 3)
    swap_teams: bool = True
    enable_bidding: bool = False
    enable_weis: bool = True
    enable_stock: bool = True
    profile: RulesetConfig | None = None

    def validate(self) -> None:
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int):
            raise ValueError("episodes must be an integer")
        if self.episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        if not self.starters:
            raise ValueError("starters must not be empty")
        if len(set(self.starters)) != len(self.starters):
            raise ValueError("starters must be unique")
        if any(starter not in range(4) for starter in self.starters):
            raise ValueError("starters must contain only seats 0..3")
        if not self.modes:
            raise ValueError("modes must not be empty")

        if self.enable_bidding:
            if self.modes != (None,):
                raise ValueError("bidding tournaments must use modes=(None,)")
        elif any(mode not in VALID_MODES for mode in self.modes):
            raise ValueError(f"modes must be selected from {VALID_MODES}")

        if MODE_TRUMP in self.modes:
            if not self.trump_suits:
                raise ValueError("trump_suits must not be empty when trump is evaluated")
            if any(suit not in SUITS for suit in self.trump_suits):
                raise ValueError(f"trump_suits must be selected from {SUITS}")

        games_per_rotation = len(self.starters) * (2 if self.swap_teams else 1)
        if self.episodes % games_per_rotation:
            raise ValueError(
                f"episodes must be divisible by the complete rotation size ({games_per_rotation})"
            )


@dataclass(frozen=True)
class EpisodeSpec:
    episode: int
    deal_index: int
    pair_id: int
    deal_seed: int
    policy_seed: int
    starter: int
    candidate_team: int
    mode: str | None
    trump_suit: str | None


@dataclass(frozen=True)
class EpisodeResult:
    spec: EpisodeSpec
    mode: str
    trump_suit: str | None
    team_points: tuple[int, int]
    match_team: int | None = None
    candidate_decisions: int = 0
    candidate_inference_seconds: float = 0.0
    reference_decisions: int = 0
    reference_inference_seconds: float = 0.0

    @property
    def candidate_points(self) -> int:
        return self.team_points[self.spec.candidate_team]

    @property
    def reference_points(self) -> int:
        return self.team_points[1 - self.spec.candidate_team]

    @property
    def point_difference(self) -> int:
        return self.candidate_points - self.reference_points


@dataclass(frozen=True)
class WilsonInterval:
    low: float
    high: float


@dataclass(frozen=True)
class AggregateMetrics:
    episodes: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    score_rate: float
    win_rate_ci95: WilsonInterval
    average_candidate_points: float
    average_reference_points: float
    average_point_difference: float
    pairs: int
    paired_wins: int
    paired_losses: int
    paired_ties: int
    paired_win_rate: float
    paired_win_rate_ci95: WilsonInterval
    candidate_matches: int = 0
    reference_matches: int = 0
    match_rate: float = 0.0
    candidate_decisions: int = 0
    candidate_inference_seconds: float = 0.0
    mean_candidate_inference_ms: float = 0.0


@dataclass(frozen=True)
class TournamentReport:
    overall: AggregateMetrics
    by_mode: dict[str, AggregateMetrics]
    results: tuple[EpisodeResult, ...]
    by_contract: dict[str, AggregateMetrics] = field(default_factory=dict)
    team_swap_supported: bool = True
    team_swap_method: str = "direct AEC callback assignment"

    def to_dict(self, include_results: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "overall": asdict(self.overall),
            "by_mode": {mode: asdict(metrics) for mode, metrics in self.by_mode.items()},
            "by_contract": {
                contract: asdict(metrics)
                for contract, metrics in self.by_contract.items()
            },
            "team_swap_supported": self.team_swap_supported,
            "team_swap_method": self.team_swap_method,
        }
        if include_results:
            payload["results"] = [asdict(result) for result in self.results]
        return payload


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> WilsonInterval:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise ValueError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a positive finite number")
    if trials == 0:
        return WilsonInterval(0.0, 1.0)

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z_squared / (4.0 * trials * trials))
        / denominator
    )
    return WilsonInterval(max(0.0, centre - margin), min(1.0, centre + margin))


def build_schedule(config: TournamentConfig) -> tuple[EpisodeSpec, ...]:
    """Build complete same-deal team pairs and starter rotations."""

    config.validate()
    teams = (0, 1) if config.swap_teams else (0,)
    games_per_rotation = len(config.starters) * len(teams)
    rotations = config.episodes // games_per_rotation
    pair_count = rotations * len(config.starters)

    schedule: list[EpisodeSpec] = []
    episode = 0
    for pair_id in range(pair_count):
        # Every duplicate pair gets an independent shuffle.  Starters are
        # balanced by cycling across pairs instead of reusing one shuffled deal
        # for four different leaders, which keeps pair-level statistics honest.
        deal_index = pair_id
        starter = config.starters[pair_id % len(config.starters)]
        mode = config.modes[pair_id % len(config.modes)]
        trump_suit: str | None = None
        if mode == MODE_TRUMP:
            trump_round = pair_id // len(config.modes)
            trump_suit = config.trump_suits[trump_round % len(config.trump_suits)]
        deal_seed = config.seed + pair_id

        # Both sides of a pair receive the same stochastic-policy reset seed.
        policy_seed = _mix_seed(config.seed, pair_id)
        for candidate_team in teams:
            schedule.append(
                EpisodeSpec(
                    episode=episode,
                    deal_index=deal_index,
                    pair_id=pair_id,
                    deal_seed=deal_seed,
                    policy_seed=policy_seed,
                    starter=starter,
                    candidate_team=candidate_team,
                    mode=mode,
                    trump_suit=trump_suit,
                )
            )
            episode += 1
    return tuple(schedule)


def _mix_seed(base_seed: int, pair_id: int) -> int:
    # Stable across processes and Python versions, unlike hash().
    return (base_seed * 6_364_136_223_846_793_005 + pair_id * 1_442_695_040_888_963_407) & (
        (1 << 63) - 1
    )


def _reset_policy(policy: PublicPolicy, seed: int) -> None:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset(seed)


def _action_index(raw_action: object) -> int:
    if isinstance(raw_action, (bool, np.bool_)):
        raise TypeError("boolean is not a valid action")
    try:
        return operator.index(raw_action)
    except TypeError:
        encoded = np.asarray(raw_action)
        if encoded.size != 1:
            raise TypeError("action must contain exactly one integer") from None
        value = encoded.reshape(-1)[0]
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("boolean is not a valid action") from None
        return operator.index(value)


def _policy_action(
    policy: PublicPolicy, observation: np.ndarray, action_mask: np.ndarray, agent: str
) -> int:
    raw_action = policy(observation, action_mask, agent)
    try:
        action = _action_index(raw_action)
    except TypeError as exc:
        raise ValueError(f"policy for {agent} returned a non-integer action") from exc
    if action < 0 or action >= action_mask.size or not bool(action_mask[action]):
        legal = [int(value) for value in np.flatnonzero(action_mask)]
        raise ValueError(f"policy for {agent} returned illegal action {action}; legal={legal}")
    return action


def run_episode(
    spec: EpisodeSpec,
    candidate: PublicPolicy,
    reference: PublicPolicy,
    config: TournamentConfig,
) -> EpisodeResult:
    """Run one scheduled game without exposing the AEC environment to policies."""

    config.validate()
    _reset_policy(candidate, spec.policy_seed)
    _reset_policy(reference, spec.policy_seed ^ 0x5DEECE66D)

    env = JassAECEnv(
        seed=spec.deal_seed,
        profile=config.profile,
        enable_bidding=config.enable_bidding,
        enable_weis=config.enable_weis,
        enable_stock=config.enable_stock,
        mode=spec.mode,
        trump_suit=spec.trump_suit,
        starter=spec.starter,
    )
    env.reset(seed=spec.deal_seed)
    candidate_decisions = 0
    candidate_inference_seconds = 0.0
    reference_decisions = 0
    reference_inference_seconds = 0.0
    try:
        while env.agents and not all(env.terminations.values()):
            agent = env.agent_selection
            if env.terminations.get(agent) or env.truncations.get(agent):
                env.step(None)
                continue

            visible = env.observe(agent)
            team = int(agent.removeprefix("p")) % 2
            policy = candidate if team == spec.candidate_team else reference
            started = time.perf_counter()
            action = _policy_action(policy, visible["observation"], visible["action_mask"], agent)
            elapsed = time.perf_counter() - started
            if team == spec.candidate_team:
                candidate_decisions += 1
                candidate_inference_seconds += elapsed
            else:
                reference_decisions += 1
                reference_inference_seconds += elapsed
            env.step(action)

        if env.state is None or env.mode is None:
            raise RuntimeError("episode ended without an initialized game state")
        return EpisodeResult(
            spec=spec,
            mode=env.mode,
            trump_suit=env.trump_suit,
            team_points=(int(env.state.team_points[0]), int(env.state.team_points[1])),
            match_team=env.match_team,
            candidate_decisions=candidate_decisions,
            candidate_inference_seconds=candidate_inference_seconds,
            reference_decisions=reference_decisions,
            reference_inference_seconds=reference_inference_seconds,
        )
    finally:
        env.close()


EpisodeRunner = Callable[[EpisodeSpec, PublicPolicy, PublicPolicy, TournamentConfig], EpisodeResult]


def run_tournament(
    candidate: PublicPolicy,
    reference: PublicPolicy,
    config: TournamentConfig,
    *,
    runner: EpisodeRunner = run_episode,
) -> TournamentReport:
    """Evaluate arbitrary public-observation callbacks on a balanced schedule."""

    schedule = build_schedule(config)
    results = tuple(runner(spec, candidate, reference, config) for spec in schedule)
    if len(results) != config.episodes:
        raise RuntimeError(f"runner produced {len(results)} results for {config.episodes} episodes")
    return summarize_results(results)


def _paired_outcomes(results: Sequence[EpisodeResult]) -> tuple[int, int, int, int]:
    grouped: dict[int, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        grouped[result.spec.pair_id].append(result)

    wins = losses = ties = pairs = 0
    for pair in grouped.values():
        if len(pair) != 2 or {result.spec.candidate_team for result in pair} != {0, 1}:
            continue
        deal_keys = {
            (
                result.spec.deal_index,
                result.spec.deal_seed,
                result.spec.policy_seed,
                result.spec.starter,
                result.spec.mode,
                result.spec.trump_suit,
            )
            for result in pair
        }
        if len(deal_keys) != 1:
            continue
        pairs += 1
        difference = sum(result.point_difference for result in pair)
        if difference > 0:
            wins += 1
        elif difference < 0:
            losses += 1
        else:
            ties += 1
    return pairs, wins, losses, ties


def aggregate_metrics(results: Sequence[EpisodeResult]) -> AggregateMetrics:
    """Aggregate game and complete-pair outcomes.

    The Wilson interval is for strict wins divided by all games (or all complete
    pairs); ties are not counted as successes.  ``score_rate`` separately gives
    the conventional win + half-tie rate.
    """

    if not results:
        raise ValueError("at least one episode result is required")

    wins = sum(result.point_difference > 0 for result in results)
    losses = sum(result.point_difference < 0 for result in results)
    ties = len(results) - wins - losses
    episodes = len(results)
    pairs, paired_wins, paired_losses, paired_ties = _paired_outcomes(results)
    candidate_matches = sum(
        result.match_team == result.spec.candidate_team for result in results
    )
    reference_matches = sum(
        result.match_team == 1 - result.spec.candidate_team for result in results
    )
    candidate_decisions = sum(result.candidate_decisions for result in results)
    candidate_inference_seconds = sum(
        result.candidate_inference_seconds for result in results
    )

    return AggregateMetrics(
        episodes=episodes,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=wins / episodes,
        score_rate=(wins + 0.5 * ties) / episodes,
        win_rate_ci95=wilson_interval(wins, episodes),
        average_candidate_points=sum(result.candidate_points for result in results) / episodes,
        average_reference_points=sum(result.reference_points for result in results) / episodes,
        average_point_difference=sum(result.point_difference for result in results) / episodes,
        pairs=pairs,
        paired_wins=paired_wins,
        paired_losses=paired_losses,
        paired_ties=paired_ties,
        paired_win_rate=paired_wins / pairs if pairs else 0.0,
        paired_win_rate_ci95=wilson_interval(paired_wins, pairs),
        candidate_matches=candidate_matches,
        reference_matches=reference_matches,
        match_rate=candidate_matches / episodes,
        candidate_decisions=candidate_decisions,
        candidate_inference_seconds=candidate_inference_seconds,
        mean_candidate_inference_ms=(
            candidate_inference_seconds * 1000.0 / candidate_decisions
            if candidate_decisions
            else 0.0
        ),
    )


def summarize_results(results: Sequence[EpisodeResult]) -> TournamentReport:
    """Produce overall and actual-mode metrics from episode results."""

    if not results:
        raise ValueError("at least one episode result is required")
    by_mode_results: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        by_mode_results[result.mode].append(result)
    by_mode = {
        mode: aggregate_metrics(mode_results)
        for mode, mode_results in sorted(by_mode_results.items())
    }
    by_contract_results: dict[str, list[EpisodeResult]] = defaultdict(list)
    for result in results:
        contract = (
            f"{result.mode}:{result.trump_suit}"
            if result.mode == MODE_TRUMP
            else result.mode
        )
        by_contract_results[contract].append(result)
    by_contract = {
        contract: aggregate_metrics(contract_results)
        for contract, contract_results in sorted(by_contract_results.items())
    }
    return TournamentReport(
        overall=aggregate_metrics(results),
        by_mode=by_mode,
        results=tuple(results),
        by_contract=by_contract,
    )


def maskable_model_policy(
    model: object,
    *,
    deterministic: bool = True,
) -> PublicPolicy:
    """Adapt any SB3-style ``predict`` model without importing SB3.

    Canonical observations are already acting-seat-relative, so the exact same
    model input shape is valid for both partners and both sides of a paired
    match.
    """

    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise TypeError("model must provide a callable predict method")
    model_shape = tuple(getattr(getattr(model, "observation_space", None), "shape", ()))
    if model_shape not in {(OBS_SIZE,), (CTDE_OBS_SIZE,)}:
        raise ValueError(
            f"model observation shape {model_shape} is neither canonical public "
            f"({OBS_SIZE},) nor CTDE ({CTDE_OBS_SIZE},)"
        )

    def _policy(observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int:
        del agent
        model_observation = (
            pad_public_observation(observation)
            if model_shape == (CTDE_OBS_SIZE,)
            else observation
        )
        action, _ = predict(
            model_observation,
            action_masks=action_mask,
            deterministic=deterministic,
        )
        return _action_index(action)

    return _policy


__all__ = [
    "AggregateMetrics",
    "EpisodeResult",
    "EpisodeSpec",
    "TournamentConfig",
    "TournamentReport",
    "WilsonInterval",
    "aggregate_metrics",
    "build_schedule",
    "maskable_model_policy",
    "run_episode",
    "run_tournament",
    "summarize_results",
    "wilson_interval",
]
