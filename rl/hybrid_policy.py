"""Public-only neural guidance for bounded PIMC search.

The neural actor is allowed to change a search decision only when its masked
probability is sufficiently high and its action lies within a configured raw
point gap of the search optimum.  Both components receive the same canonical
public observation; CTDE models are supplied only a zero-padded private suffix.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from env.jass_aec_env import (
    ACTION_COUNT,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    BIDDING_TRUMP_ACTIONS,
    BIDDING_UNEUFE_ACTION,
    OBS_CARD_COUNT,
    OBS_SIZE,
)
from rl.privileged_observation import CTDE_OBS_SIZE, pad_public_observation
from rl.search_policy import PIMCSearchPolicy


@dataclass(frozen=True)
class NeuralGuidanceConfig:
    """Conservative conditions under which the learned actor may override search."""

    max_search_gap_raw_points: float = 3.0
    min_neural_probability: float = 0.5
    use_neural_bidding: bool = False

    def validate(self) -> None:
        if (
            not math.isfinite(self.max_search_gap_raw_points)
            or self.max_search_gap_raw_points < 0.0
        ):
            raise ValueError("max_search_gap_raw_points must be non-negative and finite")
        if (
            not math.isfinite(self.min_neural_probability)
            or not 0.0 <= self.min_neural_probability <= 1.0
        ):
            raise ValueError("min_neural_probability must be between 0 and 1")
        if not isinstance(self.use_neural_bidding, bool):
            raise ValueError("use_neural_bidding must be a boolean")


@dataclass(frozen=True)
class HybridDecision:
    selected_action: int
    search_action: int
    neural_action: int | None
    neural_probability: float | None
    search_gap_raw_points: float | None
    overridden: bool
    reason: str


@dataclass(frozen=True)
class HybridStats:
    decisions: int = 0
    bidding_decisions: int = 0
    neural_bids: int = 0
    card_decisions: int = 0
    agreements: int = 0
    disagreements: int = 0
    confidence_eligible: int = 0
    gap_eligible: int = 0
    overrides: int = 0
    nonfinite_search_gaps: int = 0
    search_fallbacks: int = 0
    attempted_determinizations: int = 0
    successful_determinizations: int = 0
    search_rollouts: int = 0
    simulated_plies: int = 0
    assignment_nodes: int = 0


_BIDDING_ACTIONS = frozenset(BIDDING_TRUMP_ACTIONS) | {
    BIDDING_OBEABE_ACTION,
    BIDDING_UNEUFE_ACTION,
    BIDDING_PUSH_ACTION,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_shape(model: object) -> tuple[int, ...]:
    return tuple(getattr(getattr(model, "observation_space", None), "shape", ()))


def _masked_probabilities(
    model: object,
    observation: np.ndarray,
    action_mask: np.ndarray,
) -> np.ndarray:
    shape = _model_shape(model)
    if shape not in {(OBS_SIZE,), (CTDE_OBS_SIZE,)}:
        raise ValueError(
            f"model observation shape {shape} is neither public ({OBS_SIZE},) "
            f"nor CTDE ({CTDE_OBS_SIZE},)"
        )
    public = np.asarray(observation, dtype=np.float32).reshape(-1)
    if public.shape != (OBS_SIZE,):
        raise ValueError(f"public observation must have shape ({OBS_SIZE},)")
    mask = np.asarray(action_mask).reshape(-1)
    if mask.shape != (ACTION_COUNT,):
        raise ValueError(f"action mask must have shape ({ACTION_COUNT},)")

    policy = getattr(model, "policy", None)
    obs_to_tensor = getattr(policy, "obs_to_tensor", None)
    get_distribution = getattr(policy, "get_distribution", None)
    if not callable(obs_to_tensor) or not callable(get_distribution):
        raise TypeError("model.policy must provide obs_to_tensor and get_distribution")

    model_observation = pad_public_observation(public) if shape == (CTDE_OBS_SIZE,) else public
    tensor, _ = obs_to_tensor(model_observation)
    try:
        import torch

        no_grad = torch.no_grad()
    except ImportError:  # pragma: no cover - the RL dependency set includes torch
        no_grad = nullcontext()
    with no_grad:
        distribution = get_distribution(tensor, action_masks=mask)
    probabilities = getattr(getattr(distribution, "distribution", None), "probs", None)
    if probabilities is None:
        raise TypeError("model policy distribution must expose distribution.probs")
    if hasattr(probabilities, "detach"):
        probabilities = probabilities.detach()
    if hasattr(probabilities, "cpu"):
        probabilities = probabilities.cpu()
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("model returned malformed action probabilities")
    if np.any(values < -1e-9):
        raise ValueError("model returned negative action probabilities")
    if np.any(values[mask == 0] > 1e-6):
        raise ValueError("model assigned probability to an illegal action")
    values = np.where(mask == 1, np.maximum(values, 0.0), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("model assigned no probability to legal actions")
    return values / total


class NeuralGuidedPIMCPolicy:
    """Combine a trained public actor with auditable public-information search."""

    def __init__(
        self,
        model: object,
        search: PIMCSearchPolicy,
        *,
        config: NeuralGuidanceConfig | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.search = search
        self.config = config or NeuralGuidanceConfig()
        self.config.validate()
        if _model_shape(model) not in {(OBS_SIZE,), (CTDE_OBS_SIZE,)}:
            raise ValueError("hybrid model has an incompatible observation shape")
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        if self.model_path is not None and not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model_sha256 = _sha256(self.model_path) if self.model_path is not None else None
        self.stats = HybridStats()
        self.last_decision: HybridDecision | None = None

    def clear_stats(self) -> None:
        self.stats = HybridStats()
        self.last_decision = None

    def reset(self, seed: int | None = None) -> None:
        """Reset seeded search state while retaining tournament diagnostics."""

        self.search.reset(seed)
        self.last_decision = None

    def __call__(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        agent: str,
    ) -> int:
        search_action = self.search(observation, action_mask, agent)
        stats = replace(
            self.stats,
            decisions=self.stats.decisions + 1,
            search_fallbacks=(
                self.stats.search_fallbacks + int(self.search.last_stats.used_fallback)
            ),
            attempted_determinizations=(
                self.stats.attempted_determinizations
                + self.search.last_stats.attempted_determinizations
            ),
            successful_determinizations=(
                self.stats.successful_determinizations
                + self.search.last_stats.successful_determinizations
            ),
            search_rollouts=self.stats.search_rollouts + self.search.last_stats.rollouts,
            simulated_plies=(self.stats.simulated_plies + self.search.last_stats.simulated_plies),
            assignment_nodes=(
                self.stats.assignment_nodes + self.search.last_stats.assignment_nodes
            ),
        )
        legal = tuple(int(action) for action in np.flatnonzero(action_mask))
        if not legal or search_action not in legal:
            raise RuntimeError("search returned an action outside the legal mask")
        bidding_decision = all(action in _BIDDING_ACTIONS for action in legal)
        if self.config.use_neural_bidding and bidding_decision and len(legal) > 1:
            probabilities = _masked_probabilities(self.model, observation, action_mask)
            neural_action = max(legal, key=lambda action: (probabilities[action], -action))
            self.stats = replace(
                stats,
                bidding_decisions=stats.bidding_decisions + 1,
                neural_bids=stats.neural_bids + 1,
            )
            self.last_decision = HybridDecision(
                selected_action=neural_action,
                search_action=search_action,
                neural_action=neural_action,
                neural_probability=float(probabilities[neural_action]),
                search_gap_raw_points=None,
                overridden=neural_action != search_action,
                reason="learned public actor selected the bid",
            )
            return neural_action
        if any(action >= OBS_CARD_COUNT for action in legal) or len(legal) == 1:
            self.stats = stats
            self.last_decision = HybridDecision(
                selected_action=search_action,
                search_action=search_action,
                neural_action=None,
                neural_probability=None,
                search_gap_raw_points=None,
                overridden=False,
                reason="search-only non-card or forced decision",
            )
            return search_action

        probabilities = _masked_probabilities(self.model, observation, action_mask)
        neural_action = max(legal, key=lambda action: (probabilities[action], -action))
        neural_probability = float(probabilities[neural_action])
        stats = replace(stats, card_decisions=stats.card_decisions + 1)
        if neural_action == search_action:
            self.stats = replace(stats, agreements=stats.agreements + 1)
            self.last_decision = HybridDecision(
                selected_action=search_action,
                search_action=search_action,
                neural_action=neural_action,
                neural_probability=neural_probability,
                search_gap_raw_points=0.0,
                overridden=False,
                reason="neural and search agree",
            )
            return search_action

        stats = replace(stats, disagreements=stats.disagreements + 1)
        values = {action: value for action, value, _ in self.search.last_stats.action_values}
        search_value = values.get(search_action, float("nan"))
        neural_value = values.get(neural_action, float("nan"))
        gap = search_value - neural_value
        confidence_eligible = neural_probability >= self.config.min_neural_probability
        finite_gap = math.isfinite(gap)
        gap_eligible = finite_gap and gap <= self.config.max_search_gap_raw_points
        stats = replace(
            stats,
            confidence_eligible=stats.confidence_eligible + int(confidence_eligible),
            gap_eligible=stats.gap_eligible + int(gap_eligible),
            nonfinite_search_gaps=stats.nonfinite_search_gaps + int(not finite_gap),
        )
        overridden = confidence_eligible and gap_eligible
        selected = neural_action if overridden else search_action
        if overridden:
            stats = replace(stats, overrides=stats.overrides + 1)
        self.stats = stats
        self.last_decision = HybridDecision(
            selected_action=selected,
            search_action=search_action,
            neural_action=neural_action,
            neural_probability=neural_probability,
            search_gap_raw_points=float(gap) if finite_gap else None,
            overridden=overridden,
            reason=(
                "neural override" if overridden else "neural confidence or search-gap gate not met"
            ),
        )
        return selected


__all__ = [
    "HybridDecision",
    "HybridStats",
    "NeuralGuidanceConfig",
    "NeuralGuidedPIMCPolicy",
]
