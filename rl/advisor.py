"""Public-state input and ranked advice for a real Schieber game."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.cards import ALL_CARDS, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from core.legal_moves import legal_cards
from core.rankings import winning_card
from core.ruleset import RulesetConfig
from core.scoring import trick_points
from env.jass_aec_env import (
    ACTION_COUNT,
    OBS_ACTOR_OFFSET,
    OBS_ANNOUNCEMENT_CURRENT_OFFSET,
    OBS_ANNOUNCEMENT_ENABLED_OFFSET,
    OBS_ANNOUNCEMENT_STATUS_OFFSET,
    OBS_BID_CHOOSER_OFFSET,
    OBS_BID_PUSHED_OFFSET,
    OBS_BID_STARTER_OFFSET,
    OBS_BIDDING_ENABLED_OFFSET,
    OBS_CARD_COUNT,
    OBS_CONTRACT_FACTORS_OFFSET,
    OBS_HAND_COUNTS_OFFSET,
    OBS_HAND_OFFSET,
    OBS_HISTORY_CARDS_OFFSET,
    OBS_HISTORY_PLAYERS_OFFSET,
    OBS_LEADER_OFFSET,
    OBS_MATCH_BONUS_OFFSET,
    OBS_MODE_OFFSET,
    OBS_PHASE_OFFSET,
    OBS_PLAYER_COUNT,
    OBS_SIZE,
    OBS_TEAM_POINTS_OFFSET,
    OBS_TRICK_COMPLETE_OFFSET,
    OBS_TRICK_INDEX_OFFSET,
    OBS_TRICK_POINTS_OFFSET,
    OBS_TRICK_WINNER_OFFSET,
    OBS_TRUMP_SUIT_OFFSET,
    OBSERVATION_SCHEMA_VERSION,
)
from rl.eval import (
    EvaluationEnvironment,
    assert_model_belongs_to_run,
    assert_model_spaces,
    resolve_model_path,
)
from rl.hybrid_policy import (
    NeuralGuidanceConfig,
    NeuralGuidedPIMCPolicy,
    masked_action_probabilities,
)
from rl.run_manifest import (
    MANIFEST_FILENAME,
    assert_compatible,
    load_manifest,
    write_manifest,
)
from rl.search_policy import (
    PIMCConfig,
    PIMCSearchPolicy,
    policy_implementation_identity,
)

CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}
VALID_MODES = (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE)
ANNOUNCEMENT_STATES = {"undecided": 0, "pass": 1, "announce": 2}


def card_code(card: Card) -> str:
    return f"{card.suit}:{card.rank}"


def parse_card(value: str | Card) -> Card:
    if isinstance(value, Card):
        return value
    if not isinstance(value, str):
        raise TypeError("card must be a string or Card")
    if ":" in value:
        suit, rank = value.split(":", 1)
    elif "-" in value:
        suit, rank = value.rsplit("-", 1)
    else:
        raise ValueError(f"card {value!r} must use suit:rank")
    return Card(suit.strip().lower(), rank.strip().upper())


def _parse_play(value: Any) -> tuple[int, Card]:
    if isinstance(value, dict):
        player, card = value.get("player"), value.get("card")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        player, card = value
    else:
        raise ValueError("each play must contain player and card")
    if isinstance(player, bool) or not isinstance(player, int) or player not in range(4):
        raise ValueError("play player must be a relative seat from 0 to 3")
    return player, parse_card(card)


def _bounded(value: int | float, scale: float) -> float:
    number = max(0.0, float(value))
    return number / (number + scale) if number else 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdvisorState:
    """All information a player can observe at one card decision.

    Players are relative to the acting player: self=0, left=1, partner=2,
    right=3.  Team points are ``(our_team, opponents)``.
    """

    hand: tuple[Card, ...]
    completed_tricks: tuple[tuple[tuple[int, Card], ...], ...]
    current_trick: tuple[tuple[int, Card], ...]
    mode: str
    trump_suit: str | None
    leader: int
    team_points: tuple[int, int] = (0, 0)
    bidding_enabled: bool = True
    bid_starter: int | None = 0
    bid_chooser: int | None = 0
    bid_pushed: bool = False
    announcement_enabled: bool = True
    announcement_status: tuple[str, str, str, str] = (
        "pass",
        "pass",
        "pass",
        "pass",
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AdvisorState:
        if not isinstance(payload, dict):
            raise TypeError("advisor state must be a JSON object")
        completed = tuple(
            tuple(_parse_play(play) for play in trick)
            for trick in payload.get("completed_tricks", ())
        )
        current = tuple(_parse_play(play) for play in payload.get("current_trick", ()))
        points = payload.get("team_points", (0, 0))
        statuses = tuple(payload.get("announcement_status", ("pass",) * 4))
        state = cls(
            hand=tuple(parse_card(card) for card in payload.get("hand", ())),
            completed_tricks=completed,
            current_trick=current,
            mode=payload.get("mode", MODE_TRUMP),
            trump_suit=payload.get("trump_suit"),
            leader=payload.get("leader", 0),
            team_points=tuple(points),  # type: ignore[arg-type]
            bidding_enabled=payload.get("bidding_enabled", True),
            bid_starter=payload.get("bid_starter", 0),
            bid_chooser=payload.get("bid_chooser", 0),
            bid_pushed=payload.get("bid_pushed", False),
            announcement_enabled=payload.get("announcement_enabled", True),
            announcement_status=statuses,  # type: ignore[arg-type]
        )
        state.validate()
        return state

    @property
    def hand_counts(self) -> tuple[int, int, int, int]:
        played = [0, 0, 0, 0]
        for trick in (*self.completed_tricks, self.current_trick):
            for player, _ in trick:
                played[player] += 1
        return tuple(9 - count for count in played)  # type: ignore[return-value]

    def validate(self, profile: RulesetConfig | None = None) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        if self.mode == MODE_TRUMP and self.trump_suit not in SUITS:
            raise ValueError("a trump contract requires trump_suit")
        if self.mode != MODE_TRUMP and self.trump_suit is not None:
            raise ValueError("trump_suit must be null for Obeabe and Uneufe")
        if (
            isinstance(self.leader, bool)
            or not isinstance(self.leader, int)
            or self.leader not in range(4)
        ):
            raise ValueError("leader must be a relative seat from 0 to 3")
        if len(self.completed_tricks) > 8:
            raise ValueError("a playable state can contain at most eight completed tricks")
        if any(len(trick) != 4 for trick in self.completed_tricks):
            raise ValueError("every completed trick must contain four plays")
        if len(self.current_trick) > 3:
            raise ValueError("current_trick may contain at most three plays")
        if len(self.team_points) != 2 or any(
            isinstance(points, bool) or not isinstance(points, int) or points < 0
            for points in self.team_points
        ):
            raise ValueError("team_points must contain two non-negative integers")
        if not isinstance(self.bidding_enabled, bool) or not isinstance(self.bid_pushed, bool):
            raise ValueError("bidding flags must be booleans")
        if self.bidding_enabled:
            for name, player in (
                ("bid_starter", self.bid_starter),
                ("bid_chooser", self.bid_chooser),
            ):
                if (
                    isinstance(player, bool)
                    or not isinstance(player, int)
                    or player not in range(4)
                ):
                    raise ValueError(f"{name} must be a relative seat from 0 to 3")
            assert self.bid_starter is not None and self.bid_chooser is not None
            expected_chooser = (
                (self.bid_starter + 2) % 4 if self.bid_pushed else self.bid_starter
            )
            if self.bid_chooser != expected_chooser:
                raise ValueError(
                    "bid_chooser must be the starter, or the starter's partner after a push"
                )
        elif (
            self.bid_starter is not None
            or self.bid_chooser is not None
            or self.bid_pushed
        ):
            raise ValueError("disabled bidding requires null starter/chooser and no push")
        if not isinstance(self.announcement_enabled, bool):
            raise ValueError("announcement_enabled must be a boolean")
        if len(self.announcement_status) != 4 or any(
            status not in ANNOUNCEMENT_STATES for status in self.announcement_status
        ):
            raise ValueError("announcement_status needs four values: undecided, pass, or announce")
        if self.announcement_enabled and "undecided" in self.announcement_status:
            raise ValueError("all Weis decisions must be resolved before requesting card advice")
        if not self.announcement_enabled and self.announcement_status != ("pass",) * 4:
            raise ValueError("disabled Weis requires four pass statuses")

        cards = list(self.hand)
        for trick in (*self.completed_tricks, self.current_trick):
            players = [player for player, _ in trick]
            if players != [(players[0] + offset) % 4 for offset in range(len(players))]:
                raise ValueError("players within each trick must follow table order")
            cards.extend(card for _, card in trick)
        if len(cards) != len(set(cards)):
            raise ValueError("hand and history cards must be unique")

        expected_leader: int | None = None
        for trick in self.completed_tricks:
            if expected_leader is not None and trick[0][0] != expected_leader:
                raise ValueError("completed trick leaders are not winner-continuous")
            winner_card = winning_card(
                [card for _, card in trick],
                trick[0][1].suit,
                self.mode,
                self.trump_suit,
            )
            expected_leader = next(player for player, card in trick if card == winner_card)
        if expected_leader is not None and self.leader != expected_leader:
            raise ValueError("current leader must be the previous trick winner")
        if self.bidding_enabled:
            assert self.bid_starter is not None
            first_leader = (
                self.completed_tricks[0][0][0]
                if self.completed_tricks
                else self.leader
            )
            if first_leader != self.bid_starter:
                raise ValueError("the first trick leader must be the bidding starter")
        if self.current_trick and self.current_trick[0][0] != self.leader:
            raise ValueError("current trick must begin with leader")
        next_player = (self.leader + len(self.current_trick)) % 4
        if next_player != 0:
            raise ValueError(
                "state must be canonical to the acting player; the next relative player is not 0"
            )
        if self.hand_counts[0] != len(self.hand):
            raise ValueError(
                f"hand contains {len(self.hand)} cards but public history implies "
                f"{self.hand_counts[0]}"
            )
        if not self.hand:
            raise ValueError("acting hand must not be empty")
        if profile is not None:
            self._validate_known_hand_and_scores(profile)

    def _validate_known_hand_and_scores(self, profile: RulesetConfig) -> None:
        """Replay the acting player's known cards and minimum public score."""

        known_hand = list(self.hand)
        for trick in (*self.completed_tricks, self.current_trick):
            known_hand.extend(card for player, card in trick if player == 0)
        minimum_points = [0, 0]
        factor = profile.contract_factor(self.mode, self.trump_suit)

        for trick_index, trick in enumerate(
            (*self.completed_tricks, self.current_trick)
        ):
            plays: list[tuple[int, Card]] = []
            for player, card in trick:
                if player == 0:
                    partner_is_winning = False
                    if plays:
                        cards = [played_card for _, played_card in plays]
                        winner_card = winning_card(
                            cards,
                            cards[0].suit,
                            self.mode,
                            self.trump_suit,
                        )
                        winner = next(
                            seat
                            for seat, played_card in plays
                            if played_card == winner_card
                        )
                        partner_is_winning = winner == 2
                    legal = legal_cards(
                        known_hand,
                        [played_card for _, played_card in plays],
                        self.mode,
                        self.trump_suit,
                        partner_is_winning=partner_is_winning,
                        ruleset=profile.legal_moves,
                    )
                    if card not in legal:
                        raise ValueError(
                            "acting player's historical play is illegal given the known hand"
                        )
                    known_hand.remove(card)
                plays.append((player, card))

            if trick_index < len(self.completed_tricks):
                cards = [played_card for _, played_card in plays]
                winner_card = winning_card(
                    cards,
                    cards[0].suit,
                    self.mode,
                    self.trump_suit,
                )
                winner = next(
                    seat for seat, played_card in plays if played_card == winner_card
                )
                minimum_points[winner % 2] += factor * trick_points(
                    cards,
                    self.mode,
                    self.trump_suit,
                    last_trick=False,
                )

        for team, minimum in enumerate(minimum_points):
            if self.team_points[team] < minimum:
                raise ValueError(
                    "team_points are below the factor-scaled completed-trick total"
                )

    def action_mask(self, profile: RulesetConfig) -> np.ndarray:
        self.validate(profile)
        winner = None
        if self.current_trick:
            cards = [card for _, card in self.current_trick]
            winner_card = winning_card(
                cards,
                cards[0].suit,
                self.mode,
                self.trump_suit,
            )
            winner = next(player for player, card in self.current_trick if card == winner_card)
        legal = legal_cards(
            self.hand,
            [card for _, card in self.current_trick],
            self.mode,
            self.trump_suit,
            partner_is_winning=winner == 2,
            ruleset=profile.legal_moves,
        )
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        for card in legal:
            mask[CARD_INDEX[card]] = 1
        return mask

    def observation(self, profile: RulesetConfig) -> np.ndarray:
        self.validate(profile)
        vector = np.zeros(OBS_SIZE, dtype=np.float32)
        for card in self.hand:
            vector[OBS_HAND_OFFSET + CARD_INDEX[card]] = 1.0

        for trick_slot, trick in enumerate(self.completed_tricks):
            for play_slot, (player, card) in enumerate(trick):
                flat_slot = trick_slot * 4 + play_slot
                vector[OBS_HISTORY_CARDS_OFFSET + flat_slot * OBS_CARD_COUNT + CARD_INDEX[card]] = (
                    1.0
                )
                vector[OBS_HISTORY_PLAYERS_OFFSET + flat_slot * OBS_PLAYER_COUNT + player] = 1.0
            winner_card = winning_card(
                [card for _, card in trick],
                trick[0][1].suit,
                self.mode,
                self.trump_suit,
            )
            winner = next(player for player, card in trick if card == winner_card)
            vector[OBS_TRICK_COMPLETE_OFFSET + trick_slot] = 1.0
            vector[OBS_TRICK_WINNER_OFFSET + trick_slot * 4 + winner] = 1.0
            vector[OBS_TRICK_POINTS_OFFSET + trick_slot] = _bounded(
                trick_points(
                    [card for _, card in trick],
                    self.mode,
                    self.trump_suit,
                    last_trick=trick_slot == 8,
                ),
                20.0,
            )

        current_slot = len(self.completed_tricks)
        for play_slot, (player, card) in enumerate(self.current_trick):
            flat_slot = current_slot * 4 + play_slot
            vector[OBS_HISTORY_CARDS_OFFSET + flat_slot * OBS_CARD_COUNT + CARD_INDEX[card]] = 1.0
            vector[OBS_HISTORY_PLAYERS_OFFSET + flat_slot * OBS_PLAYER_COUNT + player] = 1.0

        vector[OBS_MODE_OFFSET + VALID_MODES.index(self.mode)] = 1.0
        if self.trump_suit is not None:
            vector[OBS_TRUMP_SUIT_OFFSET + SUITS.index(self.trump_suit)] = 1.0
        vector[OBS_TEAM_POINTS_OFFSET] = _bounded(self.team_points[0], 200.0)
        vector[OBS_TEAM_POINTS_OFFSET + 1] = _bounded(self.team_points[1], 200.0)
        vector[OBS_TRICK_INDEX_OFFSET + current_slot] = 1.0
        vector[OBS_PHASE_OFFSET + 2] = 1.0
        vector[OBS_LEADER_OFFSET + self.leader] = 1.0
        vector[OBS_ACTOR_OFFSET] = 1.0
        for player, count in enumerate(self.hand_counts):
            vector[OBS_HAND_COUNTS_OFFSET + player] = count / 9.0

        if self.bidding_enabled:
            assert self.bid_starter is not None and self.bid_chooser is not None
            vector[OBS_BIDDING_ENABLED_OFFSET] = 1.0
            vector[OBS_BID_STARTER_OFFSET + self.bid_starter] = 1.0
            vector[OBS_BID_CHOOSER_OFFSET + self.bid_chooser] = 1.0
            vector[OBS_BID_PUSHED_OFFSET] = float(self.bid_pushed)
        if self.announcement_enabled:
            vector[OBS_ANNOUNCEMENT_ENABLED_OFFSET] = 1.0
            for player, status in enumerate(self.announcement_status):
                vector[
                    OBS_ANNOUNCEMENT_STATUS_OFFSET + player * 3 + ANNOUNCEMENT_STATES[status]
                ] = 1.0
            vector[OBS_ANNOUNCEMENT_CURRENT_OFFSET] = 0.0

        factor_keys = (*SUITS, MODE_OBEABE, MODE_UNEUFE)
        for index, key in enumerate(factor_keys):
            vector[OBS_CONTRACT_FACTORS_OFFSET + index] = _bounded(
                profile.contract_factors[key], 1.0
            )
        vector[OBS_MATCH_BONUS_OFFSET] = _bounded(profile.match_bonus, 100.0)
        return vector


@dataclass(frozen=True)
class ActionAdvice:
    action: int
    card: str
    expected_margin: float | None
    standard_error: float | None
    neural_probability: float
    determinizations: int
    selected: bool


@dataclass(frozen=True)
class Advice:
    selected_action: int
    selected_card: str
    actions: tuple[ActionAdvice, ...]
    search_seconds: float
    successful_determinizations: int
    attempted_determinizations: int
    assignment_nodes: int
    used_fallback: bool
    neural_override: bool
    model_sha256: str | None
    profile_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JassAdvisor:
    """Rank legal cards from a validated public state."""

    def __init__(
        self,
        model: object,
        *,
        profile: RulesetConfig,
        search_config: PIMCConfig | None = None,
        guidance_config: NeuralGuidanceConfig | None = None,
        seed: int = 0,
        model_path: str | Path | None = None,
        bidding_enabled: bool | None = None,
        announcement_enabled: bool | None = None,
        stock_enabled: bool | None = None,
        trump_only_bidding: bool = False,
        fixed_mode: str | None = None,
        fixed_trump_suit: str | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        for name, value in (
            ("bidding_enabled", bidding_enabled),
            ("announcement_enabled", announcement_enabled),
            ("stock_enabled", stock_enabled),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean or None")
        if not isinstance(trump_only_bidding, bool):
            raise TypeError("trump_only_bidding must be a boolean")
        if trump_only_bidding and bidding_enabled is False:
            raise ValueError("trump_only_bidding requires bidding_enabled")
        if fixed_mode is not None and fixed_mode not in VALID_MODES:
            raise ValueError(f"fixed_mode must be one of {VALID_MODES} or None")
        if fixed_mode == MODE_TRUMP and (
            fixed_trump_suit is not None and fixed_trump_suit not in SUITS
        ):
            raise ValueError("fixed_trump_suit must be a Swiss suit")
        if fixed_mode != MODE_TRUMP and fixed_trump_suit is not None:
            raise ValueError("fixed_trump_suit requires fixed_mode='trump'")
        if manifest_sha256 is not None and (
            len(manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in manifest_sha256)
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        self.profile = profile
        self.bidding_enabled = bidding_enabled
        self.announcement_enabled = announcement_enabled
        self.stock_enabled = stock_enabled
        self.trump_only_bidding = trump_only_bidding
        self.fixed_mode = fixed_mode
        self.fixed_trump_suit = fixed_trump_suit
        self.manifest_sha256 = manifest_sha256
        self.implementation_identity = policy_implementation_identity()
        self.search = PIMCSearchPolicy(
            seed=seed,
            config=search_config
            or PIMCConfig(
                determinizations=24,
                max_rollouts=216,
                common_random_numbers=True,
            ),
            profile=profile,
        )
        self.policy = NeuralGuidedPIMCPolicy(
            model,
            self.search,
            config=guidance_config,
            model_path=model_path,
        )

    def advise(self, state: AdvisorState) -> Advice:
        if (
            self.bidding_enabled is not None
            and state.bidding_enabled != self.bidding_enabled
        ):
            raise ValueError(
                "advisor state bidding_enabled does not match the model environment"
            )
        if (
            self.announcement_enabled is not None
            and state.announcement_enabled != self.announcement_enabled
        ):
            raise ValueError(
                "advisor state announcement_enabled does not match the model environment"
            )
        if state.announcement_enabled and not self.profile.allow_weis:
            raise ValueError("advisor rules profile does not allow Weis")
        if self.trump_only_bidding and state.mode != MODE_TRUMP:
            raise ValueError("advisor model was trained for trump-only bidding")
        if self.fixed_mode is not None and state.mode != self.fixed_mode:
            raise ValueError(
                f"advisor model was trained for fixed contract {self.fixed_mode!r}"
            )
        if (
            self.fixed_trump_suit is not None
            and state.trump_suit != self.fixed_trump_suit
        ):
            raise ValueError(
                "advisor state trump_suit does not match the model's fixed contract"
            )
        observation = state.observation(self.profile)
        mask = state.action_mask(self.profile)
        selected = self.policy(observation, mask, "p0")
        probabilities = masked_action_probabilities(
            self.policy.model,
            observation,
            mask,
        )
        values = {
            action: (value, visits)
            for action, value, visits in self.search.last_stats.action_values
        }
        uncertainty = dict(self.search.last_stats.action_uncertainty)
        actions = []
        for action in np.flatnonzero(mask):
            action_index = int(action)
            value, visits = values.get(action_index, (None, 0))
            finite_value = (
                float(value) if value is not None and math.isfinite(value) else None
            )
            error = uncertainty.get(action_index)
            finite_error = (
                float(error) if error is not None and math.isfinite(error) else None
            )
            actions.append(
                ActionAdvice(
                    action=action_index,
                    card=card_code(ALL_CARDS[action_index]),
                    expected_margin=finite_value,
                    standard_error=finite_error,
                    neural_probability=float(probabilities[action_index]),
                    determinizations=visits,
                    selected=action_index == selected,
                )
            )
        actions.sort(
            key=lambda item: (
                item.expected_margin is not None,
                item.expected_margin if item.expected_margin is not None else 0.0,
                item.neural_probability,
            ),
            reverse=True,
        )
        last_decision = self.policy.last_decision
        return Advice(
            selected_action=selected,
            selected_card=card_code(ALL_CARDS[selected]),
            actions=tuple(actions),
            search_seconds=self.search.last_stats.elapsed_seconds,
            successful_determinizations=self.search.last_stats.successful_determinizations,
            attempted_determinizations=self.search.last_stats.attempted_determinizations,
            assignment_nodes=self.search.last_stats.assignment_nodes,
            used_fallback=self.search.last_stats.used_fallback,
            neural_override=bool(last_decision and last_decision.overridden),
            model_sha256=self.policy.model_sha256,
            profile_version=self.profile.version,
        )


def load_advisor(
    model_path: str | Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    search_config: PIMCConfig | None = None,
    guidance_config: NeuralGuidanceConfig | None = None,
) -> JassAdvisor:
    path = resolve_model_path(Path(model_path))
    manifest = load_manifest(path)
    environment_payload = manifest.get("environment")
    if not isinstance(environment_payload, dict):
        raise ValueError("model manifest environment must be a mapping")
    environment = EvaluationEnvironment.from_manifest(environment_payload)
    if environment.enable_stock != environment.profile.allow_stock:
        raise ValueError(
            "advisor requires enable_stock to match the manifested rules profile"
        )
    assert_compatible(
        manifest,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        observation_shape=environment.training_observation_shape,
        action_count=ACTION_COUNT,
        environment=environment.to_manifest_dict(),
    )
    if not assert_model_belongs_to_run(path, manifest):
        raise ValueError("advisor model manifest must include a matching SHA-256")
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - guarded by RL dependencies
        raise ImportError("sb3-contrib is required to load an advisor model") from exc
    model = MaskablePPO.load(path, device=device)
    assert_model_spaces(
        model,
        environment.training_observation_shape,
        ctde=environment.ctde,
    )
    return JassAdvisor(
        model,
        profile=environment.profile,
        search_config=search_config,
        guidance_config=guidance_config,
        seed=seed,
        model_path=path,
        bidding_enabled=environment.enable_bidding,
        announcement_enabled=environment.enable_weis,
        stock_enabled=environment.enable_stock,
        trump_only_bidding=environment.trump_only_bidding,
        fixed_mode=environment.mode,
        fixed_trump_suit=environment.trump_suit,
        manifest_sha256=_file_sha256(path.parent / MANIFEST_FILENAME),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank legal cards from a public Jass state")
    parser.add_argument("model", type=Path)
    parser.add_argument("state", type=Path, help="JSON public-state file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--determinizations", type=int, default=24)
    parser.add_argument("--max-rollouts", type=int, default=216)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        state = AdvisorState.from_dict(json.loads(args.state.read_text()))
        advisor = load_advisor(
            args.model,
            device=args.device,
            seed=args.seed,
            search_config=PIMCConfig(
                determinizations=args.determinizations,
                max_rollouts=args.max_rollouts,
                common_random_numbers=True,
            ),
        )
        payload = advisor.advise(state).to_dict()
    except (FileNotFoundError, ImportError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.output is not None:
        write_manifest(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ActionAdvice",
    "Advice",
    "AdvisorState",
    "JassAdvisor",
    "card_code",
    "load_advisor",
    "main",
    "parse_card",
]
