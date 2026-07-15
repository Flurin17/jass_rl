from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

try:
    from pettingzoo.utils import AECEnv
except ImportError as exc:  # pragma: no cover - tested via import guard
    raise ImportError("pettingzoo is required to use JassAECEnv") from exc

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional fallback
    from gym import spaces  # type: ignore

from core.announcements.stock import StockTracker
from core.announcements.weis import resolve_weis_by_player
from core.bidding import BiddingAction
from core.cards import ALL_CARDS, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from core.legal_moves import RuleSet
from core.ruleset import STANDARD_RULES_PROFILE, RulesetConfig
from core.state import GameState
from core.transitions import (
    ScoreEvent,
    ScoreEventKind,
    TrickTransition,
    award_raw_points,
    resolve_completed_trick,
)

BIDDING_TRUMP_ACTIONS = {36: "schellen", 37: "rosen", 38: "schilten", 39: "eicheln"}
BIDDING_OBEABE_ACTION = 40
BIDDING_UNEUFE_ACTION = 41
BIDDING_PUSH_ACTION = 42
ANNOUNCE_ACTION = 43
PASS_ACTION = 44
ACTION_COUNT = 45


def trump_only_action_mask(action_mask: np.ndarray) -> np.ndarray:
    """Return a copy whose bidding choices are limited to trump suits and push.

    Card and announcement actions are left unchanged.  Keeping this operation
    at the canonical action-mask boundary lets training and matched evaluation
    apply exactly the same restriction without mutating a caller-owned mask.
    """

    mask = np.asarray(action_mask).reshape(-1)
    if mask.shape != (ACTION_COUNT,):
        raise ValueError(f"action mask must contain {ACTION_COUNT} entries")
    restricted = mask.copy()
    restricted[BIDDING_OBEABE_ACTION] = 0
    restricted[BIDDING_UNEUFE_ACTION] = 0
    return restricted

# Version 2 is a canonical, acting-seat-relative schema. A model trained against
# an earlier observation layout is intentionally not shape-compatible with it.
OBSERVATION_SCHEMA_VERSION = 2
OBSERVATION_SCHEMA_NAME = "canonical_public_history_v2"

OBS_TRICK_COUNT = 9
OBS_PLAYS_PER_TRICK = 4
OBS_CARD_COUNT = 36
OBS_PLAYER_COUNT = 4

OBS_HAND_OFFSET = 0
OBS_HISTORY_CARDS_OFFSET = OBS_HAND_OFFSET + OBS_CARD_COUNT
OBS_HISTORY_CARDS_SIZE = OBS_TRICK_COUNT * OBS_PLAYS_PER_TRICK * OBS_CARD_COUNT
OBS_HISTORY_PLAYERS_OFFSET = OBS_HISTORY_CARDS_OFFSET + OBS_HISTORY_CARDS_SIZE
OBS_HISTORY_PLAYERS_SIZE = OBS_TRICK_COUNT * OBS_PLAYS_PER_TRICK * OBS_PLAYER_COUNT
OBS_TRICK_COMPLETE_OFFSET = OBS_HISTORY_PLAYERS_OFFSET + OBS_HISTORY_PLAYERS_SIZE
OBS_TRICK_WINNER_OFFSET = OBS_TRICK_COMPLETE_OFFSET + OBS_TRICK_COUNT
OBS_TRICK_WINNER_SIZE = OBS_TRICK_COUNT * OBS_PLAYER_COUNT
OBS_TRICK_POINTS_OFFSET = OBS_TRICK_WINNER_OFFSET + OBS_TRICK_WINNER_SIZE
OBS_MODE_OFFSET = OBS_TRICK_POINTS_OFFSET + OBS_TRICK_COUNT
OBS_TRUMP_SUIT_OFFSET = OBS_MODE_OFFSET + 3
OBS_TEAM_POINTS_OFFSET = OBS_TRUMP_SUIT_OFFSET + 4
OBS_TRICK_INDEX_OFFSET = OBS_TEAM_POINTS_OFFSET + 2
OBS_PHASE_OFFSET = OBS_TRICK_INDEX_OFFSET + 10
OBS_LEADER_OFFSET = OBS_PHASE_OFFSET + 4
OBS_ACTOR_OFFSET = OBS_LEADER_OFFSET + 4
OBS_HAND_COUNTS_OFFSET = OBS_ACTOR_OFFSET + 4
OBS_BIDDING_ENABLED_OFFSET = OBS_HAND_COUNTS_OFFSET + 4
OBS_BID_STARTER_OFFSET = OBS_BIDDING_ENABLED_OFFSET + 1
OBS_BID_CURRENT_OFFSET = OBS_BID_STARTER_OFFSET + 4
OBS_BID_PUSHED_OFFSET = OBS_BID_CURRENT_OFFSET + 4
OBS_BID_CHOOSER_OFFSET = OBS_BID_PUSHED_OFFSET + 1
OBS_ANNOUNCEMENT_ENABLED_OFFSET = OBS_BID_CHOOSER_OFFSET + 4
OBS_ANNOUNCEMENT_STATUS_OFFSET = OBS_ANNOUNCEMENT_ENABLED_OFFSET + 1
OBS_ANNOUNCEMENT_CURRENT_OFFSET = OBS_ANNOUNCEMENT_STATUS_OFFSET + 4 * 3
OBS_CONTRACT_FACTORS_OFFSET = OBS_ANNOUNCEMENT_CURRENT_OFFSET + 4
OBS_MATCH_BONUS_OFFSET = OBS_CONTRACT_FACTORS_OFFSET + 6
OBS_SIZE = OBS_MATCH_BONUS_OFFSET + 1

# Public layout metadata lets policies decode the versioned vector without
# relying on historical aliases whose meanings no longer match this schema.
OBSERVATION_SCHEMA_FIELDS = {
    "hand": (OBS_HAND_OFFSET, (OBS_CARD_COUNT,)),
    "history_cards": (
        OBS_HISTORY_CARDS_OFFSET,
        (OBS_TRICK_COUNT, OBS_PLAYS_PER_TRICK, OBS_CARD_COUNT),
    ),
    "history_players": (
        OBS_HISTORY_PLAYERS_OFFSET,
        (OBS_TRICK_COUNT, OBS_PLAYS_PER_TRICK, OBS_PLAYER_COUNT),
    ),
    "trick_complete": (OBS_TRICK_COMPLETE_OFFSET, (OBS_TRICK_COUNT,)),
    "trick_winner": (
        OBS_TRICK_WINNER_OFFSET,
        (OBS_TRICK_COUNT, OBS_PLAYER_COUNT),
    ),
    "trick_points": (OBS_TRICK_POINTS_OFFSET, (OBS_TRICK_COUNT,)),
    "mode": (OBS_MODE_OFFSET, (3,)),
    "trump_suit": (OBS_TRUMP_SUIT_OFFSET, (4,)),
    "team_points": (OBS_TEAM_POINTS_OFFSET, (2,)),
    "trick_index": (OBS_TRICK_INDEX_OFFSET, (10,)),
    "phase": (OBS_PHASE_OFFSET, (4,)),
    "leader": (OBS_LEADER_OFFSET, (4,)),
    "actor": (OBS_ACTOR_OFFSET, (4,)),
    "hand_counts": (OBS_HAND_COUNTS_OFFSET, (4,)),
    "bidding_enabled": (OBS_BIDDING_ENABLED_OFFSET, (1,)),
    "bid_starter": (OBS_BID_STARTER_OFFSET, (4,)),
    "bid_current": (OBS_BID_CURRENT_OFFSET, (4,)),
    "bid_pushed": (OBS_BID_PUSHED_OFFSET, (1,)),
    "bid_chooser": (OBS_BID_CHOOSER_OFFSET, (4,)),
    "announcement_enabled": (OBS_ANNOUNCEMENT_ENABLED_OFFSET, (1,)),
    "announcement_status": (OBS_ANNOUNCEMENT_STATUS_OFFSET, (4, 3)),
    "announcement_current": (OBS_ANNOUNCEMENT_CURRENT_OFFSET, (4,)),
    "contract_factors": (OBS_CONTRACT_FACTORS_OFFSET, (6,)),
    "match_bonus": (OBS_MATCH_BONUS_OFFSET, (1,)),
}


def _validated_observation_vector(observation: np.ndarray) -> np.ndarray:
    vector = np.asarray(observation).reshape(-1)
    if vector.shape != (OBS_SIZE,):
        raise ValueError(
            f"observation must have shape ({OBS_SIZE},), got {np.asarray(observation).shape}"
        )
    return vector


def decode_observation_history(
    observation: np.ndarray,
) -> list[list[tuple[int, Card]]]:
    """Decode public trick/play history as ``(relative_player, card)`` pairs."""

    vector = _validated_observation_vector(observation)
    history: list[list[tuple[int, Card]]] = []
    for trick_slot in range(OBS_TRICK_COUNT):
        plays: list[tuple[int, Card]] = []
        for play_slot in range(OBS_PLAYS_PER_TRICK):
            flat_slot = trick_slot * OBS_PLAYS_PER_TRICK + play_slot
            card_start = OBS_HISTORY_CARDS_OFFSET + flat_slot * OBS_CARD_COUNT
            card_indices = np.flatnonzero(
                vector[card_start : card_start + OBS_CARD_COUNT] > 0.5
            )
            if card_indices.size == 0:
                break
            if card_indices.size != 1:
                raise ValueError("history play must encode exactly one card")

            player_start = (
                OBS_HISTORY_PLAYERS_OFFSET + flat_slot * OBS_PLAYER_COUNT
            )
            player_indices = np.flatnonzero(
                vector[player_start : player_start + OBS_PLAYER_COUNT] > 0.5
            )
            if player_indices.size != 1:
                raise ValueError("history play must encode exactly one relative player")
            plays.append(
                (int(player_indices[0]), ALL_CARDS[int(card_indices[0])])
            )
        if not plays:
            break
        history.append(plays)
    return history


def decode_current_trick(observation: np.ndarray) -> list[Card]:
    """Return the current trick's ordered cards from a version-2 observation."""

    vector = _validated_observation_vector(observation)
    indices = np.flatnonzero(
        vector[OBS_TRICK_INDEX_OFFSET : OBS_TRICK_INDEX_OFFSET + 10] > 0.5
    )
    if indices.size != 1:
        raise ValueError("observation must encode exactly one trick index")
    trick_index = int(indices[0])
    history = decode_observation_history(vector)
    if trick_index >= OBS_TRICK_COUNT or trick_index >= len(history):
        return []
    return [card for _, card in history[trick_index]]


def decode_played_cards(observation: np.ndarray) -> set[Card]:
    """Return every publicly played card, including the current trick."""

    return {
        card
        for trick in decode_observation_history(observation)
        for _, card in trick
    }


@dataclass
class BiddingStatus:
    starter: int
    current_player: int
    pushed: bool


@dataclass
class AnnouncementStatus:
    order: list[int]
    index: int


class JassAECEnv(AECEnv):
    metadata = {
        "name": "jass_aec_env",
        "render_modes": [],
        "observation_schema": OBSERVATION_SCHEMA_NAME,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_fields": OBSERVATION_SCHEMA_FIELDS,
    }

    def __init__(
        self,
        seed: int | None = None,
        ruleset: RuleSet | None = None,
        profile: RulesetConfig | None = None,
        enable_bidding: bool = True,
        enable_weis: bool = True,
        enable_stock: bool = True,
        mode: str | None = None,
        trump_suit: str | None = None,
        starter: int = 0,
        trump_only_bidding: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(starter, bool) or not isinstance(starter, int) or starter not in range(4):
            raise ValueError("starter must be an integer from 0 to 3")
        if not isinstance(trump_only_bidding, bool):
            raise TypeError("trump_only_bidding must be a boolean")
        if trump_only_bidding and not enable_bidding:
            raise ValueError("trump_only_bidding requires bidding to be enabled")
        self.possible_agents = ["p0", "p1", "p2", "p3"]
        self.agents: list[str] = []
        self._seed = seed
        self._rng = random.Random(seed)
        self.profile = profile or STANDARD_RULES_PROFILE
        self.ruleset = ruleset or self.profile.legal_moves
        self.enable_bidding = enable_bidding
        self.enable_weis = enable_weis and self.profile.allow_weis
        self.enable_stock = enable_stock and self.profile.allow_stock
        self.trump_only_bidding = trump_only_bidding
        self.preset_mode = mode
        self.preset_trump_suit = trump_suit
        self.starter = starter

        self.card_to_index: dict[tuple[str, str], int] = {
            (card.suit, card.rank): idx for idx, card in enumerate(ALL_CARDS)
        }
        self.index_to_card: list[Card] = list(ALL_CARDS)
        self._obs_buffer: dict[str, np.ndarray] = {
            agent: np.zeros(OBS_SIZE, dtype=np.float32) for agent in self.possible_agents
        }
        self._mask_buffer: dict[str, np.ndarray] = {
            agent: np.zeros(ACTION_COUNT, dtype=np.int8) for agent in self.possible_agents
        }

        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
                ),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_COUNT,), dtype=np.int8),
            }
        )
        self._action_space = spaces.Discrete(ACTION_COUNT)

        self.phase = "bidding"
        self.bidding: BiddingStatus | None = None
        self.announcement: AnnouncementStatus | None = None
        self._announced_cards: dict[int, list[Card]] = {}
        self._announcement_decisions: dict[int, bool | None] = {}
        self._chooser: int | None = None
        self.state: GameState | None = None
        self._pending_hands: list[list[Card]] | None = None
        self._stock: StockTracker | None = None
        self.mode: str | None = None
        self.trump_suit: str | None = None
        self.contract_factor = 1
        self.match_team: int | None = None

        self.metadata = dict(type(self).metadata)
        self.metadata["rules_profile_version"] = self.profile.version

        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}
        self.last_rewards: dict[str, float] = {
            agent: 0.0 for agent in self.possible_agents
        }

    def _deal_hands(self) -> list[list[Card]]:
        deck = list(ALL_CARDS)
        self._rng.shuffle(deck)
        return [deck[i * 9 : (i + 1) * 9] for i in range(4)]

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self._observation_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.last_rewards = {agent: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._skip_agent_selection = None

        options = options or {}
        self._pending_hands = None
        self._stock = None
        self._chooser = None
        self._announcement_decisions = {}
        self.contract_factor = 1
        self.match_team = None

        if self.enable_bidding:
            # In Schieber, cards are dealt before bidding.
            self._pending_hands = self._deal_hands()
            self.phase = "bidding"
            self.bidding = BiddingStatus(
                starter=self.starter, current_player=self.starter, pushed=False
            )
            self.announcement = None
            self._announced_cards = {}
            self.mode = None
            self.trump_suit = None
            self.state = None
            self.agent_selection = f"p{self.bidding.current_player}"
        else:
            self.bidding = None
            self.announcement = None
            self._announced_cards = {}
            requested_mode = options.get("mode", self.preset_mode)
            requested_trump = options.get("trump_suit", self.preset_trump_suit)
            if requested_mode is None:
                self.mode = self._rng.choice([MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE])
                self.trump_suit = None
                if self.mode == MODE_TRUMP:
                    self.trump_suit = requested_trump or self._rng.choice(SUITS)
            else:
                self.mode = requested_mode
                self.trump_suit = requested_trump
                if self.mode == MODE_TRUMP and self.trump_suit is None:
                    self.trump_suit = self._rng.choice(SUITS)
            self._init_state(leader=self.starter)
            if self.enable_weis:
                self._start_announcement(leader=self.state.leader)
            else:
                self.phase = "play"
                self.agent_selection = f"p{self.state.leader}"

    def _init_state(
        self, leader: int, hands: list[list[Card]] | None = None
    ) -> None:
        hands = hands if hands is not None else self._deal_hands()
        assert self.mode is not None
        self.contract_factor = self.profile.contract_factor(self.mode, self.trump_suit)
        self.state = GameState(
            hands=hands,
            mode=self.mode,
            trump_suit=self.trump_suit,
            leader=leader,
        )
        if (
            self.enable_stock
            and self.mode == MODE_TRUMP
            and self.trump_suit is not None
        ):
            self._stock = StockTracker()

    def _start_announcement(self, leader: int) -> None:
        order = [(leader + offset) % 4 for offset in range(4)]
        self.announcement = AnnouncementStatus(order=order, index=0)
        self._announced_cards = {player: [] for player in range(4)}
        self._announcement_decisions = {player: None for player in range(4)}
        self.phase = "announce"
        self.agent_selection = f"p{self.announcement.order[self.announcement.index]}"

    def observe(self, agent: str):
        observation = self._build_observation(agent)
        mask = self._build_action_mask(agent)
        # Callers are allowed to retain or modify observations. Returning copies
        # prevents the reusable buffers from changing a previously returned value.
        return {"observation": observation.copy(), "action_mask": mask.copy()}

    @staticmethod
    def _relative_seat(player: int, observer: int) -> int:
        """Map an absolute seat to self/left/partner/right for ``observer``."""

        return (player - observer) % 4

    @staticmethod
    def _bounded_points(points: int, scale: float) -> float:
        """Monotonically encode any non-negative score inside ``[0, 1)``."""

        value = max(0.0, float(points))
        return value / (value + scale) if value else 0.0

    def _build_observation(self, agent: str) -> np.ndarray:
        buffer = self._obs_buffer[agent]
        buffer.fill(0.0)
        observer = int(agent[1:])

        # Private information: exactly the observing player's current hand.
        if self.state is not None:
            hand = self.state.hands[observer]
        elif self._pending_hands is not None:
            hand = self._pending_hands[observer]
        else:
            hand = []
        for card in hand:
            idx = self.card_to_index[(card.suit, card.rank)]
            buffer[OBS_HAND_OFFSET + idx] = 1.0

        # Public play history. Each of the 9 trick slots contains 4 ordered play
        # slots, and every play records both its card and relative player.
        if self.state is not None:
            public_tricks = [trick.plays for trick in self.state.completed_tricks]
            for trick_slot, plays in enumerate(public_tricks[:OBS_TRICK_COUNT]):
                for play_slot, (player, card) in enumerate(
                    plays[:OBS_PLAYS_PER_TRICK]
                ):
                    flat_slot = trick_slot * OBS_PLAYS_PER_TRICK + play_slot
                    card_index = self.card_to_index[(card.suit, card.rank)]
                    buffer[
                        OBS_HISTORY_CARDS_OFFSET
                        + flat_slot * OBS_CARD_COUNT
                        + card_index
                    ] = 1.0
                    buffer[
                        OBS_HISTORY_PLAYERS_OFFSET
                        + flat_slot * OBS_PLAYER_COUNT
                        + self._relative_seat(player, observer)
                    ] = 1.0

                result = self.state.completed_tricks[trick_slot]
                buffer[OBS_TRICK_COMPLETE_OFFSET + trick_slot] = 1.0
                buffer[
                    OBS_TRICK_WINNER_OFFSET
                    + trick_slot * OBS_PLAYER_COUNT
                    + self._relative_seat(result.winner, observer)
                ] = 1.0
                buffer[OBS_TRICK_POINTS_OFFSET + trick_slot] = self._bounded_points(
                    result.points, 20.0
                )

            current_slot = self.state.trick_index
            if 0 <= current_slot < OBS_TRICK_COUNT:
                for play_slot, (player, card) in enumerate(
                    self.state.trick.plays[:OBS_PLAYS_PER_TRICK]
                ):
                    flat_slot = current_slot * OBS_PLAYS_PER_TRICK + play_slot
                    card_index = self.card_to_index[(card.suit, card.rank)]
                    buffer[
                        OBS_HISTORY_CARDS_OFFSET
                        + flat_slot * OBS_CARD_COUNT
                        + card_index
                    ] = 1.0
                    buffer[
                        OBS_HISTORY_PLAYERS_OFFSET
                        + flat_slot * OBS_PLAYER_COUNT
                        + self._relative_seat(player, observer)
                    ] = 1.0

            self_team = self.state.team_index(observer)
            opponent_team = 1 - self_team
            buffer[OBS_TEAM_POINTS_OFFSET] = self._bounded_points(
                self.state.team_points[self_team], 200.0
            )
            buffer[OBS_TEAM_POINTS_OFFSET + 1] = self._bounded_points(
                self.state.team_points[opponent_team], 200.0
            )
            trick_index = min(max(self.state.trick_index, 0), OBS_TRICK_COUNT)
            buffer[OBS_TRICK_INDEX_OFFSET + trick_index] = 1.0

            leader = self.state.leader
            buffer[
                OBS_LEADER_OFFSET + self._relative_seat(leader, observer)
            ] = 1.0

            for relative_player in range(OBS_PLAYER_COUNT):
                absolute_player = (observer + relative_player) % OBS_PLAYER_COUNT
                buffer[OBS_HAND_COUNTS_OFFSET + relative_player] = (
                    len(self.state.hands[absolute_player]) / 9.0
                )
        else:
            buffer[OBS_TRICK_INDEX_OFFSET] = 1.0
            buffer[
                OBS_LEADER_OFFSET + self._relative_seat(self.starter, observer)
            ] = 1.0
            if self._pending_hands is not None:
                for relative_player in range(OBS_PLAYER_COUNT):
                    absolute_player = (observer + relative_player) % OBS_PLAYER_COUNT
                    buffer[OBS_HAND_COUNTS_OFFSET + relative_player] = (
                        len(self._pending_hands[absolute_player]) / 9.0
                    )

        mode = self.state.mode if self.state is not None else self.mode
        trump_suit = self.state.trump_suit if self.state is not None else self.trump_suit
        if mode == MODE_TRUMP:
            buffer[OBS_MODE_OFFSET] = 1.0
        elif mode == MODE_OBEABE:
            buffer[OBS_MODE_OFFSET + 1] = 1.0
        elif mode == MODE_UNEUFE:
            buffer[OBS_MODE_OFFSET + 2] = 1.0
        if trump_suit:
            buffer[OBS_TRUMP_SUIT_OFFSET + SUITS.index(trump_suit)] = 1.0

        phase_index = {"bidding": 0, "announce": 1, "play": 2, "terminal": 3}.get(
            self.phase, 3
        )
        buffer[OBS_PHASE_OFFSET + phase_index] = 1.0
        if self.phase != "terminal" and self.agent_selection in self.possible_agents:
            actor = int(self.agent_selection[1:])
            buffer[OBS_ACTOR_OFFSET + self._relative_seat(actor, observer)] = 1.0

        # Public bidding state: starter, current bidder, push, and final chooser.
        if self.enable_bidding:
            buffer[OBS_BIDDING_ENABLED_OFFSET] = 1.0
            bid_starter = self.bidding.starter if self.bidding is not None else self.starter
            buffer[
                OBS_BID_STARTER_OFFSET + self._relative_seat(bid_starter, observer)
            ] = 1.0
            if self.phase == "bidding" and self.bidding is not None:
                buffer[
                    OBS_BID_CURRENT_OFFSET
                    + self._relative_seat(self.bidding.current_player, observer)
                ] = 1.0
            if self.bidding is not None and self.bidding.pushed:
                buffer[OBS_BID_PUSHED_OFFSET] = 1.0
            if self._chooser is not None:
                buffer[
                    OBS_BID_CHOOSER_OFFSET
                    + self._relative_seat(self._chooser, observer)
                ] = 1.0

        # Announcement actions are public, but announced hands remain private.
        if self.enable_weis:
            buffer[OBS_ANNOUNCEMENT_ENABLED_OFFSET] = 1.0
            for player, decision in self._announcement_decisions.items():
                relative_player = self._relative_seat(player, observer)
                status = 0 if decision is None else (2 if decision else 1)
                buffer[
                    OBS_ANNOUNCEMENT_STATUS_OFFSET + relative_player * 3 + status
                ] = 1.0
            if self.phase == "announce" and self.announcement is not None:
                current = self.announcement.order[self.announcement.index]
                buffer[
                    OBS_ANNOUNCEMENT_CURRENT_OFFSET
                    + self._relative_seat(current, observer)
                ] = 1.0

        factor_keys = (*SUITS, MODE_OBEABE, MODE_UNEUFE)
        for index, key in enumerate(factor_keys):
            buffer[OBS_CONTRACT_FACTORS_OFFSET + index] = self._bounded_points(
                self.profile.contract_factors[key], 1.0
            )
        buffer[OBS_MATCH_BONUS_OFFSET] = self._bounded_points(
            self.profile.match_bonus, 100.0
        )

        return buffer

    def _build_action_mask(self, agent: str) -> np.ndarray:
        mask = self._mask_buffer[agent]
        mask.fill(0)
        if agent not in self.agents or self.phase == "terminal":
            return mask
        if self.terminations.get(agent) or self.truncations.get(agent):
            return mask
        if agent != self.agent_selection:
            return mask

        if self.phase == "bidding":
            assert self.bidding is not None
            if int(agent[1:]) != self.bidding.current_player:
                return mask
            # trump choices
            for action in BIDDING_TRUMP_ACTIONS:
                mask[action] = 1
            if not self.trump_only_bidding:
                mask[BIDDING_OBEABE_ACTION] = 1
                mask[BIDDING_UNEUFE_ACTION] = 1
            if not self.bidding.pushed and self.bidding.current_player == self.bidding.starter:
                mask[BIDDING_PUSH_ACTION] = 1
            return mask

        if self.phase == "announce":
            assert self.announcement is not None
            current_player = self.announcement.order[self.announcement.index]
            if int(agent[1:]) != current_player:
                return mask
            mask[ANNOUNCE_ACTION] = 1
            mask[PASS_ACTION] = 1
            return mask

        if self.state is None:
            return mask

        player = int(agent[1:])
        legal = self.state.legal_cards_for(player, ruleset=self.ruleset)
        for card in legal:
            idx = self.card_to_index[(card.suit, card.rank)]
            mask[idx] = 1
        return mask

    def step(self, action):
        agent = self.agent_selection

        if self.terminations.get(agent) or self.truncations.get(agent):
            self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
            self._was_dead_step(action)
            return

        # AEC rewards are immediate values from the most recent step, while
        # _cumulative_rewards holds everything earned since an agent last acted.
        # Clear the acting agent's consumed cumulative reward and the prior step's
        # immediate rewards before applying this action.
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()

        if self.phase == "bidding":
            self._step_bidding(agent, action)
            self._finish_step()
            return

        if self.phase == "announce":
            self._step_announce(agent, action)
            self._finish_step()
            return

        if action is None:
            raise ValueError("action required during play phase")

        card = self._action_to_card(action)
        player = int(agent[1:])
        self.state.play_card(player, card, ruleset=self.ruleset)
        if (
            self._stock is not None
            and self.mode == MODE_TRUMP
            and self.trump_suit is not None
        ):
            stock_points = self._stock.record_play(player, card, self.trump_suit)
            if stock_points:
                self._award_team_points(
                    self.state.team_index(player),
                    stock_points,
                    kind=ScoreEventKind.STOCK,
                )

        if len(self.state.trick.plays) == 4:
            self._resolve_trick()

        if self.state.is_terminal:
            self.state.validate_terminal()
            for a in self.agents:
                self.terminations[a] = True
            self.phase = "terminal"
        else:
            self.agent_selection = f"p{self.state.current_player}"

        self._finish_step()

    def _step_bidding(self, agent: str, action: int) -> None:
        if action is None:
            raise ValueError("action required during bidding")
        if action not in range(ACTION_COUNT):
            raise ValueError("invalid action")
        if not self._build_action_mask(agent)[action]:
            raise ValueError("illegal bidding action")

        assert self.bidding is not None
        current_player = self.bidding.current_player
        if agent != f"p{current_player}":
            raise ValueError("not this player's bidding turn")

        bidding_action = self._action_to_bidding(action)
        if bidding_action.push:
            if self.bidding.pushed:
                raise ValueError("partner may not push")
            self.bidding = BiddingStatus(
                starter=self.bidding.starter,
                current_player=(self.bidding.starter + 2) % 4,
                pushed=True,
            )
            self.agent_selection = f"p{self.bidding.current_player}"
            return

        self.mode = bidding_action.mode
        self.trump_suit = bidding_action.trump_suit
        self._chooser = current_player
        assert self._pending_hands is not None
        self._init_state(leader=self.bidding.starter, hands=self._pending_hands)
        self._pending_hands = None
        if self.enable_weis:
            self._start_announcement(leader=self.state.leader)
        else:
            self.phase = "play"
            self.agent_selection = f"p{self.state.leader}"

    def _action_to_bidding(self, action: int) -> BiddingAction:
        if action == BIDDING_PUSH_ACTION:
            return BiddingAction(mode="", push=True)
        if action in BIDDING_TRUMP_ACTIONS:
            return BiddingAction(mode=MODE_TRUMP, trump_suit=BIDDING_TRUMP_ACTIONS[action])
        if action == BIDDING_OBEABE_ACTION:
            return BiddingAction(mode=MODE_OBEABE)
        if action == BIDDING_UNEUFE_ACTION:
            return BiddingAction(mode=MODE_UNEUFE)
        raise ValueError("invalid bidding action")

    def _action_to_card(self, action: int) -> Card:
        if action < 0 or action >= 36:
            raise ValueError("card action expected")
        return self.index_to_card[action]

    def _step_announce(self, agent: str, action: int) -> None:
        if action not in (ANNOUNCE_ACTION, PASS_ACTION):
            raise ValueError("invalid announce action")
        assert self.announcement is not None
        assert self.state is not None

        current_player = self.announcement.order[self.announcement.index]
        if agent != f"p{current_player}":
            raise ValueError("not this player's announce turn")

        if action == ANNOUNCE_ACTION:
            self._announced_cards[current_player] = list(self.state.hands[current_player])
            self._announcement_decisions[current_player] = True
        else:
            self._announced_cards[current_player] = []
            self._announcement_decisions[current_player] = False

        self.announcement.index += 1

        if self.announcement.index >= len(self.announcement.order):
            points_a, points_b, _, _ = resolve_weis_by_player(
                self._announced_cards,
                list(self.announcement.order),
                mode=self.state.mode,
                trump_suit=self.state.trump_suit,
            )

            if points_a:
                self._award_team_points(0, points_a, kind=ScoreEventKind.WEIS)
            if points_b:
                self._award_team_points(1, points_b, kind=ScoreEventKind.WEIS)

            self.phase = "play"
            self.agent_selection = f"p{self.state.leader}"
            self.announcement = None
            return

        next_player = self.announcement.order[self.announcement.index]
        self.agent_selection = f"p{next_player}"

    def _award_team_points(
        self,
        team: int,
        raw_points: int,
        *,
        kind: ScoreEventKind = ScoreEventKind.BONUS,
    ) -> int:
        """Apply the active contract factor and publish a team reward."""

        assert self.state is not None
        event = award_raw_points(
            self.state,
            team,
            raw_points,
            kind=kind,
            profile=self.profile,
        )
        self._publish_score_events((event,))
        return event.scored_points

    def _publish_score_events(self, events: Iterable[ScoreEvent]) -> None:
        for event in events:
            team_agents = ("p0", "p2") if event.team == 0 else ("p1", "p3")
            for agent in team_agents:
                if agent in self.rewards:
                    self.rewards[agent] += event.scored_points

    def _resolve_trick(self) -> TrickTransition:
        transition = resolve_completed_trick(self.state, profile=self.profile)
        self._publish_score_events(transition.score_events)
        if transition.finalization is not None:
            self.match_team = transition.finalization.match_team
        self.agent_selection = f"p{self.state.leader}"
        return transition

    def _finish_step(self) -> None:
        """Publish and accumulate this step's rewards without erasing them."""

        self.last_rewards = {
            agent: float(self.rewards.get(agent, 0.0))
            for agent in self.possible_agents
        }
        self._accumulate_rewards()
