from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from pettingzoo.utils import AECEnv
except ImportError as exc:  # pragma: no cover - tested via import guard
    raise ImportError("pettingzoo is required to use JassAECEnv") from exc

try:
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional fallback
    from gym import spaces  # type: ignore

from core.bidding import BiddingAction
from core.cards import ALL_CARDS, Card, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS
from core.legal_moves import RuleSet
from core.rankings import winning_card
from core.scoring import trick_points
from core.announcements.weis import resolve_weis
from core.state import GameState, Trick, TrickResult


BIDDING_TRUMP_ACTIONS = {36: "schellen", 37: "rosen", 38: "schilten", 39: "eicheln"}
BIDDING_OBEABE_ACTION = 40
BIDDING_UNEUFE_ACTION = 41
BIDDING_PUSH_ACTION = 42
ANNOUNCE_ACTION = 43
PASS_ACTION = 44
ACTION_COUNT = 45

OBS_HAND_OFFSET = 0
OBS_TRICK_OFFSET = 36
OBS_PLAYED_OFFSET = 72
OBS_TRUMP_MODE_OFFSET = 108
OBS_TRUMP_SUIT_OFFSET = 111
OBS_POINTS_OFFSET = 115
OBS_TRICK_INDEX_OFFSET = 117


@dataclass
class BiddingStatus:
    starter: int
    current_player: int
    pushed: bool


@dataclass
class AnnouncementStatus:
    order: List[int]
    index: int


class JassAECEnv(AECEnv):
    metadata = {"name": "jass_aec_env", "render_modes": []}

    def __init__(
        self,
        seed: Optional[int] = None,
        ruleset: Optional[RuleSet] = None,
        enable_bidding: bool = True,
        enable_weis: bool = True,
        mode: Optional[str] = None,
        trump_suit: Optional[str] = None,
        starter: int = 0,
    ) -> None:
        super().__init__()
        self.possible_agents = ["p0", "p1", "p2", "p3"]
        self.agents: List[str] = []
        self._seed = seed
        self._rng = random.Random(seed)
        self.ruleset = ruleset or RuleSet()
        self.enable_bidding = enable_bidding
        self.enable_weis = enable_weis
        self.preset_mode = mode
        self.preset_trump_suit = trump_suit
        self.starter = starter

        self.card_to_index: Dict[Tuple[str, str], int] = {
            (card.suit, card.rank): idx for idx, card in enumerate(ALL_CARDS)
        }
        self.index_to_card: List[Card] = list(ALL_CARDS)
        self._obs_buffer: Dict[str, np.ndarray] = {
            agent: np.zeros(118, dtype=np.float32) for agent in self.possible_agents
        }
        self._mask_buffer: Dict[str, np.ndarray] = {
            agent: np.zeros(ACTION_COUNT, dtype=np.int8) for agent in self.possible_agents
        }

        self._observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=0.0, high=1.0, shape=(118,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_COUNT,), dtype=np.int8),
            }
        )
        self._action_space = spaces.Discrete(ACTION_COUNT)

        self.phase = "bidding"
        self.bidding: Optional[BiddingStatus] = None
        self.announcement: Optional[AnnouncementStatus] = None
        self._announced_cards: Dict[int, List[Card]] = {}
        self.state: Optional[GameState] = None

        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}
        self.last_rewards: Dict[str, float] = {}

    def action_space(self, agent: str):
        return self._action_space

    def observation_space(self, agent: str):
        return self._observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.last_rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        options = options or {}

        if self.enable_bidding:
            self.phase = "bidding"
            self.bidding = BiddingStatus(
                starter=self.starter, current_player=self.starter, pushed=False
            )
            self.announcement = None
            self._announced_cards = {}
            self.mode = None
            self.trump_suit = None
            self.agent_selection = f"p{self.bidding.current_player}"
        else:
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

    def _init_state(self, leader: int) -> None:
        deck = list(ALL_CARDS)
        self._rng.shuffle(deck)
        hands = [deck[i * 9 : (i + 1) * 9] for i in range(4)]
        self.state = GameState(hands=hands, mode=self.mode, trump_suit=self.trump_suit)
        self.state.leader = leader
        self.state.trick = Trick()
        self.state.trick_index = 0

    def _start_announcement(self, leader: int) -> None:
        order = [(leader + offset) % 4 for offset in range(4)]
        self.announcement = AnnouncementStatus(order=order, index=0)
        self._announced_cards = {player: [] for player in range(4)}
        self.phase = "announce"
        self.agent_selection = f"p{self.announcement.order[self.announcement.index]}"

    def observe(self, agent: str):
        observation = self._build_observation(agent)
        mask = self._build_action_mask(agent)
        return {"observation": observation, "action_mask": mask}

    def _build_observation(self, agent: str) -> np.ndarray:
        buffer = self._obs_buffer[agent]
        buffer.fill(0.0)
        if self.state is None:
            return buffer

        player = int(agent[1:])
        for card in self.state.hands[player]:
            idx = self.card_to_index[(card.suit, card.rank)]
            buffer[OBS_HAND_OFFSET + idx] = 1.0

        for card in self.state.trick.cards:
            idx = self.card_to_index[(card.suit, card.rank)]
            buffer[OBS_TRICK_OFFSET + idx] = 1.0

        for trick in self.state.completed_tricks:
            for card in trick.cards:
                idx = self.card_to_index[(card.suit, card.rank)]
                buffer[OBS_PLAYED_OFFSET + idx] = 1.0
        for card in self.state.trick.cards:
            idx = self.card_to_index[(card.suit, card.rank)]
            buffer[OBS_PLAYED_OFFSET + idx] = 1.0

        if self.state.mode == MODE_TRUMP:
            buffer[OBS_TRUMP_MODE_OFFSET + 0] = 1.0
        elif self.state.mode == MODE_OBEABE:
            buffer[OBS_TRUMP_MODE_OFFSET + 1] = 1.0
        elif self.state.mode == MODE_UNEUFE:
            buffer[OBS_TRUMP_MODE_OFFSET + 2] = 1.0

        if self.state.trump_suit:
            buffer[OBS_TRUMP_SUIT_OFFSET + SUITS.index(self.state.trump_suit)] = 1.0

        buffer[OBS_POINTS_OFFSET : OBS_POINTS_OFFSET + 2] = np.array(
            self.state.team_points, dtype=np.float32
        )
        buffer[OBS_TRICK_INDEX_OFFSET] = float(self.state.trick_index)

        return buffer

    def _build_action_mask(self, agent: str) -> np.ndarray:
        mask = self._mask_buffer[agent]
        mask.fill(0)
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
            self._was_dead_step(action)
            return

        if self.phase == "bidding":
            self._step_bidding(agent, action)
            self._clear_rewards()
            return

        if self.phase == "announce":
            self._step_announce(agent, action)
            self._clear_rewards()
            return

        if action is None:
            raise ValueError("action required during play phase")

        card = self._action_to_card(action)
        player = int(agent[1:])
        self.state.play_card(player, card, ruleset=self.ruleset)

        if len(self.state.trick.plays) == 4:
            self._resolve_trick()

        if all(len(hand) == 0 for hand in self.state.hands):
            for a in self.agents:
                self.terminations[a] = True
        else:
            self.agent_selection = f"p{self.state.current_player}"

        self._clear_rewards()

    def _step_bidding(self, agent: str, action: int) -> None:
        if action is None:
            raise ValueError("action required during bidding")
        if action not in range(ACTION_COUNT):
            raise ValueError("invalid action")

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
        self._init_state(leader=self.bidding.starter)
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
        else:
            self._announced_cards[current_player] = []

        self.announcement.index += 1

        if self.announcement.index >= len(self.announcement.order):
            team_a_cards = self._announced_cards.get(0, []) + self._announced_cards.get(2, [])
            team_b_cards = self._announced_cards.get(1, []) + self._announced_cards.get(3, [])
            points_a, points_b, _, _ = resolve_weis(team_a_cards, team_b_cards)

            if points_a:
                self.state.team_points[0] += points_a
                for agent_id in ("p0", "p2"):
                    self.rewards[agent_id] += points_a
            if points_b:
                self.state.team_points[1] += points_b
                for agent_id in ("p1", "p3"):
                    self.rewards[agent_id] += points_b

            self.phase = "play"
            self.agent_selection = f"p{self.state.leader}"
            self.announcement = None
            return

        next_player = self.announcement.order[self.announcement.index]
        self.agent_selection = f"p{next_player}"

    def _resolve_trick(self) -> None:
        led_suit = self.state.trick.led_suit
        cards = self.state.trick.cards
        winning = winning_card(cards, led_suit, self.state.mode, self.state.trump_suit)
        winning_player = next(player for player, card in self.state.trick.plays if card == winning)
        last_trick = self.state.trick_index == 8
        points = trick_points(cards, self.state.mode, self.state.trump_suit, last_trick=last_trick)

        self.state.team_points[self.state.team_index(winning_player)] += points
        if self.state.team_index(winning_player) == 0:
            team_a_players = {"p0", "p2"}
            for agent in self.agents:
                if agent in team_a_players:
                    self.rewards[agent] += points
        else:
            team_b_players = {"p1", "p3"}
            for agent in self.agents:
                if agent in team_b_players:
                    self.rewards[agent] += points

        self.state.completed_tricks.append(
            TrickResult(
                plays=list(self.state.trick.plays),
                winner=winning_player,
                points=points,
                last_trick=last_trick,
            )
        )

        self.state.trick = Trick()
        self.state.trick_index += 1
        self.state.leader = winning_player
        self.agent_selection = f"p{self.state.leader}"

    def _clear_rewards(self) -> None:
        self.last_rewards = dict(self.rewards)
        self._accumulate_rewards()
        for agent in self.agents:
            self.rewards[agent] = 0.0
