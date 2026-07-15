"""Seeded public-information Monte-Carlo search for Schieber.

The policy receives only the canonical observation vector and legal-action
mask.  It samples opponent hands that are consistent with public play history,
then evaluates every legal card with common-world heuristic rollouts.  It never
accepts an environment or ``GameState`` object, which keeps private hands out of
the policy boundary.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass

import numpy as np

from core.cards import ALL_CARDS, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from core.legal_moves import RuleSet, legal_cards
from core.rankings import beats, card_strength, winning_card
from core.ruleset import STANDARD_RULES_PROFILE, RulesetConfig
from core.scoring import card_points, trick_points
from env.jass_aec_env import (
    ACTION_COUNT,
    OBS_ACTOR_OFFSET,
    OBS_BID_CHOOSER_OFFSET,
    OBS_BID_PUSHED_OFFSET,
    OBS_BID_STARTER_OFFSET,
    OBS_BIDDING_ENABLED_OFFSET,
    OBS_CARD_COUNT,
    OBS_CONTRACT_FACTORS_OFFSET,
    OBS_HAND_COUNTS_OFFSET,
    OBS_HAND_OFFSET,
    OBS_LEADER_OFFSET,
    OBS_MATCH_BONUS_OFFSET,
    OBS_MODE_OFFSET,
    OBS_PLAYER_COUNT,
    OBS_SIZE,
    OBS_TEAM_POINTS_OFFSET,
    OBS_TRICK_INDEX_OFFSET,
    OBS_TRUMP_SUIT_OFFSET,
    decode_observation_history,
)
from rl.baselines import StrategicHeuristicPolicy

_VERARDO_BID_WEIGHTS = {
    "J": 7.0,
    "9": 5.0,
    "A": 4.0,
    "K": 2.0,
    "Q": 1.0,
    "10": 0.8,
    "8": 0.5,
    "7": 0.5,
    "6": 0.5,
}


@dataclass(frozen=True)
class PIMCConfig:
    """Deterministic and wall-clock bounds for one policy decision."""

    determinizations: int = 12
    max_rollouts: int = 108
    max_rollout_plies: int = 36
    assignment_node_limit: int = 8_000
    rollout_randomness: float = 0.04
    terminal_win_bonus: float = 0.0
    common_random_numbers: bool = False
    opponent_rollout_policy: str = "generic"
    infer_opponent_bids: bool = False
    prune_bid_constraints: bool = False
    time_budget_ms: float | None = None

    def validate(self) -> None:
        integer_fields = {
            "determinizations": self.determinizations,
            "max_rollouts": self.max_rollouts,
            "max_rollout_plies": self.max_rollout_plies,
            "assignment_node_limit": self.assignment_node_limit,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_rollout_plies > 36:
            raise ValueError("max_rollout_plies cannot exceed 36")
        if not math.isfinite(self.rollout_randomness) or not 0.0 <= self.rollout_randomness <= 1.0:
            raise ValueError("rollout_randomness must be between 0 and 1")
        if not math.isfinite(self.terminal_win_bonus) or self.terminal_win_bonus < 0.0:
            raise ValueError("terminal_win_bonus must be a non-negative finite number")
        if not isinstance(self.common_random_numbers, bool):
            raise ValueError("common_random_numbers must be a boolean")
        if self.opponent_rollout_policy not in {"generic", "random", "verardo"}:
            raise ValueError("opponent_rollout_policy must be 'generic', 'random', or 'verardo'")
        if not isinstance(self.infer_opponent_bids, bool):
            raise ValueError("infer_opponent_bids must be a boolean")
        if self.infer_opponent_bids and self.opponent_rollout_policy != "verardo":
            raise ValueError("infer_opponent_bids requires opponent_rollout_policy='verardo'")
        if not isinstance(self.prune_bid_constraints, bool):
            raise ValueError("prune_bid_constraints must be a boolean")
        if self.prune_bid_constraints and not self.infer_opponent_bids:
            raise ValueError("prune_bid_constraints requires infer_opponent_bids=True")
        if self.time_budget_ms is not None and (
            not math.isfinite(self.time_budget_ms) or self.time_budget_ms <= 0
        ):
            raise ValueError("time_budget_ms must be positive when set")


@dataclass(frozen=True)
class SearchStats:
    attempted_determinizations: int = 0
    successful_determinizations: int = 0
    rollouts: int = 0
    simulated_plies: int = 0
    assignment_nodes: int = 0
    elapsed_seconds: float = 0.0
    used_fallback: bool = False
    action_values: tuple[tuple[int, float, int], ...] = ()


@dataclass(frozen=True)
class _Position:
    hand: tuple[Card, ...]
    history: tuple[tuple[tuple[int, Card], ...], ...]
    current_trick: tuple[tuple[int, Card], ...]
    hand_counts: tuple[int, int, int, int]
    mode: str
    trump_suit: str | None
    leader: int
    trick_index: int
    tricks_won: tuple[int, int]
    team_points: tuple[int, int]
    bid_starter: int | None
    bid_chooser: int | None
    bid_pushed: bool
    contract_factor: int
    match_bonus: int


@dataclass
class _Simulation:
    hands: list[list[Card]]
    current_trick: list[tuple[int, Card]]
    leader: int
    trick_index: int
    tricks_won: list[int]
    initial_points: list[int]
    future_points: list[int]
    mode: str
    trump_suit: str | None
    factor: int
    match_bonus: int


class _AssignmentLimit(RuntimeError):
    pass


def _validated_vector(observation: np.ndarray) -> np.ndarray:
    vector = np.asarray(observation).reshape(-1)
    if vector.shape != (OBS_SIZE,):
        raise ValueError(
            f"canonical observation must contain {OBS_SIZE} values, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("observation must contain only finite values")
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError("canonical observation values must be in [0, 1]")
    return vector


def _legal_actions(action_mask: np.ndarray) -> tuple[int, ...]:
    mask = np.asarray(action_mask).reshape(-1)
    if mask.shape != (ACTION_COUNT,):
        raise ValueError(
            f"canonical action mask must contain {ACTION_COUNT} values, got {mask.shape}"
        )
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("canonical action mask must contain only zeroes and ones")
    legal = tuple(int(value) for value in np.flatnonzero(mask))
    if not legal:
        raise ValueError("search policy called without a legal action")
    return legal


def _one_hot_index(vector: np.ndarray, offset: int, size: int, name: str) -> int:
    indices = np.flatnonzero(vector[offset : offset + size] > 0.5)
    if indices.size != 1:
        raise ValueError(f"observation must encode exactly one {name}")
    return int(indices[0])


def _decode_bounded(value: float, scale: float, name: str) -> int:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"encoded {name} must be in [0, 1)")
    if value == 0.0:
        return 0
    return int(round(scale * value / (1.0 - value)))


def _decode_position(
    vector: np.ndarray,
    legal_actions: tuple[int, ...],
) -> _Position:
    mode_index = _one_hot_index(vector, OBS_MODE_OFFSET, 3, "game mode")
    mode = (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE)[mode_index]
    trump_suit = None
    if mode == MODE_TRUMP:
        trump_suit = SUITS[_one_hot_index(vector, OBS_TRUMP_SUIT_OFFSET, len(SUITS), "trump suit")]

    trick_index = _one_hot_index(vector, OBS_TRICK_INDEX_OFFSET, 10, "trick index")
    if trick_index >= 9:
        raise ValueError("terminal observation cannot have legal card actions")
    leader = _one_hot_index(vector, OBS_LEADER_OFFSET, OBS_PLAYER_COUNT, "leader")
    actor_indices = np.flatnonzero(
        vector[OBS_ACTOR_OFFSET : OBS_ACTOR_OFFSET + OBS_PLAYER_COUNT] > 0.5
    )
    if actor_indices.size and (actor_indices.size != 1 or int(actor_indices[0]) != 0):
        raise ValueError("canonical policy observation must be relative to the acting player")

    hand_indices = np.flatnonzero(vector[OBS_HAND_OFFSET : OBS_HAND_OFFSET + OBS_CARD_COUNT] > 0.5)
    hand = tuple(ALL_CARDS[int(index)] for index in hand_indices)
    card_actions = tuple(action for action in legal_actions if action < OBS_CARD_COUNT)
    if not card_actions or any(ALL_CARDS[action] not in hand for action in card_actions):
        raise ValueError("legal card actions must be present in the observed hand")

    history = tuple(tuple(trick) for trick in decode_observation_history(vector))
    if len(history) not in (trick_index, trick_index + 1):
        raise ValueError("public history is inconsistent with trick_index")
    for trick_number, trick in enumerate(history):
        expected_size = 4 if trick_number < trick_index else len(trick)
        if len(trick) != expected_size or len(trick) > 3 and trick_number == trick_index:
            raise ValueError("public history contains a malformed trick")
        if trick:
            players = [player for player, _ in trick]
            expected = [(players[0] + offset) % 4 for offset in range(len(players))]
            if players != expected:
                raise ValueError("public trick players are out of order")

    current_trick = history[trick_index] if len(history) > trick_index else ()
    if current_trick:
        if current_trick[0][0] != leader:
            raise ValueError("current trick leader does not match observation")
        next_player = (leader + len(current_trick)) % 4
    else:
        next_player = leader
    if next_player != 0:
        raise ValueError("canonical play observation is not at the acting player's turn")

    played = [card for trick in history for _, card in trick]
    if len(set(played)) != len(played) or set(played).intersection(hand):
        raise ValueError("observed cards must be unique")

    decoded_counts = tuple(
        int(round(float(value) * 9.0))
        for value in vector[OBS_HAND_COUNTS_OFFSET : OBS_HAND_COUNTS_OFFSET + OBS_PLAYER_COUNT]
    )
    hand_counts = (
        decoded_counts[0],
        decoded_counts[1],
        decoded_counts[2],
        decoded_counts[3],
    )
    if hand_counts[0] != len(hand):
        raise ValueError("observed hand count does not match encoded hand")
    unseen_count = len(ALL_CARDS) - len(hand) - len(played)
    if any(count < 0 or count > 9 for count in hand_counts) or sum(hand_counts[1:]) != unseen_count:
        raise ValueError("public hand counts are inconsistent with card history")

    tricks_won = [0, 0]
    for trick in history[:trick_index]:
        cards = [card for _, card in trick]
        winner_card = winning_card(cards, cards[0].suit, mode, trump_suit)
        winner = next(player for player, card in trick if card == winner_card)
        tricks_won[winner % 2] += 1

    factor_keys = (*SUITS, MODE_OBEABE, MODE_UNEUFE)
    factor_values = vector[
        OBS_CONTRACT_FACTORS_OFFSET : OBS_CONTRACT_FACTORS_OFFSET + len(factor_keys)
    ]
    key = trump_suit if mode == MODE_TRUMP else mode
    factor = _decode_bounded(float(factor_values[factor_keys.index(key)]), 1.0, "contract factor")
    if factor <= 0:
        raise ValueError("contract factor must be positive")
    match_bonus = _decode_bounded(float(vector[OBS_MATCH_BONUS_OFFSET]), 100.0, "match bonus")
    decoded_team_points = tuple(
        _decode_bounded(
            float(vector[OBS_TEAM_POINTS_OFFSET + team]),
            200.0,
            f"team {team} points",
        )
        for team in range(2)
    )
    bidding_enabled = bool(vector[OBS_BIDDING_ENABLED_OFFSET] > 0.5)
    bid_starter = (
        _one_hot_index(vector, OBS_BID_STARTER_OFFSET, OBS_PLAYER_COUNT, "bid starter")
        if bidding_enabled
        else None
    )
    bid_chooser = (
        _one_hot_index(vector, OBS_BID_CHOOSER_OFFSET, OBS_PLAYER_COUNT, "bid chooser")
        if bidding_enabled
        else None
    )
    bid_pushed = bool(vector[OBS_BID_PUSHED_OFFSET] > 0.5) if bidding_enabled else False

    return _Position(
        hand=hand,
        history=history,
        current_trick=current_trick,
        hand_counts=hand_counts,
        mode=mode,
        trump_suit=trump_suit,
        leader=leader,
        trick_index=trick_index,
        tricks_won=(tricks_won[0], tricks_won[1]),
        team_points=(decoded_team_points[0], decoded_team_points[1]),
        bid_starter=bid_starter,
        bid_chooser=bid_chooser,
        bid_pushed=bid_pushed,
        contract_factor=factor,
        match_bonus=match_bonus,
    )


def _inferred_constraints(
    position: _Position,
    rules: RuleSet,
) -> tuple[list[set[str]], list[set[Card]]]:
    forbidden_suits: list[set[str]] = [set() for _ in range(4)]
    forbidden_cards: list[set[Card]] = [set() for _ in range(4)]
    if not rules.must_follow_suit:
        return forbidden_suits, forbidden_cards

    for trick in position.history:
        if not trick:
            continue
        led_suit = trick[0][1].suit
        for player, card in trick[1:]:
            if card.suit == led_suit:
                continue
            if position.mode != MODE_TRUMP:
                forbidden_suits[player].add(led_suit)
                continue
            assert position.trump_suit is not None
            if led_suit != position.trump_suit:
                # This rules profile permits a player to trump a plain-suit
                # lead even while holding that suit.  Only a plain off-suit
                # discard, or a trump under a profile that forbids voluntary
                # trumping, proves the player is void in the led suit.
                if card.suit != position.trump_suit or not rules.allow_trump_on_non_trump_lead:
                    forbidden_suits[player].add(led_suit)
            elif rules.allow_puur_withhold:
                forbidden_cards[player].update(
                    card_in_deck
                    for card_in_deck in ALL_CARDS
                    if card_in_deck.suit == position.trump_suit and card_in_deck.rank != "J"
                )
            else:
                forbidden_suits[player].add(position.trump_suit)
    return forbidden_suits, forbidden_cards


def _verardo_bid_constraints_satisfied(
    position: _Position,
    remaining_hands: list[list[Card]],
) -> bool:
    """Check public bid evidence for the modeled opponent team.

    Relative seats one and three are the opponent of the acting team.  Their
    original hands are reconstructed from remaining and publicly played cards.
    """

    if (
        position.mode != MODE_TRUMP
        or position.trump_suit is None
        or position.bid_starter is None
        or position.bid_chooser is None
    ):
        return True

    initial_hands = [list(hand) for hand in remaining_hands]
    for trick in position.history:
        for player, card in trick:
            initial_hands[player].append(card)

    if position.bid_starter % 2 == 1:
        starter_cards = initial_hands[position.bid_starter]
        has_stopper = _has_verardo_push_stopper(starter_cards)
        if position.bid_pushed == has_stopper:
            return False

    if position.bid_chooser % 2 == 1:
        chooser_cards = initial_hands[position.bid_chooser]
        scores = {
            suit: sum(
                _VERARDO_BID_WEIGHTS[card.rank] for card in chooser_cards if card.suit == suit
            )
            for suit in SUITS
        }
        chosen = max(SUITS, key=lambda suit: (scores[suit], -SUITS.index(suit)))
        if chosen != position.trump_suit:
            return False

    return True


def _has_verardo_push_stopper(cards: list[Card]) -> bool:
    for suit in SUITS:
        suit_cards = [card for card in cards if card.suit == suit]
        ranks = {card.rank for card in suit_cards}
        if ("J" in ranks and len(suit_cards) >= 3) or ("9" in ranks and len(suit_cards) >= 4):
            return True
    return False


def _sample_hands(
    position: _Position,
    rules: RuleSet,
    rng: random.Random,
    node_limit: int,
    apply_verardo_bid_constraints: bool = False,
    prune_verardo_bid_constraints: bool = False,
) -> tuple[list[list[Card]] | None, int]:
    known = set(position.hand) | {card for trick in position.history for _, card in trick}
    unassigned = [card for card in ALL_CARDS if card not in known]
    capacities = list(position.hand_counts)
    capacities[0] = 0
    hands: list[list[Card]] = [list(position.hand), [], [], []]
    forbidden_suits, forbidden_cards = _inferred_constraints(position, rules)
    played_hands: list[list[Card]] = [[], [], [], []]
    for trick in position.history:
        for player, card in trick:
            played_hands[player].append(card)
    nodes = 0

    def allowed(card: Card, player: int) -> bool:
        return card.suit not in forbidden_suits[player] and card not in forbidden_cards[player]

    def feasible(cards: list[Card]) -> bool:
        return all(
            sum(allowed(card, player) for card in cards) >= capacities[player]
            for player in (1, 2, 3)
        )

    def bid_constraints_still_feasible(cards: list[Card]) -> bool:
        if not apply_verardo_bid_constraints or not prune_verardo_bid_constraints:
            return True
        if (
            position.mode != MODE_TRUMP
            or position.trump_suit is None
            or position.bid_starter is None
            or position.bid_chooser is None
        ):
            return True

        starter = position.bid_starter
        if starter % 2 == 1:
            starter_cards = played_hands[starter] + hands[starter]
            has_stopper = _has_verardo_push_stopper(starter_cards)
            if position.bid_pushed and has_stopper:
                return False
            if not position.bid_pushed and not has_stopper:
                slots = capacities[starter]
                possible_stopper = False
                available = [card for card in cards if allowed(card, starter)]
                for suit in SUITS:
                    current_suit = [card for card in starter_cards if card.suit == suit]
                    available_suit = [card for card in available if card.suit == suit]
                    ranks = {card.rank for card in current_suit}
                    available_ranks = {card.rank for card in available_suit}
                    maximum_count = len(current_suit) + min(slots, len(available_suit))
                    if (("J" in ranks or "J" in available_ranks) and maximum_count >= 3) or (
                        ("9" in ranks or "9" in available_ranks) and maximum_count >= 4
                    ):
                        possible_stopper = True
                        break
                if not possible_stopper:
                    return False

        chooser = position.bid_chooser
        if chooser % 2 == 1:
            chooser_cards = played_hands[chooser] + hands[chooser]
            current_scores = {
                suit: sum(
                    _VERARDO_BID_WEIGHTS[card.rank] for card in chooser_cards if card.suit == suit
                )
                for suit in SUITS
            }
            slots = capacities[chooser]
            available_trump_weights = sorted(
                (
                    _VERARDO_BID_WEIGHTS[card.rank]
                    for card in cards
                    if card.suit == position.trump_suit and allowed(card, chooser)
                ),
                reverse=True,
            )
            maximum_trump_score = current_scores[position.trump_suit] + sum(
                available_trump_weights[:slots]
            )
            trump_index = SUITS.index(position.trump_suit)
            for suit in SUITS:
                if suit == position.trump_suit:
                    continue
                if current_scores[suit] > maximum_trump_score or (
                    current_scores[suit] == maximum_trump_score and SUITS.index(suit) < trump_index
                ):
                    return False

        return True

    def assign(cards: list[Card]) -> bool:
        nonlocal nodes
        if nodes >= node_limit:
            raise _AssignmentLimit
        nodes += 1
        if not cards:
            return capacities[1:] == [0, 0, 0] and (
                not apply_verardo_bid_constraints
                or _verardo_bid_constraints_satisfied(position, hands)
            )

        candidates: list[tuple[int, int]] = []
        for index, card in enumerate(cards):
            count = sum(capacities[player] > 0 and allowed(card, player) for player in (1, 2, 3))
            candidates.append((count, index))
        minimum = min(count for count, _ in candidates)
        tied = [index for count, index in candidates if count == minimum]
        card_index = tied[rng.randrange(len(tied))]
        card = cards.pop(card_index)
        players = [
            player for player in (1, 2, 3) if capacities[player] > 0 and allowed(card, player)
        ]
        rng.shuffle(players)
        players.sort(key=lambda player: capacities[player], reverse=True)
        for player in players:
            capacities[player] -= 1
            hands[player].append(card)
            if feasible(cards) and bid_constraints_still_feasible(cards) and assign(cards):
                return True
            hands[player].pop()
            capacities[player] += 1
        cards.insert(card_index, card)
        return False

    try:
        success = assign(unassigned)
    except _AssignmentLimit:
        success = False
    return (hands if success else None), nodes


def _current_winner(
    trick: list[tuple[int, Card]], mode: str, trump_suit: str | None
) -> tuple[int, Card]:
    cards = [card for _, card in trick]
    winner_card = winning_card(cards, cards[0].suit, mode, trump_suit)
    return next((player, card) for player, card in trick if card == winner_card)


def _weakest(cards: list[Card], mode: str, trump_suit: str | None, led_suit: str) -> Card:
    return min(
        cards,
        key=lambda card: (
            card_points(card, mode, trump_suit),
            card_strength(card, led_suit, mode, trump_suit),
            ALL_CARDS.index(card),
        ),
    )


def _rollout_card(
    simulation: _Simulation,
    player: int,
    legal: list[Card],
    rng: random.Random,
    randomness: float,
) -> Card:
    if len(legal) == 1:
        return legal[0]
    if randomness and rng.random() < randomness:
        return legal[rng.randrange(len(legal))]

    if simulation.current_trick:
        led_suit = simulation.current_trick[0][1].suit
        winner, winner_card = _current_winner(
            simulation.current_trick, simulation.mode, simulation.trump_suit
        )
        winners = [
            card
            for card in legal
            if beats(
                card,
                winner_card,
                led_suit,
                simulation.mode,
                simulation.trump_suit,
            )
        ]
        non_winners = [card for card in legal if card not in winners]
        if winner == (player + 2) % 4 and non_winners:
            if len(simulation.current_trick) == 3:
                return max(
                    non_winners,
                    key=lambda card: (
                        card_points(card, simulation.mode, simulation.trump_suit),
                        -ALL_CARDS.index(card),
                    ),
                )
            return _weakest(non_winners, simulation.mode, simulation.trump_suit, led_suit)
        if winners:
            return min(
                winners,
                key=lambda card: (
                    card_strength(card, led_suit, simulation.mode, simulation.trump_suit),
                    card_points(card, simulation.mode, simulation.trump_suit),
                    ALL_CARDS.index(card),
                ),
            )
        return _weakest(legal, simulation.mode, simulation.trump_suit, led_suit)

    masters = []
    for card in legal:
        opponent_cards = simulation.hands[(player + 1) % 4] + simulation.hands[(player + 3) % 4]
        if not any(
            beats(
                opponent,
                card,
                card.suit,
                simulation.mode,
                simulation.trump_suit,
            )
            for opponent in opponent_cards
        ):
            masters.append(card)
    if masters:
        return max(
            masters,
            key=lambda card: (
                card_points(card, simulation.mode, simulation.trump_suit),
                card_strength(card, card.suit, simulation.mode, simulation.trump_suit),
                -ALL_CARDS.index(card),
            ),
        )

    suit_lengths = {
        suit: sum(card.suit == suit for card in simulation.hands[player]) for suit in SUITS
    }
    lead_suit = max(
        {card.suit for card in legal},
        key=lambda suit: (
            suit_lengths[suit],
            suit != simulation.trump_suit,
            -SUITS.index(suit),
        ),
    )
    return _weakest(
        [card for card in legal if card.suit == lead_suit],
        simulation.mode,
        simulation.trump_suit,
        lead_suit,
    )


def _verardo_opponent_card(
    simulation: _Simulation,
    player: int,
    legal: list[Card],
) -> Card:
    """Mirror the documented deterministic Verardo card policy in a rollout.

    The model is intentionally available only for trump contracts.  Callers
    retain the generic rollout for Obeabe and Uneufe, which are outside the
    pinned external benchmark.
    """

    if simulation.mode != MODE_TRUMP or simulation.trump_suit is None:
        raise ValueError("Verardo opponent rollouts require a trump contract")
    if not legal:
        raise ValueError("Verardo opponent rollout requires a legal card")
    if len(legal) == 1:
        return legal[0]

    trump_suit = simulation.trump_suit

    def strength(card: Card, led_suit: str) -> int:
        return card_strength(card, led_suit, MODE_TRUMP, trump_suit)

    def highest_strength(cards: list[Card], led_suit: str) -> Card:
        return max(cards, key=lambda card: (strength(card, led_suit), -ALL_CARDS.index(card)))

    def lowest_strength(cards: list[Card], led_suit: str) -> Card:
        return min(cards, key=lambda card: (strength(card, led_suit), ALL_CARDS.index(card)))

    def highest_value(cards: list[Card]) -> Card:
        return max(
            cards,
            key=lambda card: (
                card_points(card, MODE_TRUMP, trump_suit),
                -ALL_CARDS.index(card),
            ),
        )

    def lowest_value(cards: list[Card]) -> Card:
        return min(
            cards,
            key=lambda card: (
                card_points(card, MODE_TRUMP, trump_suit),
                ALL_CARDS.index(card),
            ),
        )

    trumps = [card for card in legal if card.suit == trump_suit]
    if not simulation.current_trick:
        for rank in ("J", "9"):
            matching = [card for card in trumps if card.rank == rank]
            if matching:
                return matching[0]
        non_trumps = [card for card in legal if card.suit != trump_suit]
        if non_trumps:
            return highest_value(non_trumps)
        return highest_strength(trumps, trump_suit)

    led_suit = simulation.current_trick[0][1].suit
    if led_suit == trump_suit:
        return highest_strength(trumps, trump_suit) if trumps else lowest_value(legal)

    follows = [card for card in legal if card.suit == led_suit]
    winner, _ = _current_winner(
        simulation.current_trick,
        simulation.mode,
        simulation.trump_suit,
    )
    partner_winning = winner == (player + 2) % 4
    valuable_trick = any(card.rank in {"A", "10"} for _, card in simulation.current_trick)
    if follows:
        if not partner_winning and valuable_trick and trumps:
            return lowest_strength(trumps, trump_suit)
        return highest_strength(follows, led_suit)
    if not partner_winning and trumps:
        return lowest_strength(trumps, trump_suit)

    discards = [card for card in legal if card.suit != trump_suit]
    return lowest_value(discards or legal)


def _play(simulation: _Simulation, player: int, card: Card) -> None:
    simulation.hands[player].remove(card)
    simulation.current_trick.append((player, card))
    if len(simulation.current_trick) < 4:
        return

    winner, _ = _current_winner(simulation.current_trick, simulation.mode, simulation.trump_suit)
    cards = [played for _, played in simulation.current_trick]
    points = trick_points(
        cards,
        simulation.mode,
        simulation.trump_suit,
        last_trick=simulation.trick_index == 8,
    )
    team = winner % 2
    simulation.future_points[team] += points
    simulation.tricks_won[team] += 1
    simulation.leader = winner
    simulation.trick_index += 1
    simulation.current_trick = []


def _utility(simulation: _Simulation, terminal_win_bonus: float = 0.0) -> float:
    points = list(simulation.future_points)
    if simulation.trick_index == 9:
        for team, tricks in enumerate(simulation.tricks_won):
            if tricks == 9:
                points[team] += simulation.match_bonus
    margin = float(
        simulation.initial_points[0]
        - simulation.initial_points[1]
        + (points[0] - points[1]) * simulation.factor
    )
    if simulation.trick_index == 9 and terminal_win_bonus:
        if margin > 0.0:
            margin += terminal_win_bonus
        elif margin < 0.0:
            margin -= terminal_win_bonus
    return margin


def _evaluate_action(
    position: _Position,
    sampled_hands: list[list[Card]],
    action: int,
    rules: RuleSet,
    config: PIMCConfig,
    rng: random.Random,
) -> tuple[float, int]:
    simulation = _Simulation(
        hands=[list(hand) for hand in sampled_hands],
        current_trick=list(position.current_trick),
        leader=position.leader,
        trick_index=position.trick_index,
        tricks_won=list(position.tricks_won),
        initial_points=list(position.team_points),
        future_points=[0, 0],
        mode=position.mode,
        trump_suit=position.trump_suit,
        factor=position.contract_factor,
        match_bonus=position.match_bonus,
    )
    _play(simulation, 0, ALL_CARDS[action])
    plies = 1
    while simulation.trick_index < 9 and plies < config.max_rollout_plies:
        player = (simulation.leader + len(simulation.current_trick)) % 4
        winner = None
        if simulation.current_trick:
            winner, _ = _current_winner(
                simulation.current_trick,
                simulation.mode,
                simulation.trump_suit,
            )
        legal = legal_cards(
            simulation.hands[player],
            [card for _, card in simulation.current_trick],
            simulation.mode,
            simulation.trump_suit,
            partner_is_winning=winner == (player + 2) % 4,
            ruleset=rules,
        )
        if not legal:
            raise RuntimeError("rollout reached a non-terminal state without a legal card")
        if player % 2 == 1 and config.opponent_rollout_policy == "random":
            card = legal[rng.randrange(len(legal))]
        elif (
            player % 2 == 1
            and config.opponent_rollout_policy == "verardo"
            and simulation.mode == MODE_TRUMP
        ):
            card = _verardo_opponent_card(simulation, player, legal)
        else:
            card = _rollout_card(
                simulation,
                player,
                legal,
                rng,
                config.rollout_randomness,
            )
        _play(simulation, player, card)
        plies += 1
    return _utility(simulation, config.terminal_win_bonus), plies


class PIMCSearchPolicy:
    """Deterministic public-information search with seeded hand sampling."""

    def __init__(
        self,
        *,
        seed: int = 0,
        config: PIMCConfig | None = None,
        profile: RulesetConfig | None = None,
    ) -> None:
        self.config = config or PIMCConfig()
        self.config.validate()
        self.profile = profile or STANDARD_RULES_PROFILE
        self._fallback = StrategicHeuristicPolicy()
        self._seed = 0
        self.last_stats = SearchStats()
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            self._seed = seed
        self.last_stats = SearchStats()

    def _decision_rng(self, vector: np.ndarray, mask: np.ndarray) -> random.Random:
        digest = hashlib.blake2b(digest_size=16, person=b"jass-pimc-v1")
        digest.update(str(self._seed).encode("ascii"))
        digest.update(np.asarray(vector, dtype=np.float32).tobytes())
        digest.update(np.asarray(mask, dtype=np.int8).tobytes())
        return random.Random(int.from_bytes(digest.digest(), "big"))

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int:
        del agent  # canonical observations are already acting-seat-relative
        started = time.perf_counter()
        vector = _validated_vector(observation)
        legal = _legal_actions(action_mask)
        if len(legal) == 1:
            self.last_stats = SearchStats(elapsed_seconds=time.perf_counter() - started)
            return legal[0]

        card_actions = tuple(action for action in legal if action < OBS_CARD_COUNT)
        if len(card_actions) != len(legal):
            action = self._fallback(vector, np.asarray(action_mask), "p0")
            self.last_stats = SearchStats(
                elapsed_seconds=time.perf_counter() - started,
                used_fallback=True,
            )
            return action

        position = _decode_position(vector, legal)
        worlds = min(
            self.config.determinizations,
            self.config.max_rollouts // len(card_actions),
        )
        if worlds <= 0:
            action = self._fallback(vector, np.asarray(action_mask), "p0")
            self.last_stats = SearchStats(
                elapsed_seconds=time.perf_counter() - started,
                used_fallback=True,
            )
            return action

        rng = self._decision_rng(vector, np.asarray(action_mask))
        totals = {action: 0.0 for action in card_actions}
        visits = {action: 0 for action in card_actions}
        attempted = successful = rollouts = plies = assignment_nodes = 0
        deadline = (
            started + self.config.time_budget_ms / 1000.0
            if self.config.time_budget_ms is not None
            else None
        )

        for _ in range(worlds):
            if deadline is not None and successful and time.perf_counter() >= deadline:
                break
            attempted += 1
            sampled_hands, nodes = _sample_hands(
                position,
                self.profile.legal_moves,
                rng,
                self.config.assignment_node_limit,
                apply_verardo_bid_constraints=self.config.infer_opponent_bids,
                prune_verardo_bid_constraints=self.config.prune_bid_constraints,
            )
            assignment_nodes += nodes
            if sampled_hands is None:
                continue
            successful += 1
            if self.config.common_random_numbers:
                shared_seed = rng.getrandbits(64)
                rollout_seeds = {action: shared_seed for action in card_actions}
            else:
                rollout_seeds = {action: rng.getrandbits(64) for action in card_actions}
            for action in card_actions:
                value, used_plies = _evaluate_action(
                    position,
                    sampled_hands,
                    action,
                    self.profile.legal_moves,
                    self.config,
                    random.Random(rollout_seeds[action]),
                )
                totals[action] += value
                visits[action] += 1
                rollouts += 1
                plies += used_plies

        if not successful:
            chosen = self._fallback(vector, np.asarray(action_mask), "p0")
            used_fallback = True
        else:
            fallback_choice = self._fallback(vector, np.asarray(action_mask), "p0")
            chosen = max(
                card_actions,
                key=lambda action: (
                    totals[action] / visits[action],
                    action == fallback_choice,
                    -action,
                ),
            )
            used_fallback = False

        values = tuple(
            (
                action,
                totals[action] / visits[action] if visits[action] else float("-inf"),
                visits[action],
            )
            for action in card_actions
        )
        self.last_stats = SearchStats(
            attempted_determinizations=attempted,
            successful_determinizations=successful,
            rollouts=rollouts,
            simulated_plies=plies,
            assignment_nodes=assignment_nodes,
            elapsed_seconds=time.perf_counter() - started,
            used_fallback=used_fallback,
            action_values=values,
        )
        if chosen not in legal:
            raise RuntimeError("search selected an action outside the legal mask")
        return chosen


def search_policy(
    *,
    seed: int = 0,
    config: PIMCConfig | None = None,
    profile: RulesetConfig | None = None,
) -> PIMCSearchPolicy:
    return PIMCSearchPolicy(seed=seed, config=config, profile=profile)


__all__ = [
    "PIMCConfig",
    "PIMCSearchPolicy",
    "SearchStats",
    "search_policy",
]
