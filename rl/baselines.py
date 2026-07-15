"""Public-information baseline policies for Jass evaluation.

The policies in this module deliberately receive only an observation, its legal
action mask, and the acting seat.  They therefore cannot inspect another
player's hand or any other private environment state.  :func:`as_aec_policy`
adapts this stricter interface to the ``(env, agent)`` callback used by the
single-agent wrappers.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from typing import Protocol, TypeAlias

import numpy as np

from core.announcements.weis import find_weis
from core.cards import ALL_CARDS, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from core.rankings import beats, card_strength, winning_card
from core.scoring import card_points
from env.jass_aec_env import (
    ACTION_COUNT,
    ANNOUNCE_ACTION,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    BIDDING_TRUMP_ACTIONS,
    BIDDING_UNEUFE_ACTION,
    OBS_HAND_OFFSET,
    OBS_MODE_OFFSET,
    OBS_TRICK_COUNT,
    OBS_TRICK_INDEX_OFFSET,
    OBS_TRUMP_SUIT_OFFSET,
    PASS_ACTION,
    decode_current_trick,
    decode_observation_history,
    decode_played_cards,
)


class PublicPolicy(Protocol):
    """A policy that can use only information exposed to the acting player."""

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int: ...


class ObservableEnv(Protocol):
    def observe(self, agent: str) -> dict[str, np.ndarray]: ...


AECPolicy: TypeAlias = Callable[[ObservableEnv, str], int]


def _legal_actions(action_mask: np.ndarray) -> tuple[int, ...]:
    mask = np.asarray(action_mask).reshape(-1)
    if mask.size != ACTION_COUNT:
        raise ValueError(f"action mask must contain {ACTION_COUNT} entries, got {mask.size}")
    legal = tuple(int(action) for action in np.flatnonzero(mask))
    if not legal:
        raise ValueError("policy called without a legal action")
    return legal


def _cards_from_slice(
    observation: np.ndarray, offset: int, size: int = len(ALL_CARDS)
) -> list[Card]:
    values = np.asarray(observation).reshape(-1)[offset : offset + size]
    return [ALL_CARDS[index] for index in np.flatnonzero(values > 0.5)]


def _hand(observation: np.ndarray) -> list[Card]:
    return _cards_from_slice(observation, OBS_HAND_OFFSET)


def _mode_and_trump(observation: np.ndarray) -> tuple[str, str | None]:
    values = np.asarray(observation).reshape(-1)
    mode_values = values[OBS_MODE_OFFSET : OBS_MODE_OFFSET + 3]
    if mode_values.size != 3 or not np.any(mode_values > 0.5):
        raise ValueError("play observation does not identify a game mode")
    mode = (MODE_TRUMP, MODE_OBEABE, MODE_UNEUFE)[int(np.argmax(mode_values))]

    trump_suit: str | None = None
    if mode == MODE_TRUMP:
        suit_values = values[OBS_TRUMP_SUIT_OFFSET : OBS_TRUMP_SUIT_OFFSET + len(SUITS)]
        if suit_values.size != len(SUITS) or not np.any(suit_values > 0.5):
            raise ValueError("trump observation does not identify the trump suit")
        trump_suit = SUITS[int(np.argmax(suit_values))]
    return mode, trump_suit


def _trick_index(observation: np.ndarray) -> int:
    values = np.asarray(observation).reshape(-1)
    encoded = values[OBS_TRICK_INDEX_OFFSET : OBS_TRICK_INDEX_OFFSET + 10]
    present = np.flatnonzero(encoded > 0.5)
    return int(present[0]) if present.size else 0


def _current_trick(observation: np.ndarray) -> tuple[list[Card], list[int]]:
    trick_index = _trick_index(observation)
    if trick_index >= OBS_TRICK_COUNT:
        return [], []
    cards = decode_current_trick(observation)
    history = decode_observation_history(observation)
    players = (
        [relative_player for relative_player, _ in history[trick_index]]
        if trick_index < len(history)
        else []
    )
    return cards, players


def _played_cards(observation: np.ndarray) -> set[Card]:
    return decode_played_cards(observation)


class SeededRandomPolicy:
    """Uniform random legal play backed by a reproducibly seeded RNG."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset the stream, which lets paired evaluations share policy noise."""

        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int:
        del observation, agent
        legal = _legal_actions(action_mask)
        return legal[self._rng.randrange(len(legal))]


_TRUMP_WEIGHTS = {
    "J": 15.0,
    "9": 11.0,
    "A": 7.0,
    "K": 4.0,
    "Q": 3.0,
    "10": 2.0,
    "8": 1.0,
    "7": 0.5,
    "6": 0.0,
}
_OBEABE_WEIGHTS = {
    "A": 10.0,
    "K": 6.0,
    "Q": 3.5,
    "J": 2.5,
    "10": 2.0,
    "9": 0.5,
    "8": 0.5,
    "7": 0.0,
    "6": 0.0,
}
_UNEUFE_WEIGHTS = {
    "6": 10.0,
    "7": 6.0,
    "8": 3.5,
    "9": 2.5,
    "10": 2.0,
    "J": 0.5,
    "Q": 0.5,
    "K": 0.0,
    "A": 0.0,
}


def _suit_control_score(cards: list[Card], weights: dict[str, float]) -> float:
    by_suit = {suit: {card.rank for card in cards if card.suit == suit} for suit in SUITS}
    score = sum(weights[card.rank] for card in cards)
    # Consecutive top cards are substantially safer than isolated honours.
    for ranks in by_suit.values():
        ordered = sorted((weights[rank] for rank in ranks), reverse=True)
        if len(ordered) >= 2 and ordered[1] >= 6.0:
            score += 2.0
        if len(ordered) >= 3 and ordered[2] >= 3.5:
            score += 2.0
    return score


def bid_scores(observation: np.ndarray) -> dict[int, float]:
    """Score every non-push bid using only the encoded hand.

    Scores are intended for deterministic ranking, not as predicted game points.
    """

    cards = _hand(observation)
    scores: dict[int, float] = {}
    for action, suit in BIDDING_TRUMP_ACTIONS.items():
        trumps = [card for card in cards if card.suit == suit]
        ranks = {card.rank for card in trumps}
        length = len(trumps)
        score = sum(_TRUMP_WEIGHTS[card.rank] for card in trumps)
        score += max(0, length - 2) * 4.0
        if {"J", "9"}.issubset(ranks):
            score += 6.0
        # Side aces and protected tens remain useful after drawing trump.
        for side_suit in SUITS:
            if side_suit == suit:
                continue
            side_ranks = {card.rank for card in cards if card.suit == side_suit}
            if "A" in side_ranks:
                score += 3.0
                if "10" in side_ranks:
                    score += 1.0
        scores[action] = score

    scores[BIDDING_OBEABE_ACTION] = _suit_control_score(cards, _OBEABE_WEIGHTS)
    scores[BIDDING_UNEUFE_ACTION] = _suit_control_score(cards, _UNEUFE_WEIGHTS)
    return scores


def _weakest(cards: list[Card], mode: str, trump_suit: str | None, led_suit: str) -> Card:
    return min(
        cards,
        key=lambda card: (
            card_points(card, mode, trump_suit),
            card_strength(card, led_suit, mode, trump_suit),
            ALL_CARDS.index(card),
        ),
    )


def _play_action(observation: np.ndarray, legal_actions: tuple[int, ...]) -> int:
    legal_cards = [ALL_CARDS[action] for action in legal_actions]
    mode, trump_suit = _mode_and_trump(observation)
    trick, trick_players = _current_trick(observation)

    if trick:
        led_suit = trick[0].suit
        current_winner = winning_card(trick, led_suit, mode, trump_suit)
        winning_cards = [
            card for card in legal_cards if beats(card, current_winner, led_suit, mode, trump_suit)
        ]

        partner_winning = False
        winning_slot = trick.index(current_winner)
        if winning_slot < len(trick_players):
            # Player identities in schema v2 are acting-seat-relative:
            # self=0, left=1, partner=2, right=3.
            partner_winning = trick_players[winning_slot] == 2

        non_winners = [card for card in legal_cards if card not in winning_cards]
        if partner_winning and non_winners:
            if len(trick) == 3:
                # The partner has secured the trick: smear the most points without
                # unnecessarily overtaking them.
                chosen = max(
                    non_winners,
                    key=lambda card: (
                        card_points(card, mode, trump_suit),
                        -card_strength(card, led_suit, mode, trump_suit)[1],
                        -ALL_CARDS.index(card),
                    ),
                )
            else:
                chosen = _weakest(non_winners, mode, trump_suit, led_suit)
        elif winning_cards:
            # Spend the cheapest card that currently takes the trick.
            chosen = min(
                winning_cards,
                key=lambda card: (
                    card_strength(card, led_suit, mode, trump_suit),
                    card_points(card, mode, trump_suit),
                    ALL_CARDS.index(card),
                ),
            )
        else:
            chosen = _weakest(legal_cards, mode, trump_suit, led_suit)
        return ALL_CARDS.index(chosen)

    hand = _hand(observation)
    known = set(hand) | _played_cards(observation)
    unseen = [card for card in ALL_CARDS if card not in known]

    masters: list[Card] = []
    for card in legal_cards:
        if not any(beats(other, card, card.suit, mode, trump_suit) for other in unseen):
            masters.append(card)
    if masters:
        # Cash a guaranteed winner, preferring a valuable card before opponents
        # can become void in its suit.
        chosen = max(
            masters,
            key=lambda card: (
                card_points(card, mode, trump_suit),
                card_strength(card, card.suit, mode, trump_suit),
                -ALL_CARDS.index(card),
            ),
        )
        return ALL_CARDS.index(chosen)

    suit_lengths = Counter(card.suit for card in hand)
    candidate_suit = max(
        (card.suit for card in legal_cards),
        key=lambda suit: (
            suit_lengths[suit],
            suit != trump_suit,
            -SUITS.index(suit),
        ),
    )
    suit_cards = [card for card in legal_cards if card.suit == candidate_suit]
    chosen = _weakest(suit_cards, mode, trump_suit, candidate_suit)
    return ALL_CARDS.index(chosen)


class StrategicHeuristicPolicy:
    """Deterministic bidding, Weis, and trick-play baseline.

    The policy values suit strength while bidding, announces only a real Weis,
    preserves a partner's winning card, smears points into secured partner
    tricks, takes opponent tricks with the cheapest winner, and cashes known
    master cards when leading.
    """

    def __init__(self, push_threshold: float = 30.0) -> None:
        self.push_threshold = push_threshold

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int:
        del agent
        legal = _legal_actions(action_mask)

        if ANNOUNCE_ACTION in legal or PASS_ACTION in legal:
            if ANNOUNCE_ACTION in legal and find_weis(_hand(observation)):
                return ANNOUNCE_ACTION
            if PASS_ACTION in legal:
                return PASS_ACTION
            return ANNOUNCE_ACTION

        bidding_actions = [action for action in legal if action >= len(ALL_CARDS)]
        if bidding_actions:
            scores = bid_scores(observation)
            choices = [action for action in bidding_actions if action in scores]
            if not choices:
                return legal[0]
            best = max(choices, key=lambda action: (scores[action], -action))
            if BIDDING_PUSH_ACTION in legal and scores[best] < self.push_threshold:
                return BIDDING_PUSH_ACTION
            return best

        return _play_action(observation, legal)


def as_aec_policy(policy: PublicPolicy) -> AECPolicy:
    """Adapt a public policy to the callback accepted by existing wrappers."""

    def _policy(env: ObservableEnv, agent: str) -> int:
        visible = env.observe(agent)
        return int(policy(visible["observation"], visible["action_mask"], agent))

    return _policy


def random_policy(seed: int = 0) -> SeededRandomPolicy:
    return SeededRandomPolicy(seed)


def strategic_policy(push_threshold: float = 30.0) -> StrategicHeuristicPolicy:
    return StrategicHeuristicPolicy(push_threshold=push_threshold)


__all__ = [
    "PublicPolicy",
    "SeededRandomPolicy",
    "StrategicHeuristicPolicy",
    "as_aec_policy",
    "bid_scores",
    "random_policy",
    "strategic_policy",
]
