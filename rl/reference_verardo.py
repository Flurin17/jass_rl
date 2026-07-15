"""Clean-room reference for the published Verardo Jass policy summary.

This module intentionally implements only the publicly described behaviour.  It
does not contain or depend on code from the unlicensed reference repository.
The policy consumes the canonical v2 observation and action mask, just like a
trained public-information policy used by :mod:`rl.tournament`.

The summary leaves ties unspecified.  This implementation resolves equal bid
scores and equal card values by choosing the lowest legal action index.  It
also treats an Ace or Ten anywhere in the current trick as a valuable card to
steal.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

from core.cards import ALL_CARDS, MODE_OBEABE, MODE_TRUMP, MODE_UNEUFE, SUITS, Card
from core.legal_moves import RuleSet
from core.rankings import card_strength, winning_card
from core.ruleset import RulesetConfig
from core.scoring import card_points
from env.jass_aec_env import (
    ACTION_COUNT,
    ANNOUNCE_ACTION,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    BIDDING_TRUMP_ACTIONS,
    BIDDING_UNEUFE_ACTION,
    OBS_CARD_COUNT,
    OBS_HAND_OFFSET,
    OBS_MODE_OFFSET,
    OBS_SIZE,
    OBS_TRICK_INDEX_OFFSET,
    OBS_TRUMP_SUIT_OFFSET,
    PASS_ACTION,
    decode_observation_history,
    trump_only_action_mask,
)
from rl.baselines import PublicPolicy
from rl.privileged_observation import CTDE_OBS_SIZE
from rl.tournament import (
    TournamentConfig,
    TournamentReport,
    maskable_model_policy,
    run_tournament,
)

VERARDO_REFERENCE_COMMIT = "4ec7c665ce740fa1d792f653875a63f755551e7c"
VERARDO_PROFILE_VERSION = "verardo-v1"
VERARDO_QUALIFICATION_GAMES = 4_000

VERARDO_V1_PROFILE = RulesetConfig(
    allow_stock=False,
    allow_weis=False,
    legal_moves=RuleSet(must_overtrump_when_only_trumps=True),
    version=VERARDO_PROFILE_VERSION,
    contract_factors={
        **{suit: 1 for suit in SUITS},
        MODE_OBEABE: 1,
        MODE_UNEUFE: 1,
    },
    match_bonus=100,
)

VERARDO_BID_WEIGHTS = MappingProxyType(
    {
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
)

_VALUABLE_RANKS = frozenset(("A", "10"))
_BIDDING_ACTIONS = frozenset(BIDDING_TRUMP_ACTIONS) | {
    BIDDING_OBEABE_ACTION,
    BIDDING_UNEUFE_ACTION,
    BIDDING_PUSH_ACTION,
}


def _observation_vector(observation: np.ndarray) -> np.ndarray:
    vector = np.asarray(observation).reshape(-1)
    if vector.shape != (OBS_SIZE,):
        raise ValueError(
            f"canonical v2 observation must contain {OBS_SIZE} values, "
            f"got {np.asarray(observation).shape}"
        )
    return vector


def _legal_actions(action_mask: np.ndarray) -> tuple[int, ...]:
    mask = np.asarray(action_mask).reshape(-1)
    if mask.shape != (ACTION_COUNT,):
        raise ValueError(
            f"canonical action mask must contain {ACTION_COUNT} values, "
            f"got {np.asarray(action_mask).shape}"
        )
    legal = tuple(int(action) for action in np.flatnonzero(mask))
    if not legal:
        raise ValueError("Verardo reference called without a legal action")
    return legal


def _hand(vector: np.ndarray) -> tuple[Card, ...]:
    encoded = vector[OBS_HAND_OFFSET : OBS_HAND_OFFSET + OBS_CARD_COUNT]
    return tuple(ALL_CARDS[int(index)] for index in np.flatnonzero(encoded > 0.5))


def _card_action(card: Card) -> int:
    return ALL_CARDS.index(card)


def verardo_bid_scores(observation: np.ndarray) -> dict[str, float]:
    """Return the published weighted hand score for each trump suit."""

    cards = _hand(_observation_vector(observation))
    return {
        suit: sum(VERARDO_BID_WEIGHTS[card.rank] for card in cards if card.suit == suit)
        for suit in SUITS
    }


def _has_push_stopper(cards: Sequence[Card]) -> bool:
    for suit in SUITS:
        suit_cards = tuple(card for card in cards if card.suit == suit)
        ranks = {card.rank for card in suit_cards}
        if ("J" in ranks and len(suit_cards) >= 3) or (
            "9" in ranks and len(suit_cards) >= 4
        ):
            return True
    return False


def _bid_action(
    vector: np.ndarray,
    legal_actions: tuple[int, ...],
) -> int:
    choices = tuple(
        (action, suit)
        for action, suit in BIDDING_TRUMP_ACTIONS.items()
        if action in legal_actions
    )
    if not choices:
        raise ValueError("Verardo reference has no legal trump bid")

    cards = _hand(vector)
    if BIDDING_PUSH_ACTION in legal_actions and not _has_push_stopper(cards):
        return BIDDING_PUSH_ACTION

    scores = {
        suit: sum(VERARDO_BID_WEIGHTS[card.rank] for card in cards if card.suit == suit)
        for suit in SUITS
    }
    return max(choices, key=lambda choice: (scores[choice[1]], -choice[0]))[0]


def _trump_suit(vector: np.ndarray) -> str:
    modes = np.flatnonzero(vector[OBS_MODE_OFFSET : OBS_MODE_OFFSET + 3] > 0.5)
    if modes.size != 1 or int(modes[0]) != 0:
        raise ValueError("Verardo reference supports trump contracts only")
    suits = np.flatnonzero(
        vector[OBS_TRUMP_SUIT_OFFSET : OBS_TRUMP_SUIT_OFFSET + len(SUITS)] > 0.5
    )
    if suits.size != 1:
        raise ValueError("canonical trump observation must identify exactly one trump suit")
    return SUITS[int(suits[0])]


def _current_trick(vector: np.ndarray) -> tuple[tuple[int, Card], ...]:
    indices = np.flatnonzero(
        vector[OBS_TRICK_INDEX_OFFSET : OBS_TRICK_INDEX_OFFSET + 10] > 0.5
    )
    if indices.size != 1:
        raise ValueError("canonical v2 observation must identify exactly one trick index")
    trick_index = int(indices[0])
    history = decode_observation_history(vector)
    if trick_index >= 9 or trick_index >= len(history):
        return ()
    return tuple(history[trick_index])


def _highest_strength(cards: Sequence[Card], led_suit: str, trump_suit: str) -> Card:
    return max(
        cards,
        key=lambda card: (
            card_strength(card, led_suit, MODE_TRUMP, trump_suit),
            -_card_action(card),
        ),
    )


def _lowest_strength(cards: Sequence[Card], led_suit: str, trump_suit: str) -> Card:
    return min(
        cards,
        key=lambda card: (
            card_strength(card, led_suit, MODE_TRUMP, trump_suit),
            _card_action(card),
        ),
    )


def _highest_value(cards: Sequence[Card], trump_suit: str) -> Card:
    return max(
        cards,
        key=lambda card: (
            card_points(card, MODE_TRUMP, trump_suit),
            -_card_action(card),
        ),
    )


def _lowest_value(cards: Sequence[Card], trump_suit: str) -> Card:
    return min(
        cards,
        key=lambda card: (
            card_points(card, MODE_TRUMP, trump_suit),
            _card_action(card),
        ),
    )


def _partner_is_winning(plays: Sequence[tuple[int, Card]], trump_suit: str) -> bool:
    cards = [card for _, card in plays]
    winner = winning_card(cards, cards[0].suit, MODE_TRUMP, trump_suit)
    return next(relative_player for relative_player, card in plays if card == winner) == 2


def _lead_action(legal_cards: Sequence[Card], trump_suit: str) -> int:
    trumps = tuple(card for card in legal_cards if card.suit == trump_suit)
    for rank in ("J", "9"):
        for card in trumps:
            if card.rank == rank:
                return _card_action(card)

    non_trumps = tuple(card for card in legal_cards if card.suit != trump_suit)
    if non_trumps:
        return _card_action(_highest_value(non_trumps, trump_suit))
    return _card_action(_highest_strength(trumps, trump_suit, trump_suit))


def _follow_action(
    legal_cards: Sequence[Card],
    plays: Sequence[tuple[int, Card]],
    trump_suit: str,
) -> int:
    led_suit = plays[0][1].suit
    trumps = tuple(card for card in legal_cards if card.suit == trump_suit)

    if led_suit == trump_suit:
        if trumps:
            return _card_action(_highest_strength(trumps, trump_suit, trump_suit))
        return _card_action(_lowest_value(legal_cards, trump_suit))

    follows = tuple(card for card in legal_cards if card.suit == led_suit)
    partner_winning = _partner_is_winning(plays, trump_suit)
    valuable_trick = any(card.rank in _VALUABLE_RANKS for _, card in plays)

    if follows:
        if not partner_winning and valuable_trick and trumps:
            return _card_action(_lowest_strength(trumps, trump_suit, trump_suit))
        return _card_action(_highest_strength(follows, led_suit, trump_suit))

    if not partner_winning and trumps:
        return _card_action(_lowest_strength(trumps, trump_suit, trump_suit))

    discards = tuple(card for card in legal_cards if card.suit != trump_suit)
    return _card_action(_lowest_value(discards or legal_cards, trump_suit))


class VerardoReferencePolicy:
    """Deterministic public-information policy matching the published summary."""

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int:
        del agent
        vector = _observation_vector(observation)
        legal = _legal_actions(action_mask)

        if ANNOUNCE_ACTION in legal or PASS_ACTION in legal:
            return PASS_ACTION if PASS_ACTION in legal else ANNOUNCE_ACTION

        if any(action in _BIDDING_ACTIONS for action in legal):
            return _bid_action(vector, legal)

        card_actions = tuple(action for action in legal if action < len(ALL_CARDS))
        if not card_actions:
            raise ValueError(f"Verardo reference cannot handle legal actions {list(legal)}")
        trump_suit = _trump_suit(vector)
        legal_cards = tuple(ALL_CARDS[action] for action in card_actions)
        plays = _current_trick(vector)
        if not plays:
            return _lead_action(legal_cards, trump_suit)
        return _follow_action(legal_cards, plays, trump_suit)


class TrumpOnlyPolicy:
    """Restrict another public policy to the reference's four-suit bid space."""

    def __init__(self, policy: PublicPolicy) -> None:
        self.policy = policy

    def reset(self, seed: int) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(seed)

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray, agent: str) -> int:
        mask = np.asarray(action_mask).reshape(-1)
        restricted = mask
        if mask.shape == (ACTION_COUNT,) and any(mask[action] for action in _BIDDING_ACTIONS):
            restricted = trump_only_action_mask(mask)

        action = self.policy(observation, restricted, agent)
        if action in (BIDDING_OBEABE_ACTION, BIDDING_UNEUFE_ACTION):
            raise ValueError("wrapped policy selected a bid outside the trump-only benchmark")
        return action


def verardo_policy() -> VerardoReferencePolicy:
    """Return a fresh deterministic Verardo reference policy."""

    return VerardoReferencePolicy()


def verardo_tournament_config(
    *, episodes: int = VERARDO_QUALIFICATION_GAMES, seed: int = 0
) -> TournamentConfig:
    """Build the paired, seat-rotated benchmark configuration."""

    return TournamentConfig(
        episodes=episodes,
        seed=seed,
        modes=(None,),
        swap_teams=True,
        enable_bidding=True,
        trump_only_bidding=True,
        enable_weis=False,
        enable_stock=False,
        profile=VERARDO_V1_PROFILE,
    )


def run_verardo_benchmark(
    candidate: PublicPolicy,
    *,
    episodes: int = VERARDO_QUALIFICATION_GAMES,
    seed: int = 0,
) -> TournamentReport:
    """Run a same-deal, team-swapped tournament against the reference."""

    return run_tournament(
        TrumpOnlyPolicy(candidate),
        VerardoReferencePolicy(),
        verardo_tournament_config(episodes=episodes, seed=seed),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired tournament against the clean-room Verardo reference"
    )
    parser.add_argument("model", help="MaskablePPO model archive")
    parser.add_argument("--episodes", type=int, default=VERARDO_QUALIFICATION_GAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-results", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - optional CLI dependency
        raise SystemExit("sb3-contrib is required to load a benchmark model") from exc

    model = MaskablePPO.load(args.model)
    model_shape = tuple(getattr(model.observation_space, "shape", ()))
    if model_shape not in {(OBS_SIZE,), (CTDE_OBS_SIZE,)}:
        raise SystemExit(
            f"model observation shape {model_shape} is incompatible with canonical v2 "
            f"shape ({OBS_SIZE},) or CTDE shape ({CTDE_OBS_SIZE},)"
        )
    model_actions = getattr(model.action_space, "n", None)
    if model_actions != ACTION_COUNT:
        raise SystemExit(
            f"model action count {model_actions!r} is incompatible with canonical action "
            f"count {ACTION_COUNT}"
        )
    candidate = maskable_model_policy(model)
    report = run_verardo_benchmark(candidate, episodes=args.episodes, seed=args.seed)
    print(
        json.dumps(
            report.to_dict(include_results=args.include_results),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "VERARDO_BID_WEIGHTS",
    "VERARDO_PROFILE_VERSION",
    "VERARDO_QUALIFICATION_GAMES",
    "VERARDO_REFERENCE_COMMIT",
    "VERARDO_V1_PROFILE",
    "TrumpOnlyPolicy",
    "VerardoReferencePolicy",
    "main",
    "run_verardo_benchmark",
    "verardo_bid_scores",
    "verardo_policy",
    "verardo_tournament_config",
]


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI
    raise SystemExit(main())
