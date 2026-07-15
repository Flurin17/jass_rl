"""Training-only observations for centralized-critic/decentralized execution.

The actor must never receive another player's cards.  During training, the
critic may additionally see the four remaining hands in acting-seat-relative
order.  Keeping the public prefix byte-for-byte identical to the canonical
schema makes that separation auditable and lets inference pad the private
suffix with zeros.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.cards import ALL_CARDS, Card
from env.jass_aec_env import OBS_SIZE

PRIVILEGED_HAND_COUNT = 4
PRIVILEGED_CARDS_PER_HAND = len(ALL_CARDS)
PRIVILEGED_HANDS_SIZE = PRIVILEGED_HAND_COUNT * PRIVILEGED_CARDS_PER_HAND
CTDE_OBS_SIZE = OBS_SIZE + PRIVILEGED_HANDS_SIZE
PRIVILEGED_OBSERVATION_SCHEMA_NAME = "relative_remaining_hands_v1"
PRIVILEGED_OBSERVATION_SCHEMA_VERSION = 1

_CARD_INDEX = {card: index for index, card in enumerate(ALL_CARDS)}


def encode_privileged_hands(
    hands: Sequence[Sequence[Card]], observer: int
) -> np.ndarray:
    """Encode remaining hands as self/left/partner/right one-hot blocks."""

    if len(hands) != PRIVILEGED_HAND_COUNT:
        raise ValueError("privileged encoding requires exactly four hands")
    if isinstance(observer, bool) or not isinstance(observer, int) or observer not in range(4):
        raise ValueError("observer must be an integer from 0 to 3")

    encoded = np.zeros(PRIVILEGED_HANDS_SIZE, dtype=np.float32)
    seen: set[Card] = set()
    for relative_seat in range(PRIVILEGED_HAND_COUNT):
        absolute_seat = (observer + relative_seat) % PRIVILEGED_HAND_COUNT
        for card in hands[absolute_seat]:
            if card in seen:
                raise ValueError("a card cannot appear in more than one remaining hand")
            seen.add(card)
            encoded[
                relative_seat * PRIVILEGED_CARDS_PER_HAND + _CARD_INDEX[card]
            ] = 1.0
    return encoded


def augment_with_privileged_hands(
    public_observation: np.ndarray,
    hands: Sequence[Sequence[Card]],
    observer: int,
) -> np.ndarray:
    """Append the training-only hand encoding to a canonical public vector."""

    public = np.asarray(public_observation, dtype=np.float32).reshape(-1)
    if public.shape != (OBS_SIZE,):
        raise ValueError(f"public observation must have shape ({OBS_SIZE},), got {public.shape}")
    return np.concatenate((public, encode_privileged_hands(hands, observer)))


def pad_public_observation(public_observation: np.ndarray) -> np.ndarray:
    """Create a CTDE model input without inventing private inference data."""

    public = np.asarray(public_observation, dtype=np.float32).reshape(-1)
    if public.shape != (OBS_SIZE,):
        raise ValueError(f"public observation must have shape ({OBS_SIZE},), got {public.shape}")
    return np.pad(public, (0, PRIVILEGED_HANDS_SIZE))


__all__ = [
    "CTDE_OBS_SIZE",
    "PRIVILEGED_HANDS_SIZE",
    "PRIVILEGED_OBSERVATION_SCHEMA_NAME",
    "PRIVILEGED_OBSERVATION_SCHEMA_VERSION",
    "augment_with_privileged_hands",
    "encode_privileged_hands",
    "pad_public_observation",
]
