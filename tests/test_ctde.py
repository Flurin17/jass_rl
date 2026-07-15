from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from core.cards import ALL_CARDS
from env.jass_aec_env import OBS_SIZE
from rl.ctde_policy import (
    CTDEMaskableActorCriticPolicy,
    PublicActorPrivilegedCriticExtractor,
)
from rl.privileged_observation import (
    CTDE_OBS_SIZE,
    PRIVILEGED_HANDS_SIZE,
    augment_with_privileged_hands,
    encode_privileged_hands,
    pad_public_observation,
)


def test_privileged_hands_are_relative_and_public_prefix_is_unchanged() -> None:
    hands = [list(ALL_CARDS[index : index + 1]) for index in range(4)]
    public = np.linspace(0.0, 1.0, OBS_SIZE, dtype=np.float32)

    encoded = encode_privileged_hands(hands, observer=2)
    augmented = augment_with_privileged_hands(public, hands, observer=2)

    assert encoded.shape == (PRIVILEGED_HANDS_SIZE,)
    assert encoded[ALL_CARDS.index(hands[2][0])] == 1.0
    assert encoded[36 + ALL_CARDS.index(hands[3][0])] == 1.0
    assert encoded[72 + ALL_CARDS.index(hands[0][0])] == 1.0
    assert encoded[108 + ALL_CARDS.index(hands[1][0])] == 1.0
    assert augmented.shape == (CTDE_OBS_SIZE,)
    np.testing.assert_array_equal(augmented[:OBS_SIZE], public)


def test_public_padding_never_invents_private_cards() -> None:
    public = np.ones(OBS_SIZE, dtype=np.float32)
    padded = pad_public_observation(public)
    np.testing.assert_array_equal(padded[:OBS_SIZE], public)
    assert not np.any(padded[OBS_SIZE:])


def test_actor_is_invariant_to_privileged_suffix_but_critic_is_not() -> None:
    torch.manual_seed(7)
    extractor = PublicActorPrivilegedCriticExtractor(
        CTDE_OBS_SIZE,
        {"pi": [32, 16], "vf": [32, 16]},
        torch.nn.Tanh,
    )
    first = torch.zeros((1, CTDE_OBS_SIZE))
    second = first.clone()
    second[:, OBS_SIZE:] = 1.0

    torch.testing.assert_close(extractor.forward_actor(first), extractor.forward_actor(second))
    assert not torch.allclose(extractor.forward_critic(first), extractor.forward_critic(second))


def test_ctde_policy_refuses_to_expand_the_actor_private_boundary() -> None:
    with pytest.raises(ValueError, match="canonical public prefix"):
        CTDEMaskableActorCriticPolicy(
            spaces.Box(0.0, 1.0, (CTDE_OBS_SIZE,), dtype=np.float32),
            spaces.Discrete(45),
            lambda _: 3e-4,
            net_arch=[16],
            public_observation_size=CTDE_OBS_SIZE,
        )
