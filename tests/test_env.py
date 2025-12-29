import pytest


def _skip_if_missing():
    try:
        import pettingzoo  # noqa: F401
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

from env.jass_aec_env import (
    ANNOUNCE_ACTION,
    PASS_ACTION,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    JassAECEnv,
)
from core.cards import MODE_TRUMP


def test_action_mask_matches_legal_cards() -> None:
    env = JassAECEnv(
        enable_bidding=False, enable_weis=False, mode=MODE_TRUMP, trump_suit="rosen", seed=1
    )
    env.reset()
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]

    player = int(agent[1:])
    legal = env.state.legal_cards_for(player)
    legal_indices = {env.card_to_index[(card.suit, card.rank)] for card in legal}

    for idx in range(36):
        assert mask[idx] == (1 if idx in legal_indices else 0)
    assert mask[36:].sum() == 0


def test_bidding_flow_push_then_partner() -> None:
    env = JassAECEnv(enable_bidding=True, seed=5, starter=0)
    env.reset()
    assert env.phase == "bidding"
    assert env.agent_selection == "p0"

    env.step(BIDDING_PUSH_ACTION)
    assert env.agent_selection == "p2"

    obs = env.observe(env.agent_selection)
    assert obs["action_mask"][BIDDING_PUSH_ACTION] == 0

    env.step(BIDDING_OBEABE_ACTION)
    assert env.phase == "announce"
    assert env.agent_selection == "p0"

    for _ in range(4):
        env.step(PASS_ACTION)
    assert env.phase == "play"
    assert env.agent_selection == "p0"


def test_play_through_game() -> None:
    import numpy as np

    env = JassAECEnv(
        enable_bidding=False, enable_weis=False, mode=MODE_TRUMP, trump_suit="schilten", seed=42
    )
    env.reset()

    for agent in env.agent_iter():
        obs = env.observe(agent)
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        mask = obs["action_mask"]
        legal_actions = np.flatnonzero(mask)
        assert len(legal_actions) > 0
        env.step(int(legal_actions[0]))

    assert all(env.terminations.values())
