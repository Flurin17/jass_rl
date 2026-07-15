import pytest


def _skip_if_missing():
    try:
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
        import pettingzoo  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

import numpy as np
from pettingzoo.test import api_test

from core.cards import ALL_CARDS, MODE_OBEABE, MODE_TRUMP, SUITS
from core.ruleset import RulesetConfig
from env.jass_aec_env import (
    ANNOUNCE_ACTION,
    BIDDING_OBEABE_ACTION,
    BIDDING_PUSH_ACTION,
    OBS_ANNOUNCEMENT_STATUS_OFFSET,
    OBS_BID_CHOOSER_OFFSET,
    OBS_BID_CURRENT_OFFSET,
    OBS_BID_PUSHED_OFFSET,
    OBS_BID_STARTER_OFFSET,
    OBS_HISTORY_CARDS_OFFSET,
    OBS_HISTORY_PLAYERS_OFFSET,
    OBS_PHASE_OFFSET,
    OBS_PLAYER_COUNT,
    OBS_TRICK_COMPLETE_OFFSET,
    OBS_TRICK_WINNER_OFFSET,
    OBSERVATION_SCHEMA_VERSION,
    PASS_ACTION,
    JassAECEnv,
    decode_current_trick,
    decode_observation_history,
    decode_played_cards,
)


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
    # Cards are dealt before bidding (Schieber).
    obs = env.observe("p0")["observation"]
    assert int(obs[:36].sum()) == 9

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


def test_pettingzoo_api_contract() -> None:
    env = JassAECEnv(seed=11, enable_bidding=True, enable_weis=True, enable_stock=True)
    api_test(env, num_cycles=100)


def test_observation_schema_tracks_public_bidding_and_announcements() -> None:
    env = JassAECEnv(seed=5, starter=3, enable_bidding=True, enable_weis=True)
    env.reset()
    assert env.metadata["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION

    # From p1, absolute starter/current player p3 is relative seat 2.
    obs = env.observe("p1")["observation"]
    assert obs[OBS_BID_STARTER_OFFSET + 2] == 1.0
    assert obs[OBS_BID_CURRENT_OFFSET + 2] == 1.0

    env.step(BIDDING_PUSH_ACTION)
    obs = env.observe("p0")["observation"]
    assert obs[OBS_BID_PUSHED_OFFSET] == 1.0
    assert obs[OBS_BID_CURRENT_OFFSET + 1] == 1.0
    assert obs[OBS_BID_CHOOSER_OFFSET : OBS_BID_CHOOSER_OFFSET + 4].sum() == 0

    env.step(BIDDING_OBEABE_ACTION)
    obs = env.observe("p2")["observation"]
    # The chooser was p1, which is relative seat 3 from p2.
    assert obs[OBS_BID_CHOOSER_OFFSET + 3] == 1.0
    assert obs[OBS_PHASE_OFFSET + 1] == 1.0

    # Announcement order begins at leader p3. Pass/announce decisions are public
    # and rotate into each observer's relative-seat slots.
    env.step(PASS_ACTION)
    env.step(ANNOUNCE_ACTION)
    obs = env.observe("p2")["observation"]
    p3_relative = (3 - 2) % 4
    p0_relative = (0 - 2) % 4
    assert obs[OBS_ANNOUNCEMENT_STATUS_OFFSET + p3_relative * 3 + 1] == 1.0
    assert obs[OBS_ANNOUNCEMENT_STATUS_OFFSET + p0_relative * 3 + 2] == 1.0


def test_observation_history_is_ordered_attributed_and_private() -> None:
    env = JassAECEnv(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        starter=1,
        seed=9,
    )
    env.reset()

    for _ in range(4):
        agent = env.agent_selection
        action = int(np.flatnonzero(env.observe(agent)["action_mask"])[0])
        env.step(action)

    result = env.state.completed_tricks[0]
    for observer in range(4):
        vector = env.observe(f"p{observer}")["observation"]
        decoded_history = decode_observation_history(vector)
        assert decoded_history[0] == [
            ((player - observer) % 4, card) for player, card in result.plays
        ]
        assert decode_current_trick(vector) == []
        assert decode_played_cards(vector) == set(result.cards)
        assert vector[OBS_TRICK_COMPLETE_OFFSET] == 1.0
        for play_slot, (player, card) in enumerate(result.plays):
            card_index = env.card_to_index[(card.suit, card.rank)]
            card_start = OBS_HISTORY_CARDS_OFFSET + play_slot * 36
            player_start = OBS_HISTORY_PLAYERS_OFFSET + play_slot * OBS_PLAYER_COUNT
            assert np.flatnonzero(vector[card_start : card_start + 36]).tolist() == [
                card_index
            ]
            assert np.flatnonzero(
                vector[player_start : player_start + OBS_PLAYER_COUNT]
            ).tolist() == [(player - observer) % 4]
        winner_start = OBS_TRICK_WINNER_OFFSET
        assert np.flatnonzero(
            vector[winner_start : winner_start + OBS_PLAYER_COUNT]
        ).tolist() == [(result.winner - observer) % 4]

    current_agent = env.agent_selection
    current_action = int(
        np.flatnonzero(env.observe(current_agent)["action_mask"])[0]
    )
    current_card = env.index_to_card[current_action]
    env.step(current_action)
    p0_vector = env.observe("p0")["observation"]
    assert decode_current_trick(p0_vector) == [current_card]
    assert decode_played_cards(p0_vector) == set(result.cards) | {current_card}

    # Changing unseen, unplayed cards between two other hands cannot change p0's
    # observation. Only p0's own hand and public play/actions are represented.
    before = env.observe("p0")["observation"]
    env.state.hands[1][0], env.state.hands[2][0] = (
        env.state.hands[2][0],
        env.state.hands[1][0],
    )
    after = env.observe("p0")["observation"]
    np.testing.assert_array_equal(after, before)


def test_high_weis_and_terminal_observations_fit_declared_space() -> None:
    env = JassAECEnv(
        enable_bidding=False,
        enable_weis=True,
        enable_stock=False,
        mode=MODE_OBEABE,
        seed=2,
    )
    env.reset()
    # Four nine-card suit runs yield a 600-point winning-team Weis score, well
    # above the old observation's 200-point normalization ceiling.
    env.state.hands = [
        [card for card in ALL_CARDS if card.suit == suit] for suit in SUITS
    ]
    for _ in range(4):
        env.step(ANNOUNCE_ACTION)
    # Obeabe has factor 3, so the 600 raw Weis points score 1,800.
    assert env.state.team_points == [1800, 0]
    for agent in env.possible_agents:
        assert env.observation_space(agent).contains(env.observe(agent))

    while not all(env.terminations.values()):
        agent = env.agent_selection
        observation = env.observe(agent)
        assert env.observation_space(agent).contains(observation)
        action = int(np.flatnonzero(observation["action_mask"])[0])
        env.step(action)

    assert env.phase == "terminal"
    # Team A also won all tricks: (600 Weis + 157 cards + 100 Match) * 3.
    assert env.state.team_points == [2571, 0]
    assert env.match_team == 0
    for agent in env.possible_agents:
        terminal_observation = env.observe(agent)
        assert env.observation_space(agent).contains(terminal_observation)
        assert terminal_observation["action_mask"].sum() == 0
        assert terminal_observation["observation"][OBS_PHASE_OFFSET + 3] == 1.0

    while env.agents:
        env.step(None)
    assert env.agents == []


def test_stock_uses_the_same_contract_factor_as_tricks() -> None:
    env = JassAECEnv(
        enable_bidding=False,
        enable_weis=False,
        enable_stock=True,
        mode=MODE_TRUMP,
        trump_suit="schilten",
        starter=0,
        seed=1,
    )
    env.reset()
    # p0 holds every trump, so it can lead K, win, then play Q. The second
    # Stoeck card scores 20 raw points with the Schilten factor 2.
    suit_order = ("schilten", "schellen", "rosen", "eicheln")
    env.state.hands = [
        [card for card in ALL_CARDS if card.suit == suit] for suit in suit_order
    ]
    king = env.card_to_index[("schilten", "K")]
    queen = env.card_to_index[("schilten", "Q")]
    env.step(king)
    for _ in range(3):
        agent = env.agent_selection
        env.step(int(np.flatnonzero(env.observe(agent)["action_mask"])[0]))

    before = env.state.team_points[0]
    env.step(queen)
    assert env.state.team_points[0] - before == 40
    assert env.rewards["p0"] == 40
    assert env.rewards["p2"] == 40


def test_custom_rules_profile_controls_legality_features_and_scoring() -> None:
    profile = RulesetConfig(
        allow_stock=False,
        allow_weis=False,
        version="env-test-v1",
        contract_factors={"rosen": 5},
        match_bonus=0,
    )
    env = JassAECEnv(
        profile=profile,
        enable_bidding=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        seed=6,
    )
    env.reset()
    assert env.phase == "play"
    assert env.enable_weis is False
    assert env.enable_stock is False
    assert env.contract_factor == 5
    assert env.ruleset is profile.legal_moves
    assert env.metadata["rules_profile_version"] == "env-test-v1"

    for _ in range(4):
        agent = env.agent_selection
        env.step(int(np.flatnonzero(env.observe(agent)["action_mask"])[0]))
    raw_points = env.state.completed_tricks[0].points
    assert sum(env.state.team_points) == raw_points * 5
