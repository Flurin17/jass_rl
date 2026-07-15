from dataclasses import replace

import pytest


def _skip_if_missing() -> None:
    try:
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
        import pettingzoo  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

import numpy as np

from core.cards import MODE_TRUMP, Card
from core.legal_moves import RuleSet
from env.jass_aec_env import ACTION_COUNT, OBS_SIZE, JassAECEnv
from rl.search_policy import (
    PIMCConfig,
    PIMCSearchPolicy,
    _inferred_constraints,
    _play,
    _Position,
    _Simulation,
    _utility,
    _verardo_bid_constraints_satisfied,
    _verardo_opponent_card,
)


def _play_observation(seed: int = 7) -> tuple[JassAECEnv, str, dict[str, np.ndarray]]:
    env = JassAECEnv(
        seed=seed,
        enable_bidding=False,
        enable_weis=False,
        enable_stock=False,
        mode=MODE_TRUMP,
        trump_suit="rosen",
        starter=0,
    )
    env.reset(seed=seed)
    agent = env.agent_selection
    return env, agent, env.observe(agent)


def _small_config(**overrides) -> PIMCConfig:
    values = {
        "determinizations": 3,
        "max_rollouts": 27,
        "max_rollout_plies": 36,
        "assignment_node_limit": 2_000,
        "rollout_randomness": 0.05,
    }
    values.update(overrides)
    return PIMCConfig(**values)


def test_search_is_seeded_reproducible_and_strictly_legal() -> None:
    env, agent, visible = _play_observation()
    try:
        left = PIMCSearchPolicy(seed=123, config=_small_config())
        right = PIMCSearchPolicy(seed=123, config=_small_config())

        left_action = left(visible["observation"], visible["action_mask"], agent)
        right_action = right(visible["observation"], visible["action_mask"], agent)

        assert left_action == right_action
        assert visible["action_mask"][left_action] == 1
        assert left.last_stats.action_values == right.last_stats.action_values
        assert left.last_stats.successful_determinizations == 3
    finally:
        env.close()


def test_search_uses_no_hidden_environment_state() -> None:
    left_env, left_agent, left_visible = _play_observation(seed=19)
    right_env, right_agent, _ = _play_observation(seed=19)
    try:
        assert right_env.state is not None
        right_env.state.hands[1], right_env.state.hands[3] = (
            right_env.state.hands[3],
            right_env.state.hands[1],
        )
        right_visible = right_env.observe(right_agent)

        np.testing.assert_array_equal(left_visible["observation"], right_visible["observation"])
        np.testing.assert_array_equal(left_visible["action_mask"], right_visible["action_mask"])
        left = PIMCSearchPolicy(seed=88, config=_small_config())
        right = PIMCSearchPolicy(seed=88, config=_small_config())
        assert left(left_visible["observation"], left_visible["action_mask"], left_agent) == right(
            right_visible["observation"], right_visible["action_mask"], right_agent
        )
    finally:
        left_env.close()
        right_env.close()


def test_compute_controls_bound_rollouts_plies_and_assignment_work() -> None:
    env, agent, visible = _play_observation(seed=3)
    try:
        config = _small_config(
            determinizations=20,
            max_rollouts=9,
            max_rollout_plies=12,
            assignment_node_limit=100,
            rollout_randomness=0.0,
        )
        policy = PIMCSearchPolicy(seed=4, config=config)
        action = policy(visible["observation"], visible["action_mask"], agent)

        assert visible["action_mask"][action] == 1
        assert policy.last_stats.rollouts <= config.max_rollouts
        assert policy.last_stats.simulated_plies <= (
            policy.last_stats.rollouts * config.max_rollout_plies
        )
        assert policy.last_stats.assignment_nodes <= (
            policy.last_stats.attempted_determinizations * config.assignment_node_limit
        )
    finally:
        env.close()


def test_bidding_and_announcement_actions_use_public_fallback() -> None:
    env = JassAECEnv(seed=5, enable_bidding=True, enable_weis=True)
    env.reset(seed=5)
    try:
        agent = env.agent_selection
        visible = env.observe(agent)
        policy = PIMCSearchPolicy(seed=9, config=_small_config())
        action = policy(visible["observation"], visible["action_mask"], agent)
        assert action >= 36
        assert visible["action_mask"][action] == 1
        assert policy.last_stats.used_fallback
    finally:
        env.close()


def test_voluntary_trump_does_not_falsely_prove_a_plain_suit_void() -> None:
    position = _Position(
        hand=(),
        history=(
            (
                (0, Card("rosen", "A")),
                (1, Card("eicheln", "6")),
                (2, Card("schilten", "6")),
                (3, Card("rosen", "6")),
            ),
        ),
        current_trick=(),
        hand_counts=(0, 0, 0, 0),
        mode=MODE_TRUMP,
        trump_suit="eicheln",
        leader=0,
        trick_index=1,
        tricks_won=(1, 0),
        team_points=(0, 0),
        bid_starter=None,
        bid_chooser=None,
        bid_pushed=False,
        contract_factor=1,
        match_bonus=100,
    )

    forbidden_suits, _ = _inferred_constraints(position, RuleSet())

    assert "rosen" not in forbidden_suits[1]
    assert "rosen" in forbidden_suits[2]
    assert "rosen" not in forbidden_suits[3]

    strict_follow, _ = _inferred_constraints(
        position,
        RuleSet(allow_trump_on_non_trump_lead=False),
    )
    assert "rosen" in strict_follow[1]


def test_terminal_utility_includes_current_score_and_win_bonus() -> None:
    simulation = _Simulation(
        hands=[[], [], [], []],
        current_trick=[],
        leader=0,
        trick_index=9,
        tricks_won=[5, 4],
        initial_points=[40, 60],
        future_points=[80, 50],
        mode=MODE_TRUMP,
        trump_suit="rosen",
        factor=1,
        match_bonus=100,
    )

    assert _utility(simulation) == 10.0
    assert _utility(simulation, terminal_win_bonus=30.0) == 40.0

    simulation.future_points = [50, 80]
    assert _utility(simulation, terminal_win_bonus=30.0) == -80.0


def test_simulated_future_stock_is_awarded_once_before_contract_factor() -> None:
    simulation = _Simulation(
        hands=[
            [Card("eicheln", "K"), Card("eicheln", "Q")],
            [],
            [],
            [],
        ],
        current_trick=[
            (1, Card("rosen", "6")),
            (2, Card("rosen", "7")),
            (3, Card("rosen", "8")),
        ],
        leader=1,
        trick_index=0,
        tricks_won=[0, 0],
        initial_points=[0, 0],
        future_points=[0, 0],
        mode=MODE_TRUMP,
        trump_suit="eicheln",
        factor=2,
        match_bonus=100,
        allow_stock=True,
    )

    _play(simulation, 0, Card("eicheln", "K"))
    points_after_trick = simulation.future_points[0]
    _play(simulation, 0, Card("eicheln", "Q"))

    assert simulation.future_points[0] == points_after_trick + 20
    assert simulation.stock_announced_by == {0}
    assert _utility(simulation) == (points_after_trick + 20) * 2


def test_verardo_opponent_rollout_uses_documented_follow_policy() -> None:
    simulation = _Simulation(
        hands=[[], [], [], []],
        current_trick=[(0, Card("rosen", "6"))],
        leader=0,
        trick_index=2,
        tricks_won=[1, 1],
        initial_points=[20, 20],
        future_points=[0, 0],
        mode=MODE_TRUMP,
        trump_suit="eicheln",
        factor=1,
        match_bonus=100,
    )
    legal = [Card("rosen", "7"), Card("rosen", "A")]

    assert _verardo_opponent_card(simulation, 1, legal) == Card("rosen", "A")


def test_verardo_hand_samples_respect_public_push_and_trump_choice() -> None:
    position = _Position(
        hand=(),
        history=(),
        current_trick=(),
        hand_counts=(0, 9, 0, 9),
        mode=MODE_TRUMP,
        trump_suit="eicheln",
        leader=0,
        trick_index=0,
        tricks_won=(0, 0),
        team_points=(0, 0),
        bid_starter=1,
        bid_chooser=3,
        bid_pushed=True,
        contract_factor=1,
        match_bonus=100,
    )
    starter_without_stopper = [
        Card("schellen", "6"),
        Card("schellen", "7"),
        Card("rosen", "6"),
        Card("rosen", "7"),
        Card("schilten", "6"),
        Card("schilten", "7"),
        Card("eicheln", "6"),
        Card("eicheln", "7"),
        Card("eicheln", "8"),
    ]
    eicheln_chooser = [
        Card("eicheln", "J"),
        Card("eicheln", "9"),
        Card("eicheln", "A"),
        Card("eicheln", "K"),
        Card("eicheln", "Q"),
        Card("schellen", "8"),
        Card("rosen", "8"),
        Card("schilten", "8"),
        Card("schilten", "10"),
    ]

    assert _verardo_bid_constraints_satisfied(
        position,
        [[], starter_without_stopper, [], eicheln_chooser],
    )

    rosen_chooser = [
        Card("rosen", card.rank) if card.suit == "eicheln" else card for card in eicheln_chooser
    ]
    assert not _verardo_bid_constraints_satisfied(
        position,
        [[], starter_without_stopper, [], rosen_chooser],
    )

    not_pushed = replace(position, bid_pushed=False)
    assert not _verardo_bid_constraints_satisfied(
        not_pushed,
        [[], starter_without_stopper, [], eicheln_chooser],
    )


def test_same_canonical_position_is_seat_label_invariant() -> None:
    env, _, visible = _play_observation(seed=41)
    try:
        policy = PIMCSearchPolicy(seed=17, config=_small_config())
        first = policy(visible["observation"], visible["action_mask"], "p0")
        second = policy(visible["observation"], visible["action_mask"], "p2")
        assert first == second
    finally:
        env.close()


def test_policy_rejects_malformed_inputs_and_empty_mask() -> None:
    policy = PIMCSearchPolicy(config=_small_config())
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    with pytest.raises(ValueError, match="canonical observation"):
        policy(np.zeros(10), mask, "p0")
    with pytest.raises(ValueError, match="without a legal action"):
        policy(np.zeros(OBS_SIZE), mask, "p0")
    with pytest.raises(ValueError, match="canonical action mask"):
        policy(np.zeros(OBS_SIZE), np.zeros(10), "p0")


@pytest.mark.parametrize(
    "config",
    [
        PIMCConfig(determinizations=0),
        PIMCConfig(max_rollouts=0),
        PIMCConfig(max_rollout_plies=37),
        PIMCConfig(assignment_node_limit=0),
        PIMCConfig(rollout_randomness=-0.1),
        PIMCConfig(rollout_randomness=1.1),
        PIMCConfig(terminal_win_bonus=-0.1),
        PIMCConfig(common_random_numbers=1),
        PIMCConfig(opponent_rollout_policy="unknown"),
        PIMCConfig(infer_opponent_bids=1),
        PIMCConfig(infer_opponent_bids=True, opponent_rollout_policy="generic"),
        PIMCConfig(prune_bid_constraints=1),
        PIMCConfig(prune_bid_constraints=True),
        PIMCConfig(time_budget_ms=0),
    ],
)
def test_invalid_compute_configuration_is_rejected(config: PIMCConfig) -> None:
    with pytest.raises(ValueError):
        PIMCSearchPolicy(config=config)
