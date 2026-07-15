from types import SimpleNamespace

import numpy as np
import pytest

from env.jass_aec_env import ACTION_COUNT, OBS_SIZE
from rl.hybrid_policy import NeuralGuidanceConfig, NeuralGuidedPIMCPolicy
from rl.privileged_observation import CTDE_OBS_SIZE
from rl.search_policy import SearchStats


class _FakeSearch:
    def __init__(self, action: int, values: tuple[tuple[int, float, int], ...]) -> None:
        self.action = action
        self.last_stats = SearchStats(action_values=values)
        self.seeds: list[int | None] = []

    def __call__(self, observation, action_mask, agent):
        del observation, action_mask, agent
        return self.action

    def reset(self, seed=None) -> None:
        self.seeds.append(seed)


class _FakeModelPolicy:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities
        self.last_observation: np.ndarray | None = None

    def obs_to_tensor(self, observation):
        self.last_observation = np.asarray(observation)
        return self.last_observation, False

    def get_distribution(self, tensor, action_masks):
        del tensor
        probabilities = np.where(action_masks, self.probabilities, 0.0)
        probabilities = probabilities / probabilities.sum()
        return SimpleNamespace(distribution=SimpleNamespace(probs=probabilities.reshape(1, -1)))


class _FakeModel:
    def __init__(self, probabilities: np.ndarray, shape=(OBS_SIZE,)) -> None:
        self.observation_space = SimpleNamespace(shape=shape)
        self.policy = _FakeModelPolicy(probabilities)


def _inputs(*legal: int) -> tuple[np.ndarray, np.ndarray]:
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[list(legal)] = 1
    return observation, mask


def test_confident_neural_action_overrides_only_inside_search_gap() -> None:
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[3] = 0.2
    probabilities[5] = 0.8
    search = _FakeSearch(3, ((3, 10.0, 4), (5, 8.0, 4)))
    search.last_stats = SearchStats(
        attempted_determinizations=4,
        successful_determinizations=3,
        rollouts=6,
        simulated_plies=120,
        assignment_nodes=80,
        action_values=((3, 10.0, 4), (5, 8.0, 4)),
    )
    policy = NeuralGuidedPIMCPolicy(
        _FakeModel(probabilities),
        search,  # type: ignore[arg-type]
        config=NeuralGuidanceConfig(
            max_search_gap_raw_points=3.0,
            min_neural_probability=0.5,
        ),
    )
    observation, mask = _inputs(3, 5)

    assert policy(observation, mask, "p0") == 5
    assert policy.last_decision is not None
    assert policy.last_decision.overridden
    assert policy.last_decision.search_gap_raw_points == 2.0
    assert policy.stats.overrides == 1
    assert policy.stats.disagreements == 1
    assert policy.stats.attempted_determinizations == 4
    assert policy.stats.successful_determinizations == 3
    assert policy.stats.search_rollouts == 6
    assert policy.stats.simulated_plies == 120
    assert policy.stats.assignment_nodes == 80


def test_neural_action_cannot_override_a_material_search_advantage() -> None:
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[3] = 0.1
    probabilities[5] = 0.9
    search = _FakeSearch(3, ((3, 12.0, 4), (5, 8.0, 4)))
    policy = NeuralGuidedPIMCPolicy(
        _FakeModel(probabilities),
        search,  # type: ignore[arg-type]
        config=NeuralGuidanceConfig(max_search_gap_raw_points=3.0),
    )
    observation, mask = _inputs(3, 5)

    assert policy(observation, mask, "p0") == 3
    assert policy.last_decision is not None
    assert not policy.last_decision.overridden
    assert policy.stats.confidence_eligible == 1
    assert policy.stats.gap_eligible == 0


def test_ctde_actor_receives_public_prefix_and_zero_private_suffix() -> None:
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[3] = 0.6
    probabilities[5] = 0.4
    model = _FakeModel(probabilities, shape=(CTDE_OBS_SIZE,))
    search = _FakeSearch(3, ((3, 10.0, 2), (5, 9.0, 2)))
    policy = NeuralGuidedPIMCPolicy(model, search)  # type: ignore[arg-type]
    observation, mask = _inputs(3, 5)
    observation[17] = 1.0

    assert policy(observation, mask, "p0") == 3
    assert model.policy.last_observation is not None
    np.testing.assert_array_equal(model.policy.last_observation[:OBS_SIZE], observation)
    np.testing.assert_array_equal(
        model.policy.last_observation[OBS_SIZE:],
        np.zeros(CTDE_OBS_SIZE - OBS_SIZE),
    )


def test_non_card_and_forced_decisions_remain_search_only() -> None:
    probabilities = np.zeros(ACTION_COUNT)
    search = _FakeSearch(38, ())
    model = _FakeModel(probabilities)
    policy = NeuralGuidedPIMCPolicy(model, search)  # type: ignore[arg-type]
    observation, mask = _inputs(38, 39)

    assert policy(observation, mask, "p0") == 38
    assert model.policy.last_observation is None
    assert policy.stats.card_decisions == 0


def test_neural_bidding_is_explicit_and_uses_only_masked_public_policy() -> None:
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[38] = 0.1
    probabilities[39] = 0.9
    search = _FakeSearch(38, ())
    policy = NeuralGuidedPIMCPolicy(
        _FakeModel(probabilities),
        search,  # type: ignore[arg-type]
        config=NeuralGuidanceConfig(use_neural_bidding=True),
    )
    observation, mask = _inputs(38, 39)

    assert policy(observation, mask, "p0") == 39
    assert policy.stats.bidding_decisions == 1
    assert policy.stats.neural_bids == 1
    assert policy.last_decision is not None
    assert policy.last_decision.reason == "learned public actor selected the bid"


def test_reset_preserves_diagnostics_and_model_artifact_is_hashed(tmp_path) -> None:
    artifact = tmp_path / "model.zip"
    artifact.write_bytes(b"model")
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[3] = 1.0
    search = _FakeSearch(3, ((3, 1.0, 1),))
    policy = NeuralGuidedPIMCPolicy(
        _FakeModel(probabilities),
        search,
        model_path=artifact,  # type: ignore[arg-type]
    )
    observation, mask = _inputs(3)
    assert policy(observation, mask, "p0") == 3

    policy.reset(42)

    assert search.seeds == [42]
    assert policy.stats.decisions == 1
    assert policy.model_sha256 == (
        "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"
    )


@pytest.mark.parametrize(
    "config",
    [
        NeuralGuidanceConfig(max_search_gap_raw_points=-1.0),
        NeuralGuidanceConfig(min_neural_probability=-0.1),
        NeuralGuidanceConfig(min_neural_probability=1.1),
        NeuralGuidanceConfig(use_neural_bidding=1),
    ],
)
def test_invalid_guidance_config_is_rejected(config: NeuralGuidanceConfig) -> None:
    probabilities = np.zeros(ACTION_COUNT)
    probabilities[3] = 1.0
    with pytest.raises(ValueError):
        NeuralGuidedPIMCPolicy(
            _FakeModel(probabilities),
            _FakeSearch(3, ((3, 1.0, 1),)),  # type: ignore[arg-type]
            config=config,
        )
