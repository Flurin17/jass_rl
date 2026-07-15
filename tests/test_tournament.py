import pytest


def _skip_if_missing() -> None:
    try:
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
        import pettingzoo  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

from collections import defaultdict

import numpy as np

from core.cards import MODE_OBEABE, MODE_TRUMP
from env.jass_aec_env import ACTION_COUNT, OBS_SIZE
from rl.baselines import SeededRandomPolicy, StrategicHeuristicPolicy
from rl.tournament import (
    EpisodeResult,
    EpisodeSpec,
    TournamentConfig,
    build_schedule,
    maskable_model_policy,
    run_tournament,
    summarize_results,
    wilson_interval,
)


def _spec(pair_id: int, candidate_team: int, episode: int) -> EpisodeSpec:
    return EpisodeSpec(
        episode=episode,
        deal_index=pair_id,
        pair_id=pair_id,
        deal_seed=100 + pair_id,
        policy_seed=200 + pair_id,
        starter=pair_id % 4,
        candidate_team=candidate_team,
        mode=MODE_TRUMP if pair_id == 0 else MODE_OBEABE,
        trump_suit="schellen" if pair_id == 0 else None,
    )


def test_wilson_interval_known_value_and_validation() -> None:
    interval = wilson_interval(5, 10)
    assert interval.low == pytest.approx(0.236593, abs=1e-6)
    assert interval.high == pytest.approx(0.763407, abs=1e-6)
    assert wilson_interval(0, 0).low == 0.0
    assert wilson_interval(0, 0).high == 1.0

    with pytest.raises(ValueError, match="0 <= successes <= trials"):
        wilson_interval(2, 1)


@pytest.mark.parametrize("episodes", [0, -8, 7, 9])
def test_schedule_rejects_invalid_or_incomplete_episode_counts(episodes: int) -> None:
    with pytest.raises(ValueError, match="episodes"):
        build_schedule(TournamentConfig(episodes=episodes))


def test_schedule_pairs_same_deal_and_rotates_starters_and_teams() -> None:
    config = TournamentConfig(episodes=24, seed=17)
    schedule = build_schedule(config)
    pairs: dict[int, list[EpisodeSpec]] = defaultdict(list)
    for spec in schedule:
        pairs[spec.pair_id].append(spec)

    assert len(schedule) == 24
    assert {spec.starter for spec in schedule} == {0, 1, 2, 3}
    assert {spec.mode for spec in schedule} == {"trump", "obeabe", "uneufe"}
    assert len({spec.deal_seed for spec in schedule}) == 12
    for pair in pairs.values():
        assert len(pair) == 2
        assert {spec.candidate_team for spec in pair} == {0, 1}
        assert len({spec.deal_seed for spec in pair}) == 1
        assert len({spec.starter for spec in pair}) == 1
        assert len({spec.mode for spec in pair}) == 1
        assert len({spec.trump_suit for spec in pair}) == 1
        assert len({spec.policy_seed for spec in pair}) == 1


def test_summary_has_game_pair_mode_and_overall_metrics() -> None:
    results = (
        EpisodeResult(_spec(0, 0, 0), MODE_TRUMP, "schellen", (100, 57)),
        EpisodeResult(_spec(0, 1, 1), MODE_TRUMP, "schellen", (90, 67)),
        EpisodeResult(_spec(1, 0, 2), MODE_OBEABE, None, (50, 50)),
        EpisodeResult(_spec(1, 1, 3), MODE_OBEABE, None, (50, 50)),
    )

    report = summarize_results(results)

    assert report.overall.episodes == 4
    assert (report.overall.wins, report.overall.losses, report.overall.ties) == (1, 1, 2)
    assert report.overall.score_rate == 0.5
    assert report.overall.pairs == 2
    assert (
        report.overall.paired_wins,
        report.overall.paired_losses,
        report.overall.paired_ties,
    ) == (1, 0, 1)
    assert set(report.by_mode) == {MODE_TRUMP, MODE_OBEABE}
    assert report.by_mode[MODE_TRUMP].pairs == 1
    assert report.team_swap_supported is True


def test_maskable_model_adapter_preserves_canonical_observation_shape() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray, bool]] = []
            self.observation_space = type("Space", (), {"shape": (OBS_SIZE,)})()

        def predict(self, observation, *, action_masks, deterministic):
            self.calls.append((observation, action_masks, deterministic))
            return np.array([np.flatnonzero(action_masks)[0]]), None

    model = FakeModel()
    policy = maskable_model_policy(model)
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    mask = np.zeros(ACTION_COUNT, dtype=np.int8)
    mask[7] = 1

    assert policy(observation, mask, "p3") == 7
    encoded, passed_mask, deterministic = model.calls[0]
    assert encoded is observation
    assert encoded.shape == (OBS_SIZE,)
    assert passed_mask is mask
    assert deterministic is True


def test_small_tournament_is_reproducible_and_truly_swaps_teams() -> None:
    config = TournamentConfig(
        episodes=8,
        seed=31,
        modes=(MODE_OBEABE,),
        enable_weis=False,
        enable_stock=False,
    )

    first = run_tournament(StrategicHeuristicPolicy(), SeededRandomPolicy(9), config)
    second = run_tournament(StrategicHeuristicPolicy(), SeededRandomPolicy(9), config)

    assert [result.team_points for result in first.results] == [
        result.team_points for result in second.results
    ]
    assert first.overall.candidate_decisions > 0
    assert first.overall.mean_candidate_inference_ms > 0.0
    assert set(first.by_contract) == {MODE_OBEABE}
    assert len(first.results) == 8
    assert {result.spec.candidate_team for result in first.results} == {0, 1}
    assert first.overall.pairs == 4
    assert set(first.by_mode) == {MODE_OBEABE}
    totals = {sum(result.team_points) for result in first.results}
    assert len(totals) == 1
    assert next(iter(totals)) > 0
