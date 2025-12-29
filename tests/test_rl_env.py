import pytest


def _skip_if_missing():
    try:
        import pettingzoo  # noqa: F401
        import gymnasium  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        pytest.skip("pettingzoo/gymnasium/numpy required", allow_module_level=True)


_skip_if_missing()

from rl.single_agent_env import JassSingleAgentEnv
from core.cards import MODE_TRUMP


def test_single_agent_env_step() -> None:
    env = JassSingleAgentEnv(
        enable_bidding=False, enable_weis=False, mode=MODE_TRUMP, trump_suit="schilten", seed=3
    )
    obs, _ = env.reset()
    assert obs.shape[0] == 118

    for _ in range(5):
        mask = env.get_action_mask()
        action = int(mask.nonzero()[0][0])
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
