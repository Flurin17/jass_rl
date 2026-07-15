import json

import pytest

from rl.benchmark_training import (
    BenchmarkConfig,
    _worker_command,
    peak_rss_bytes,
    run_worker,
)


def test_benchmark_config_and_rss_units() -> None:
    config = BenchmarkConfig(n_steps=16, n_envs=2, batch_size=16, rollouts=3)
    config.validate()
    assert config.total_timesteps == 96
    assert peak_rss_bytes(123, "darwin") == 123
    assert peak_rss_bytes(123, "linux") == 123 * 1024

    with pytest.raises(ValueError, match="evenly divide"):
        BenchmarkConfig(n_steps=10, batch_size=6).validate()


def test_worker_command_is_an_isolated_module_invocation() -> None:
    config = BenchmarkConfig(n_steps=8, batch_size=8, n_epochs=1, rollouts=1)
    command = _worker_command(config, "cpu")
    assert command[1:4] == ["-m", "rl.benchmark_training", "--worker"]
    assert command[command.index("--devices") + 1] == "cpu"


def test_tiny_cpu_worker_reports_exact_steps_and_memory() -> None:
    result = run_worker(
        BenchmarkConfig(
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            rollouts=1,
            net_arch=(16,),
            cpu_threads=1,
        ),
        "cpu",
    )

    assert result["status"] == "ok"
    assert result["algorithm"] == "MaskablePPO-CTDE"
    assert result["cpu_threads"] == 1
    assert result["requested_timesteps"] == result["actual_timesteps"] == 8
    assert result["steps_per_second"] > 0
    assert result["peak_rss_bytes"] > 0
    json.dumps(result)
