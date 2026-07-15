import pytest

from rl.qualify_hybrid import (
    EXTERNAL_OPPONENTS,
    FULL_OPPONENTS,
    _resolved_opponents,
    main,
)


def test_benchmark_opponent_scopes_are_explicit() -> None:
    assert _resolved_opponents("full", None) == FULL_OPPONENTS
    assert _resolved_opponents("external", None) == EXTERNAL_OPPONENTS
    assert _resolved_opponents("external", ["verardo"]) == ("verardo",)

    with pytest.raises(ValueError, match="does not support"):
        _resolved_opponents("full", ["verardo"])
    with pytest.raises(ValueError, match="duplicates"):
        _resolved_opponents("external", ["verardo", "verardo"])


def test_bid_inference_cannot_be_mislabeled_as_random_opponent_model() -> None:
    with pytest.raises(SystemExit, match="separate reports"):
        main(
            [
                "missing.zip",
                "--benchmark",
                "external",
                "--infer-opponent-bids",
            ]
        )
