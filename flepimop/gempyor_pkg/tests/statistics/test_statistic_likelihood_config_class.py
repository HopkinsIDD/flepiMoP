"""Unit tests for the `gempyor.statistics.StatisticLikelihoodConfig` class."""

import pytest

from gempyor.statistics import _DIST_MAP, StatisticLikelihoodConfig


@pytest.mark.parametrize("dist", ("invalid", "false", "nonexistent"))
def test_validate_dist(dist: str) -> None:
    """Test the `validate_dist` class method."""
    assert dist not in _DIST_MAP
    with pytest.raises(
        ValueError,
        match=(
            f"Given an unsupported distribution name, '{dist}', " f"must be one of: .*."
        ),
    ):
        StatisticLikelihoodConfig.validate_dist(dist)
