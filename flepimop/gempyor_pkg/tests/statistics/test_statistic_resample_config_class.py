"""Unit tests for the `gempyor.statistics.StatisticResampleConfig` class."""

import pytest
from xarray.core.resample import DataArrayResample

from gempyor.statistics import StatisticResampleConfig


@pytest.mark.parametrize("freq", ("ABC", "DEF", "GHI"))
def test_validate_freq(freq: str) -> None:
    """Test the `validate_freq` class method."""
    with pytest.raises(
        ValueError, match=f"Invalid frequency: {freq}, failed to parse with error message"
    ):
        StatisticResampleConfig.validate_freq(freq)


@pytest.mark.parametrize("aggregator", ("exp", "log", "sqrt"))
def test_validate_aggregator(aggregator: str) -> None:
    """Test the `validate_aggregator` class method."""
    assert aggregator not in dir(DataArrayResample)
    with pytest.raises(
        ValueError,
        match=(
            f"Given an unsupported aggregator name, '{aggregator}', "
            "must be a valid xarray DataArrayResample method."
        ),
    ):
        StatisticResampleConfig.validate_aggregator(aggregator)
