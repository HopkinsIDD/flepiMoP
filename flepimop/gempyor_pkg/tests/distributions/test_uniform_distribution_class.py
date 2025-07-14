import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import UniformDistribution


@pytest.mark.parametrize(
    "low, high",
    [
        (0.0, 10.0),
        (-10.0, -5.0),
        (-5.0, 5.0),
    ],
    ids=["positive_range", "negative_range", "spanning_range"],
)
def test_uniform_distribution_init_valid(low: float, high: float) -> None:
    dist = UniformDistribution(low=low, high=high)
    assert dist.low == low
    assert dist.high == high
    assert dist.distribution == "uniform"


@pytest.mark.parametrize(
    "low, high",
    [
        (10.0, 0.0),
        (5.0, 5.0),
    ],
    ids=["high_less_than_low", "high_equals_low"],
)
def test_uniform_distribution_init_invalid_bounds(low: float, high: float) -> None:
    with pytest.raises(ValidationError, match="must be greater than the 'low' value"):
        UniformDistribution(low=low, high=high)


@pytest.mark.parametrize(
    "low, high",
    [
        (10.0, 20.0),
        (-5.0, 5.0),
        (0.0, 1.0),
    ],
    ids=["positive_range", "centered_range", "unit_range"],
)
def test_uniform_distribution_sample_range(low: float, high: float) -> None:
    dist = UniformDistribution(low=low, high=high)
    sample = dist.sample(size=(10, 10))
    assert np.all((sample >= low) & (sample < high))
