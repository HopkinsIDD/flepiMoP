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
    "size, expected_shape",
    [
        ((6, 4), (6, 4)),
        (20, (20,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["2d_tuple_size", "integer_size", "3d_tuple_size"],
)
def test_uniform_distribution_sample_properties(size, expected_shape) -> None:
    low, high = 10.0, 20.0
    dist = UniformDistribution(low=low, high=high)
    sample = dist.sample(size=size)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample >= low)
    assert np.all(sample < high)
