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
    """
    Tests that creating a UniformDistribution with invalid bounds (high <= low)
    raises a Pydantic ValidationError, triggering the custom validator.
    """
    # The custom validator raises a ValueError, which Pydantic wraps.
    # We can match the custom error message.
    with pytest.raises(ValidationError, match="must be greater than the 'low' value"):
        UniformDistribution(low=low, high=high)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((6, 4), (6, 4)),  # Case for the default size
        (20, (20,)),  # Case for an integer size
        ((4, 5), (4, 5)),  # Case for a tuple size
    ],
    ids=["default_size", "integer_size", "tuple_size"],
)
@pytest.mark.parametrize("use_rng", [True, False], ids=["with_rng", "without_rng"])
def test_uniform_distribution_sample_properties(size, expected_shape, use_rng) -> None:
    low, high = 10.0, 20.0
    dist = UniformDistribution(low=low, high=high)
    kwargs = {}
    kwargs["size"] = size
    if use_rng:
        kwargs["rng"] = np.random.default_rng()
    sample = dist.sample(**kwargs)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample >= low)
    assert np.all(sample < high)


def test_uniform_distribution_sample_rng_reproducibility() -> None:
    dist = UniformDistribution(low=-10.0, high=10.0)

    rng1 = np.random.default_rng(seed=123)
    sample1 = dist.sample(size=(3, 3), rng=rng1)

    rng2 = np.random.default_rng(seed=123)
    sample2 = dist.sample(size=(3, 3), rng=rng2)

    assert np.array_equal(sample1, sample2)
