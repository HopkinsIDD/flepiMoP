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
    assert dist.allow_edge_cases is False


def test_uniform_init_fails_on_edge_case_false_by_default() -> None:
    with pytest.raises(ValidationError, match="'high' value .* must be > 'low' value"):
        UniformDistribution(low=5.0, high=5.0)


def test_uniform_init_succeeds_on_edge_case_true() -> None:
    dist = UniformDistribution(low=5.0, high=5.0, allow_edge_cases=True)
    assert dist.low == 5.0
    assert dist.high == 5.0
    assert dist.allow_edge_cases is True


def test_uniform_init_fails_on_high_less_than_low() -> None:
    with pytest.raises(ValidationError, match="'high' value .* must be > 'low' value"):
        UniformDistribution(low=10.0, high=5.0, allow_edge_cases=False)
    with pytest.raises(
        ValidationError, match="'high' value .* must be ≥ to the 'low' value"
    ):
        UniformDistribution(low=10.0, high=5.0, allow_edge_cases=True)


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


def test_uniform_distribution_sample_range_edge_case() -> None:
    dist = UniformDistribution(low=10.0, high=10.0, allow_edge_cases=True)
    sample = dist.sample(size=(10, 10))
    assert np.all(sample == 10.0)
