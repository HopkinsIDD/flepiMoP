import numpy as np
import pytest

from gempyor.distributions import FixedDistribution


@pytest.mark.parametrize(
    "value", [123.45, -100.0, 0.0], ids=["positive", "negative", "zero"]
)
def test_fixed_distribution_init(value: float) -> None:
    dist = FixedDistribution(value=value)
    assert dist.value == value
    assert dist.distribution == "fixed"


@pytest.mark.parametrize("value", [0.0, 5.5, -1.2], ids=["zero", "positive", "negative"])
def test_fixed_distribution_samples_values(value: float) -> None:
    dist = FixedDistribution(value=value)
    sample = dist.sample(size=(5, 5))
    assert np.all(sample == value)
