import numpy as np
import pytest

from gempyor.distributions import FixedDistribution


@pytest.mark.parametrize(
    "value", [123.45, -100.0, 0.0], ids=["positive", "negative", "zero"]
)
def test_fixed_distribution_ini(value: float) -> None:
    dist = FixedDistribution(value=value)
    assert dist.value == value
    assert dist.distribution == "fixed"


@pytest.mark.parametrize("value", [5.5, -1.2], ids=["value_positive", "value_negative"])
@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((2, 5), (2, 5)),
        (5, (5,)),
        ((2, 3), (2, 3)),
    ],
    ids=["default_size", "integer_size", "tuple_size"],
)
@pytest.mark.parametrize("use_rng", [True, False], ids=["with_rng", "without_rng"])
def test_fixed_distribution_sample_properties(
    value: float, size, expected_shape, use_rng
) -> None:
    dist = FixedDistribution(value=value)
    kwargs = {}
    kwargs["size"] = size
    if use_rng:
        kwargs["rng"] = np.random.default_rng()
    sample = dist.sample(**kwargs)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample == value)
