import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import WeibullDistribution


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.5, 5.0),
        (0.5, 1.0),
        (5.0, 20.0),
    ],
    ids=[
        "standard_case",
        "shape_less_than_one",
        "large_values",
    ],
)
def test_weibull_distribution_init_valid(shape: float, scale: float) -> None:
    dist = WeibullDistribution(shape=shape, scale=scale)
    assert dist.shape == shape
    assert dist.scale == scale
    assert dist.distribution == "weibull"


@pytest.mark.parametrize(
    "invalid_shape",
    [0.0, -1.5, -20.0],
    ids=["zero", "small_negative", "large_negative"],
)
def test_weibull_distribution_init_invalid_shape(invalid_shape: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        WeibullDistribution(shape=invalid_shape, scale=1.0)


@pytest.mark.parametrize(
    "invalid_scale",
    [0.0, -2.0, -100.0],
    ids=["zero", "small_negative", "large_negative"],
)
def test_weibull_distribution_init_invalid_scale(invalid_scale: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        WeibullDistribution(shape=1.0, scale=invalid_scale)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((5, 5), (5, 5)),
        (5, (5,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["2d_tuple_size", "integer_size", "3d_tuple_size"],
)
def test_weibull_distribution_sample_properties(size, expected_shape) -> None:
    dist = WeibullDistribution(shape=2.0, scale=10.0)
    sample = dist.sample(size=size)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample > 0)
