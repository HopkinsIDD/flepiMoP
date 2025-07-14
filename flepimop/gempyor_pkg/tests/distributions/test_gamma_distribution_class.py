import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import GammaDistribution


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 1.5),
        (0.1, 0.1),
        (9.0, 2.0),
        (100.0, 50.0),
    ],
    ids=[
        "standard_case",
        "small_values",
        "numpy_docs_example",
        "large_values",
    ],
)
def test_gamma_distribution_init_valid(shape: float, scale: float) -> None:
    dist = GammaDistribution(shape=shape, scale=scale)
    assert dist.shape == shape
    assert dist.scale == scale
    assert dist.distribution == "gamma"


@pytest.mark.parametrize(
    "invalid_shape",
    [0.0, -0.1, -10.0],
    ids=["zero", "small_negative", "large_negative"],
)
def test_gamma_distribution_init_invalid_shape(invalid_shape: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        GammaDistribution(shape=invalid_shape, scale=1.0)


@pytest.mark.parametrize(
    "invalid_scale",
    [0.0, -1.0, -50.0],
    ids=["zero", "small_negative", "large_negative"],
)
def test_gamma_distribution_init_invalid_scale(invalid_scale: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        GammaDistribution(shape=1.0, scale=invalid_scale)


@pytest.mark.parametrize(
    "shape, scale",
    [
        (2.0, 2.0),
        (1.0, 1.0), 
        (5.0, 0.5),
    ],
    ids=["shape_2_scale_2", "standard_gamma", "shape_5_scale_0.5"],
)
def test_gamma_sample_is_non_negative(shape: float, scale: float) -> None:
    dist = GammaDistribution(shape=shape, scale=scale)
    sample = dist.sample(size=(10, 10))
    assert np.all(sample > 0.0)
