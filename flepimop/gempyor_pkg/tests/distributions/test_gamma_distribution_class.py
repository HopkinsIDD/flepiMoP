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
    "size, expected_shape",
    [
        ((3, 3), (3, 3)),
        (5, (5,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["2d_tuple_size", "integer_size", "3d_tuple_size"],
)
@pytest.mark.parametrize("use_rng", [True, False], ids=["with_rng", "without_rng"])
def test_gamma_distribution_sample_properties(size, expected_shape, use_rng) -> None:
    dist = GammaDistribution(shape=2.0, scale=2.0)
    kwargs = {"size": size}
    if use_rng:
        kwargs["rng"] = np.random.default_rng()
    sample = dist.sample(**kwargs)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample > 0)


def test_gamma_distribution_sample_rng_reproducibility() -> None:
    dist = GammaDistribution(shape=9.0, scale=0.5)
    rng1 = np.random.default_rng(seed=100)
    sample1 = dist.sample(size=(4, 4), rng=rng1)
    rng2 = np.random.default_rng(seed=100)
    sample2 = dist.sample(size=(4, 4), rng=rng2)
    np.testing.assert_array_equal(sample1, sample2)
