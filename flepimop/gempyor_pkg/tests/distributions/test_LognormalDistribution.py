import numpy as np
import pytest
from pydantic import ValidationError

from gempyor.distributions import LognormalDistribution


@pytest.mark.parametrize(
    "meanlog", [-10.0, 0.0, 5.0], ids=["meanlog_neg", "meanlog_zero", "meanlog_pos"]
)
@pytest.mark.parametrize(
    "sdlog", [0.1, 1.0, 10.0], ids=["sdlog_small", "sdlog_one", "sdlog_large"]
)
def test_lognormal_distribution_init_valid(meanlog: float, sdlog: float) -> None:
    dist = LognormalDistribution(meanlog=meanlog, sdlog=sdlog)
    assert dist.meanlog == meanlog
    assert dist.sdlog == sdlog
    assert dist.distribution == "lognorm"


@pytest.mark.parametrize(
    "invalid_sdlog", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_lognormal_distribution_init_raises_error_for_invalid_sdlog(
    invalid_sdlog: float,
) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        LognormalDistribution(meanlog=0.0, sdlog=invalid_sdlog)


@pytest.mark.parametrize(
    "size, expected_shape",
    [
        ((2, 3), (2, 3)),
        (25, (25,)),
        ((2, 3, 4), (2, 3, 4)),
    ],
    ids=["tuple_size1", "integer_size", "tuple_size2"],
)
@pytest.mark.parametrize("use_rng", [True, False], ids=["with_rng", "without_rng"])
def test_lognormal_distribution_sample_properties(size, expected_shape, use_rng) -> None:
    dist = LognormalDistribution(meanlog=0.0, sdlog=1.0)
    kwargs = {}
    kwargs["size"] = size
    if use_rng:
        kwargs["rng"] = np.random.default_rng()
    sample = dist.sample(**kwargs)
    assert isinstance(sample, np.ndarray)
    assert sample.shape == expected_shape
    assert sample.dtype == np.float64
    assert np.all(sample > 0)


def test_lognormal_distribution_sample_rng_reproducibility() -> None:
    dist = LognormalDistribution(meanlog=2.0, sdlog=0.5)
    rng1 = np.random.default_rng(seed=100)
    sample1 = dist.sample(size=(3, 3), rng=rng1)
    rng2 = np.random.default_rng(seed=100)
    sample2 = dist.sample(size=(3, 3), rng=rng2)
    np.testing.assert_array_equal(sample1, sample2)
