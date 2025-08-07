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
    "meanlog, sdlog",
    [
        (0.0, 1.0),
        (-2.0, 0.5),
        (5.0, 3.0),
    ],
    ids=["standard", "low_variance", "high_variance"],
)
def test_lognormal_sample_is_non_negative(meanlog: float, sdlog: float) -> None:
    dist = LognormalDistribution(meanlog=meanlog, sdlog=sdlog)
    sample = dist.sample(size=(10, 10))
    assert np.all(sample > 0.0)
