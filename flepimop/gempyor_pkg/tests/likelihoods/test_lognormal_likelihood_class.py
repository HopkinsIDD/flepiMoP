import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.likelihoods import LognormalLoglikelihood


@pytest.mark.parametrize(
    "sdlog", [0.1, 1.0, 10.0], ids=["small_sdlog", "unit_sdlog", "large_sdlog"]
)
def test_lognormal_loglikelihood_init_valid(sdlog: float) -> None:
    dist = LognormalLoglikelihood(sdlog=sdlog)
    assert dist.sdlog == sdlog
    assert dist.distribution == "lognorm"


@pytest.mark.parametrize(
    "invalid_sdlog", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_lognormal_loglikelihood_init_invalid_sdlog(invalid_sdlog: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        LognormalLoglikelihood(sdlog=invalid_sdlog)


@pytest.mark.parametrize(
    "sdlog", [0.5, 1.0, 2.0], ids=["sdlog_0.5", "sdlog_1.0", "sdlog_2.0"]
)
def test_lognormal_loglikelihood_calculation(sdlog: float) -> None:
    dist = LognormalLoglikelihood(sdlog=sdlog)
    gt_data = np.array([10, 15, 20, 25])
    model_data = np.array([11, 14, 22, 25])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    expected = scipy.stats.lognorm.logpdf(x=gt_data, s=sdlog, scale=model_data)
    assert np.allclose(result, expected)
