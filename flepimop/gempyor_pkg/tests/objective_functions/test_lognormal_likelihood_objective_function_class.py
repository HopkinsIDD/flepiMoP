import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.objective_functions import LognormalLoglikelihood


@pytest.mark.parametrize(
    "sdlog", [0.1, 1.0, 10.0], ids=["small_sdlog", "unit_sdlog", "large_sdlog"]
)
def test_lognormal_loglikelihood_init_valid(sdlog: float) -> None:
    dist = LognormalLoglikelihood(sdlog=sdlog)
    assert dist.sigmalog == sdlog
    assert dist.distribution == "lognorm"


@pytest.mark.parametrize(
    "invalid_sdlog", [0.0, -0.1, -10.0], ids=["zero", "small_negative", "large_negative"]
)
def test_lognormal_loglikelihood_init_invalid_sdlog(invalid_sdlog: float) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than 0"):
        LognormalLoglikelihood(sdlog=invalid_sdlog)


@pytest.mark.parametrize(
    "sdlog, gt_data, model_data",
    [
        (
            0.5,
            np.array([10, 15, 20, 25]),
            np.array([11, 14, 22, 25]),
        ),
        (
            2.0,
            np.array([10, 15, 20, 25]),
            np.array([11, 14, 22, 25]),
        ),
        (
            1.0,
            np.array([15, 20]),
            np.array([15, 20]),
        ),
    ],
    ids=["original_case", "high_variance", "perfect_match"],
)
def test_lognormal_loglikelihood_error_metric_calculation(
    sdlog: float,
    gt_data: np.ndarray,
    model_data: np.ndarray,
) -> None:
    dist = LognormalLoglikelihood(sigmalog=sdlog)
    result = dist.error_metric_calculation(gt_data=gt_data, model_data=model_data)
    expected_scale = model_data / np.exp(sdlog**2 / 2)
    expected = scipy.stats.lognorm.logpdf(x=gt_data, s=sdlog, scale=expected_scale)

    assert np.allclose(result, expected, atol=1e-5)
