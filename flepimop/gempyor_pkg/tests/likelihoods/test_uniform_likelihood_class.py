import numpy as np
import pytest
import scipy.stats

from gempyor.likelihoods import UniformLoglikelihood


@pytest.mark.parametrize(
    "low, high",
    [
        (0.0, 10.0),
        (-10.0, -5.0),
        (-5.0, 5.0),
    ],
    ids=["positive_range", "negative_range", "spanning_range"],
)
def test_uniform_loglikelihood_init_valid(low: float, high: float) -> None:
    dist = UniformLoglikelihood(low=low, high=high)
    assert dist.low == low
    assert dist.high == high
    assert dist.distribution == "uniform"


@pytest.mark.parametrize(
    "low, high",
    [
        (0.0, 1.0),
        (-10.0, 10.0),
    ],
    ids=["unit_range", "wide_range"],
)
def test_uniform_loglikelihood_calculation(low: float, high: float) -> None:
    dist = UniformLoglikelihood(low=low, high=high)
    gt_data = np.array([10, 15, 20, 25, 30])
    model_data = np.array([11, 14, 22, 25, 28])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    loc = model_data - ((high - low) / 2.0)
    scale = high - low
    expected = scipy.stats.uniform.logpdf(x=gt_data, loc=loc, scale=scale)
    assert np.allclose(result, expected)
