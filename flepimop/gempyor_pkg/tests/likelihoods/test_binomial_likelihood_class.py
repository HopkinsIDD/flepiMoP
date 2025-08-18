import numpy as np
import pytest
import scipy.stats
from pydantic import ValidationError

from gempyor.likelihoods import BinomialLoglikelihood


@pytest.mark.parametrize("n", [0, 1, 50], ids=["zero_n", "one_n", "large_n"])
def test_binomial_loglikelihood_init_valid(n: int) -> None:
    dist = BinomialLoglikelihood(n=n)
    assert dist.n == n
    assert dist.distribution == "binomial"


@pytest.mark.parametrize("invalid_n", [-1, -10], ids=["-1", "-10"])
def test_binomial_loglikelihood_init_invalid_n(invalid_n: int) -> None:
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        BinomialLoglikelihood(n=invalid_n)


def test_binomial_loglikelihood_calculation() -> None:
    n_trials = 20
    dist = BinomialLoglikelihood(n=n_trials)
    gt_data = np.array([0, 5, 10, 15, 20])
    model_data = np.array([-0.5, 0.25, 0.5, 0.75, 1.5])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    clipped_p = np.clip(model_data, 0, 1)
    expected = scipy.stats.binom.logpmf(k=gt_data, n=n_trials, p=clipped_p)
    assert np.allclose(result, expected)
