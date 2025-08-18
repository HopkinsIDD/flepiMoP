import numpy as np
import pytest
import scipy.stats

from gempyor.likelihoods import PoissonLoglikelihood


@pytest.mark.parametrize("dist_name", ["poisson", "pois"])
def test_poisson_loglikelihood_init_valid(dist_name: str) -> None:
    dist = PoissonLoglikelihood(distribution=dist_name)
    assert dist.distribution == dist_name


def test_poisson_loglikelihood_calculation() -> None:
    dist = PoissonLoglikelihood()
    gt_data = np.array([0, 5, 10, 15])
    model_data = np.array([1.0, 5.5, 9.5, 16.0])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    expected = scipy.stats.poisson.logpmf(k=gt_data, mu=model_data)
    assert np.allclose(result, expected)
