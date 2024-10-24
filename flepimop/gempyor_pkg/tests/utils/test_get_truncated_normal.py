import numpy as np
import pytest
import scipy.stats

from gempyor.utils import get_truncated_normal


class TestGetTruncatedNormal:
    """Unit tests for the `gempyor.utils.get_truncated_normal` function."""

    @pytest.mark.parametrize(
        "mean,sd,a,b",
        [
            (0.0, 1.0, 0.0, 10.0),
            (0.0, 2.0, -4.0, 4.0),
            (-5.0, 3.0, -5.0, 10.0),
            (-3.25, 1.4, -8.74, 4.89),
            (0, 1, 0, 10),
            (0, 2, -4, 4),
            (-5, 3, -5, 10),
        ],
    )
    def test_construct_distribution(
        self,
        mean: float | int,
        sd: float | int,
        a: float | int,
        b: float | int,
    ) -> None:
        """Test the construction of a truncated normal distribution.

        This test checks whether the `get_truncated_normal` function correctly
        constructs a truncated normal distribution with the specified parameters.
        It verifies that the returned object is an instance of `rv_frozen`, and that
        its support and parameters (mean and standard deviation) are correctly set.

        Args:
            mean: The mean of the truncated normal distribution.
            sd: The standard deviation of the truncated normal distribution.
            a: The lower bound of the truncated normal distribution.
            b: The upper bound of the truncated normal distribution.
        """
        dist = get_truncated_normal(mean=mean, sd=sd, a=a, b=b)
        assert isinstance(dist, scipy.stats._distn_infrastructure.rv_frozen)
        lower, upper = dist.support()
        assert np.isclose(lower, a)
        assert np.isclose(upper, b)
        assert np.isclose(dist.kwds.get("loc"), mean)
        assert np.isclose(dist.kwds.get("scale"), sd)
