import numpy as np
import pytest
import scipy.stats

from gempyor.utils import get_log_normal


class TestGetLogNormal:
    """Unit tests for the `gempyor.utils.get_log_normal` function."""

    @pytest.mark.parametrize(
        "meanlog,sdlog",
        [
            (1.0, 1.0),
            (0.0, 2.0),
            (10.0, 30.0),
            (0.33, 4.56),
            (9.87, 4.21),
            (1, 1),
            (0, 2),
            (10, 30),
        ],
    )
    def test_construct_distribution(
        self,
        meanlog: float | int,
        sdlog: float | int,
    ) -> None:
        """Test the construction of a log normal distribution.

        This test checks whether the `get_log_normal` function correctly constructs
        a log normal distribution with the specified parameters. It verifies that
        the returned object is an instance of `rv_frozen`, and that its support and
        parameters (log mean and log standard deviation) are correctly set.

        Args:
            mean: The mean of the truncated normal distribution.
            sd: The standard deviation of the truncated normal distribution.
            a: The lower bound of the truncated normal distribution.
            b: The upper bound of the truncated normal distribution.
        """
        dist = get_log_normal(meanlog=meanlog, sdlog=sdlog)
        assert isinstance(dist, scipy.stats._distn_infrastructure.rv_frozen)
        lower, upper = dist.support()
        assert np.isclose(lower, 0.0)
        assert np.isclose(upper, np.inf)
        assert np.isclose(dist.kwds.get("s"), sdlog)
        assert np.isclose(dist.kwds.get("scale"), np.exp(meanlog))
        assert np.isclose(dist.kwds.get("loc"), 0.0)
