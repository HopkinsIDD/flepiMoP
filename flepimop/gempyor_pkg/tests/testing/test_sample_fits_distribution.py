from typing import Any, Literal

import pytest

from gempyor.testing import sample_fits_distribution


class TestSampleFitsDistribution:
    @pytest.mark.parametrize(
        "sample,distribution,kwargs,expected",
        [
            # Fixed distribution
            (0.5, "fixed", {"value": 0.5}, True),
            (0.5, "fixed", {"value": 0.6}, False),
            (1, "fixed", {"value": 1}, True),
            (1.0, "fixed", {"value": 1.0}, True),
            (1, "fixed", {"value": 1.0}, True),
            (1.0, "fixed", {"value": 1}, True),
            (0.0000001, "fixed", {"value": 0.0}, False),
            (0.00000001, "fixed", {"value": 0.0}, True),
            # Uniform distribution
            (0.5, "uniform", {"low": 0.5, "high": 0.5}, True),
            (0.5, "uniform", {"low": 0.0, "high": 1.0}, True),
            (0.0, "uniform", {"low": 0.0, "high": 1.0}, True),
            (1.0, "uniform", {"low": 0.0, "high": 1.0}, False),
            (-0.1, "uniform", {"low": 0.0, "high": 1.0}, False),
            # Poisson distribution
            (0.5, "poisson", {"lam": 1.0}, False),
            (1.0, "poisson", {"lam": 1.5}, True),
            (1, "poisson", {"lam": 1.5}, True),
            (-1.0, "poisson", {"lam": 1.0}, False),
            (-1, "poisson", {"lam": 1.0}, False),
            (9999.0, "poisson", {"lam": 0.1}, True),  # Extremely unlikely
            # Binomial distribution
            (0.5, "binomial", {"n": 10, "p": 0.5}, False),
            (1.0, "binomial", {"n": 10, "p": 0.5}, True),
            (1, "binomial", {"n": 10, "p": 0.5}, True),
            (-1.0, "binomial", {"n": 5, "p": 0.75}, False),
            (-1, "binomial", {"n": 5, "p": 0.75}, False),
            (0, "binomial", {"n": 45, "p": 0.1}, True),
            (0.0, "binomial", {"n": 45, "p": 0.1}, True),
            (1000.0, "binomial", {"n": 1000, "p": 0.001}, True),  # Extremely unlikely
            (0, "binomial", {"n": 1000, "p": 0.999}, True),
            # Truncated normal distribution
            (-0.5, "truncnorm", {"a": -3.0, "b": 3.0, "mean": 0.0, "sd": 1.0}, True),
            (-3.5, "truncnorm", {"a": -3.0, "b": 3.0, "mean": 0.0, "sd": 1.0}, False),
            (3.1, "truncnorm", {"a": -3.0, "b": 3.0, "mean": 0.0, "sd": 1.0}, False),
            (  # Extremely unlikely
                99.9,
                "truncnorm",
                {"a": -100.0, "b": 100.0, "mean": 0.0, "sd": 1.0},
                True,
            ),
            # Log-normal distribution
            (1.1, "lognorm", {"meanlog": 1.0, "sdlog": 1.0}, True),
            (-0.5, "lognorm", {"meanlog": 2.0, "sdlog": 2.0}, False),
            (-7.8, "lognorm", {"meanlog": 3.4, "sdlog": 5.6}, False),
            (0.0, "lognorm", {"meanlog": 1.2, "sdlog": 3.4}, False),
            (0.0000001, "lognorm", {"meanlog": 1.2, "sdlog": 3.4}, True),
            (  # Extremely unlikely
                99999.9,
                "lognorm",
                {"meanlog": 1.2, "sdlog": 3.4},
                True,
            ),
        ],
    )
    def test_output_validation(
        self,
        sample: float | int,
        distribution: Literal[
            "fixed", "uniform", "poisson", "binomial", "truncnorm", "lognorm"
        ],
        kwargs: dict[str, Any],
        expected: bool,
    ) -> None:
        assert sample_fits_distribution(sample, distribution, **kwargs) == expected
