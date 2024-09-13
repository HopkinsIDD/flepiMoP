from functools import partial
import inspect
from typing import Any

import numpy as np
import pytest

from gempyor.testing import partials_are_similar
from gempyor.utils import random_distribution_sampler


class TestRandomDistributionSampler:
    @pytest.mark.parametrize("distribution", [("abc"), ("def"), ("ghi")])
    def test_not_implemented_error_exception(self, distribution: str) -> None:
        with pytest.raises(
            NotImplementedError,
            match=rf"^unknown distribution \[got\: {distribution}\]$",
        ):
            random_distribution_sampler(distribution)

    @pytest.mark.parametrize("p", [(-0.5), (1.2), (0.0), (1.0)])
    def test_binomial_p_value_error(self, p: float) -> None:
        with pytest.raises(
            ValueError,
            match=rf"^p value {p} is out of range \[0\,1\]$",
        ):
            random_distribution_sampler("binomial", n=100, p=p)

    @pytest.mark.parametrize(
        "distribution,kwargs",
        [
            ("fixed", {"value": 0.12}),
            ("fixed", {"value": -3.45}),
            ("fixed", {"value": 0.0}),
            ("uniform", {"low": 0.0, "high": 1.0}),
            ("uniform", {"low": 50.0, "high": 200.0}),
            ("uniform", {"low": -1.0, "high": 1.0}),
            ("uniform", {"low": 1.0, "high": -1.0}),
            ("poisson", {"lam": 0.1}),
            ("poisson", {"lam": 1.23}),
            ("poisson", {"lam": -0.1}),
            ("binomial", {"n": 10, "p": 0.1}),
            ("binomial", {"n": -10, "p": 0.1}),
            ("binomial", {"n": 50, "p": 0.67}),
            ("truncnorm", {"mean": 0.0, "sd": 1.0, "a": -2.0, "b": 2.0}),
            ("truncnorm", {"mean": 1.4, "sd": 0.34, "a": -0.3, "b": 22.8}),
            ("lognorm", {"sdlog": 1.0, "meanlog": 1.0}),
            ("lognorm", {"sdlog": 3.4, "meanlog": -0.56}),
        ],
    )
    def test_output_validation(self, distribution: str, kwargs: dict[str, Any]) -> None:
        actual = random_distribution_sampler(distribution, **kwargs)
        if distribution == "fixed":
            expected = partial(
                np.random.uniform, kwargs.get("value"), kwargs.get("value")
            )
            assert partials_are_similar(actual, expected)
        elif distribution == "uniform":
            expected = partial(np.random.uniform, kwargs.get("low"), kwargs.get("high"))
            assert partials_are_similar(actual, expected)
        elif distribution == "poisson":
            expected = partial(np.random.poisson, kwargs.get("lam"))
            assert partials_are_similar(actual, expected)
        elif distribution == "binomial":
            expected = partial(np.random.binomial, kwargs.get("n"), kwargs.get("p"))
            assert partials_are_similar(actual, expected)
        elif distribution == "truncnorm":
            assert inspect.ismethod(actual)
            assert actual.__self__.kwds.get("loc") == kwargs.get("mean")
            assert actual.__self__.kwds.get("scale") == kwargs.get("sd")
            assert actual.__self__.a == (
                kwargs.get("a") - kwargs.get("mean")
            ) / kwargs.get("sd")
            assert actual.__self__.b == (
                kwargs.get("b") - kwargs.get("mean")
            ) / kwargs.get("sd")
        elif distribution == "lognorm":
            assert inspect.ismethod(actual)
            assert actual.__self__.kwds.get("s") == kwargs.get("sdlog")
            assert actual.__self__.kwds.get("scale") == np.exp(kwargs.get("meanlog"))
