import numpy as np
import numpy.typing as npt
import pytest

from gempyor.likelihoods import LoglikelihoodABC


class DummyLoglikelihood(LoglikelihoodABC):
    """A simple dummy implementation for testing the LoglikelihoodABC logic."""

    distribution: str = "dummy"

    def _loglikelihood(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """
        A predictable dummy llik implementation.
        """
        return -((gt_data - model_data) ** 2)


def test_loglikelihood_abc_wrapper() -> None:
    dist = DummyLoglikelihood(distribution="dummy")
    gt_data = np.array([1, 2, 3, 4, 5])
    model_data = np.array([1, 3, 2, 5, 4])
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    expected = -((gt_data - model_data) ** 2)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "gt_data, model_data",
    [
        (np.array([10.0, 20.0]), np.array([11.0, 19.0])),
        (np.array([-5.0, 0.0]), np.array([0.0, -5.0])),
        (np.array([1.5, 2.5]), np.array([1.5, 2.5])),
    ],
    ids=["positive_floats", "mixed_sign_floats", "identical_data"],
)
def test_loglikelihood_calculation(
    gt_data: npt.NDArray, model_data: npt.NDArray
) -> None:
    dist = DummyLoglikelihood(distribution="dummy")
    result = dist.loglikelihood(gt_data=gt_data, model_data=model_data)
    expected = -((gt_data - model_data) ** 2)
    assert np.allclose(result, expected)