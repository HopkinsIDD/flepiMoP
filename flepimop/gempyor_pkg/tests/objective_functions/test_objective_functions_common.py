import numpy as np
import numpy.typing as npt
import pytest

from pydantic import Field, AliasChoices

from gempyor.objective_functions import ObjectiveFunctionABC


class ObjectiveFunction(ObjectiveFunctionABC):
    """A simple dummy implementation for testing the ObjectiveFunctionABC logic."""

    distribution: str = Field(validation_alias=AliasChoices("distribution", "dist"))

    def _error_metric_calculation(
        self, gt_data: npt.NDArray, model_data: npt.NDArray
    ) -> npt.NDArray:
        """
        A predictable dummy error metric calculation implementation.
        """
        return -((gt_data - model_data) ** 2)


def test_loglikelihood_abc_wrapper() -> None:
    dist = ObjectiveFunction(distribution="dummy")
    gt_data = np.array([1, 2, 3, 4, 5])
    model_data = np.array([1, 3, 2, 5, 4])
    result = dist._error_metric_calculation(gt_data=gt_data, model_data=model_data)
    expected = -((gt_data - model_data) ** 2)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, expected)


def test_objective_function_init_with_alias() -> None:
    dist = ObjectiveFunction(dist="dummy")
    assert dist.distribution == "dummy"


@pytest.mark.parametrize(
    "gt_data, model_data",
    [
        (np.array([10.0, 20.0]), np.array([11.0, 19.0])),
        (np.array([-5.0, 0.0]), np.array([0.0, -5.0])),
        (np.array([1.5, 2.5]), np.array([1.5, 2.5])),
    ],
    ids=["positive_floats", "mixed_sign_floats", "identical_data"],
)
def test_error_metric_calculation(gt_data: npt.NDArray, model_data: npt.NDArray) -> None:
    dist = ObjectiveFunction(distribution="dummy")
    result = dist._error_metric_calculation(gt_data=gt_data, model_data=model_data)
    expected = -((gt_data - model_data) ** 2)
    assert np.allclose(result, expected)
