"""Unit tests for the `gempyor.statistics.Statistic` class."""

from datetime import date
import re
from typing import Any, Callable, Final, Literal

import confuse
import numpy as np
import pandas as pd
import pytest
import scipy
import xarray as xr

from gempyor.statistics import _AVAILABLE_REGULARIZATIONS, Statistic
from gempyor.testing import create_confuse_configview_from_dict


class MockStatisticInput:
    """
    A representation of the input to the `Statistic` class for testing purposes.

    Attributes:
        name: The name of the statistic.
        config: The configuration dictionary for the statistic.
        model_data: The model data for the statistic.
        gt_data: The ground truth data for the statistic.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        model_data: xr.DataArray | None = None,
        gt_data: xr.DataArray | None = None,
    ) -> None:
        self.name = name
        self.config = config
        self.model_data = model_data
        self.gt_data = gt_data
        self._confuse_subview = None

    def create_confuse_subview(self) -> confuse.Subview:
        """
        Get the `confuse` subview for the statistic configuration.

        Returns:
            The `config` attribute as a confuse subview.
        """
        if self._confuse_subview is None:
            self._confuse_subview = create_confuse_configview_from_dict(
                self.config, name=self.name
            )
        return self._confuse_subview

    def create_statistic_instance(self) -> Statistic:
        """
        Create an instance of the `Statistic` class.

        Returns:
            A `Statistic` instance with this mock input's `name` and `config`.
        """
        return Statistic(self.name, self.create_confuse_subview())


def invalid_regularization_factory() -> MockStatisticInput:
    """
    Create a mock input with an unsupported regularization name.

    Returns:
        A `MockStatisticInput` instance with an unsupported regularization name.
    """
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "rmse"},
            "regularize": [{"name": "forecast"}, {"name": "invalid"}],
        },
    )


def invalid_misshaped_data_factory() -> MockStatisticInput:
    """
    Create a mock input with model and ground truth data of different shapes.

    Returns:
        A `MockStatisticInput` instance with model and ground truth data of different
        shapes.
    """
    model_data = xr.Dataset(
        data_vars={"incidH": (["date", "subpop"], np.random.randn(10, 3))},
        coords={
            "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 10)),
            "subpop": ["01", "02", "03"],
        },
    )
    gt_data = xr.Dataset(
        data_vars={"incidH": (["date", "subpop"], np.random.randn(11, 2))},
        coords={
            "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 11)),
            "subpop": ["02", "03"],
        },
    )
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "norm", "params": {"scale": 2.0}},
        },
        model_data=model_data,
        gt_data=gt_data,
    )


def valid_factory() -> MockStatisticInput:
    """
    Create a mock input with valid configuration and data.

    Returns:
        A `MockStatisticInput` instance with valid configuration and data for 'incidH'
        and 'incidD' using a normal distribution for likelihood.
    """
    data_coords = {
        "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 10)),
        "subpop": ["01", "02", "03"],
    }
    data_dim = [len(v) for v in data_coords.values()]
    model_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    gt_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "norm", "params": {"scale": 2.0}},
        },
        model_data=model_data,
        gt_data=gt_data,
    )


def valid_resample_factory() -> MockStatisticInput:
    """
    Create a mock input with resampling configuration.

    Returns:
        A `MockStatisticInput` instance with configuration for resampling the data with
        a monthly frequency and sum aggregator.
    """
    data_coords = {
        "date": pd.date_range(date(2024, 1, 1), date(2024, 12, 31)),
        "subpop": ["01", "02", "03", "04"],
    }
    data_dim = [len(v) for v in data_coords.values()]
    model_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    gt_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "rmse"},
            "resample": {"freq": "MS", "aggregator": "sum"},
        },
        model_data=model_data,
        gt_data=gt_data,
    )


def valid_scale_factory() -> MockStatisticInput:
    """
    Create a mock input with scaling configuration.

    Returns:
        A `MockStatisticInput` instance with configuration for scaling the data with
        `np.exp`.
    """
    data_coords = {
        "date": pd.date_range(date(2024, 1, 1), date(2024, 12, 31)),
        "subpop": ["01", "02", "03", "04"],
    }
    data_dim = [len(v) for v in data_coords.values()]
    model_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    gt_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "rmse"},
            "scale": "exp",
        },
        model_data=model_data,
        gt_data=gt_data,
    )


def valid_resample_and_scale_factory() -> MockStatisticInput:
    """
    Create a mock input with resampling and scaling configuration.

    Returns:
        A `MockStatisticInput` instance with configuration for resampling the data with
        a weekly frequency and max aggregator, and scaling the data with `np.sin`.
    """
    data_coords = {
        "date": pd.date_range(date(2024, 1, 1), date(2024, 12, 31)),
        "subpop": ["01", "02", "03", "04"],
    }
    data_dim = [len(v) for v in data_coords.values()]
    model_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    gt_data = xr.Dataset(
        data_vars={
            "incidH": (list(data_coords.keys()), np.random.randn(*data_dim)),
            "incidD": (list(data_coords.keys()), np.random.randn(*data_dim)),
        },
        coords=data_coords,
    )
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidD",
            "data_var": "incidD",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "rmse"},
            "resample": {"freq": "W", "aggregator": "max"},
            "scale": "sin",
        },
        model_data=model_data,
        gt_data=gt_data,
    )


def valid_factory_with_pois() -> MockStatisticInput:
    """
    Create a mock input with Poisson distributed likelihood configuration.

    Returns:
        A `MockStatisticInput` instance with configuration for Poisson distributed
        likelihood.
    """
    data_coords = {
        "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 10)),
        "subpop": ["01", "02", "03"],
    }
    data_dim = [len(v) for v in data_coords.values()]
    model_data = xr.Dataset(
        data_vars={
            "incidH": (
                list(data_coords.keys()),
                np.random.poisson(lam=20.0, size=data_dim),
            ),
        },
        coords=data_coords,
    )
    gt_data = xr.Dataset(
        data_vars={
            "incidH": (
                list(data_coords.keys()),
                np.random.poisson(lam=20.0, size=data_dim),
            ),
        },
        coords=data_coords,
    )
    return MockStatisticInput(
        "total_hospitalizations",
        {
            "name": "sum_hospitalizations",
            "sim_var": "incidH",
            "data_var": "incidH",
            "remove_na": True,
            "add_one": True,
            "likelihood": {"dist": "pois"},
        },
        model_data=model_data,
        gt_data=gt_data,
    )


def valid_factory_with_pois_with_some_zeros() -> MockStatisticInput:
    """
    Create a mock input with Poisson distributed likelihood configuration and some zeros.

    Returns:
        A `MockStatisticInput` instance with configuration for Poisson distributed
        likelihood and some zeros in the data. The zeros are strategically located to
        test situations where:
        - The model data has zeros and the ground truth data does not.
        - The ground truth data has zeros and the model data does not.
        - Both the model and ground truth data have zeros.
    """
    mock_input = valid_factory_with_pois()
    mock_input.config["zero_to_one"] = True
    mock_input.model_data["incidH"].loc[
        {
            "date": mock_input.model_data.coords["date"][0],
            "subpop": mock_input.model_data.coords["subpop"][0],
        }
    ] = 0
    mock_input.gt_data["incidH"].loc[
        {
            "date": mock_input.gt_data.coords["date"][2],
            "subpop": mock_input.gt_data.coords["subpop"][2],
        }
    ] = 0
    mock_input.model_data["incidH"].loc[
        {
            "date": mock_input.model_data.coords["date"][1],
            "subpop": mock_input.model_data.coords["subpop"][1],
        }
    ] = 0
    mock_input.gt_data["incidH"].loc[
        {
            "date": mock_input.gt_data.coords["date"][1],
            "subpop": mock_input.gt_data.coords["subpop"][1],
        }
    ] = 0
    return mock_input


def valid_factory_with_nans() -> MockStatisticInput:
    mock_input = valid_factory()
    mock_input.model_data["incidH"].loc[
        {
            "date": mock_input.model_data.coords["date"][0],
            "subpop": mock_input.model_data.coords["subpop"][0],
        }
    ] = np.nan
    mock_input.gt_data["incidH"].loc[
        {
            "date": mock_input.gt_data.coords["date"][2],
            "subpop": mock_input.gt_data.coords["subpop"][2],
        }
    ] = np.nan
    mock_input.model_data["incidH"].loc[
        {
            "date": mock_input.model_data.coords["date"][1],
            "subpop": mock_input.model_data.coords["subpop"][1],
        }
    ] = np.nan
    mock_input.gt_data["incidH"].loc[
        {
            "date": mock_input.gt_data.coords["date"][1],
            "subpop": mock_input.gt_data.coords["subpop"][1],
        }
    ] = np.nan
    return mock_input


def valid_factory_with_nans_date_skipna() -> MockStatisticInput:
    mock_input = valid_factory_with_nans()
    mock_input.config["date_skipna"] = True
    return mock_input


def valid_factory_with_nans_date_min_count() -> MockStatisticInput:
    mock_input = valid_factory_with_nans()
    mock_input.config["date_min_count"] = 8
    mock_input.gt_data["incidH"].loc[
        {
            "date": mock_input.gt_data.coords["date"][1],
            "subpop": mock_input.gt_data.coords["subpop"][2],
        }
    ] = np.nan
    mock_input.gt_data["incidH"].loc[
        {
            "date": mock_input.gt_data.coords["date"][3],
            "subpop": mock_input.gt_data.coords["subpop"][2],
        }
    ] = np.nan
    return mock_input


ALL_VALID_FACTORIES: Final = (
    valid_factory,
    valid_resample_factory,
    valid_scale_factory,
    valid_resample_and_scale_factory,
    valid_factory_with_pois,
    valid_factory_with_pois_with_some_zeros,
    valid_factory_with_nans,
    valid_factory_with_nans_date_skipna,
    valid_factory_with_nans_date_min_count,
)


@pytest.mark.parametrize("factory", (invalid_regularization_factory,))
def test_unsupported_regularizations_value_error(
    factory: Callable[[], MockStatisticInput],
) -> None:
    """Test that an unsupported regularization name raises a `ValueError`."""
    mock_inputs = factory()
    unsupported_name = next(
        reg_name
        for reg_name in [reg["name"] for reg in mock_inputs.config.get("regularize", [])]
        if reg_name not in _AVAILABLE_REGULARIZATIONS
    )
    with pytest.raises(
        ValueError,
        match=(
            f"Given an unsupported regularization name, "
            f"'{unsupported_name}', must be one of:"
        ),
    ):
        mock_inputs.create_statistic_instance()


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
def test_statistic_instance_attributes(factory: Callable[[], MockStatisticInput]) -> None:
    """Test that the `Statistic` instance has the expected attributes."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    assert statistic.data_var == mock_inputs.config["data_var"]
    assert statistic.name == mock_inputs.name
    assert statistic.sim_var == mock_inputs.config["sim_var"]


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
def test_statistic_str_and_repr(factory: Callable[[], MockStatisticInput]) -> None:
    """Test that the `Statistic` instance has the expected `__str__` and `__repr__`."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    for func in (str, repr):
        for var in (
            mock_inputs.name,
            mock_inputs.config["sim_var"],
            mock_inputs.config["data_var"],
        ):
            assert func(var) in func(statistic)
    assert str(mock_inputs.config["likelihood"]["dist"]) in str(statistic)


@pytest.mark.parametrize(
    ("regularization", "factory", "kwargs"),
    (
        ("forecast", valid_factory, {"last_n": 4, "mult": 2.0}),
        ("allsubpop", valid_factory, {"mult": 2.0}),
    ),
)
def test_regularization_output_validation(
    regularization: str, factory: Callable[[], MockStatisticInput], kwargs: dict[str, Any]
) -> None:
    """Test that the regularization methods return a float."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    assert hasattr(statistic, f"_{regularization}_regularize")
    regularization_func = getattr(statistic, f"_{regularization}_regularize")
    regularization_penalty = regularization_func(
        mock_inputs.model_data[mock_inputs.config["sim_var"]],
        mock_inputs.gt_data[mock_inputs.config["data_var"]],
        **kwargs,
    )
    assert isinstance(regularization_penalty, float)


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
@pytest.mark.parametrize("data_type", ("data_var", "sim_var"))
def test_apply_resample(
    factory: Callable[[], MockStatisticInput], data_type: Literal["data_var", "sim_var"]
) -> None:
    """Test that the `apply_resample` method resamples the data as expected."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    resampled_data = statistic.apply_resample(
        mock_inputs.model_data[mock_inputs.config[data_type]]
    )
    if resample_config := mock_inputs.config.get("resample", {}):
        assert (
            resampled_data.shape
            <= mock_inputs.model_data[mock_inputs.config[data_type]].shape
        )
        if resample_config.get("skipna", False):
            assert not resampled_data.isnull().any()
    else:
        assert resampled_data.identical(
            mock_inputs.model_data[mock_inputs.config[data_type]]
        )


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
@pytest.mark.parametrize("data_type", ("data_var", "sim_var"))
def test_apply_scale(
    factory: Callable[[], MockStatisticInput], data_type: Literal["data_var", "sim_var"]
) -> None:
    """Test that the `apply_scale` method scales the data as expected."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    scaled_data = statistic.apply_scale(
        mock_inputs.model_data[mock_inputs.config[data_type]]
    )
    if mock_inputs.config.get("scale") is not None:
        assert (
            scaled_data.shape == mock_inputs.model_data[mock_inputs.config[data_type]].shape
        )
        assert not scaled_data.identical(
            mock_inputs.model_data[mock_inputs.config[data_type]]
        )
        assert scaled_data.isnull().identical(
            mock_inputs.model_data[mock_inputs.config[data_type]].isnull()
        )
    else:
        assert scaled_data.identical(mock_inputs.model_data[mock_inputs.config[data_type]])


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
@pytest.mark.parametrize("data_type", ("data_var", "sim_var"))
def test_apply_transforms(
    factory: Callable[[], MockStatisticInput], data_type: Literal["data_var", "sim_var"]
) -> None:
    """Test that the `apply_transforms` method applies resample then scale transforms."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    transformed_data = statistic.apply_transforms(
        mock_inputs.model_data[mock_inputs.config[data_type]]
    )
    expected_transformed_data = statistic.apply_scale(
        statistic.apply_resample(
            mock_inputs.model_data[mock_inputs.config[data_type]].copy()
        )
    )
    assert transformed_data.identical(expected_transformed_data)


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
def test_llik(factory: Callable[[], MockStatisticInput]) -> None:
    """Test that the `llik` method returns the expected log-likelihood."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    log_likelihood = statistic.llik(
        mock_inputs.model_data[mock_inputs.config["sim_var"]],
        mock_inputs.gt_data[mock_inputs.config["data_var"]],
    )
    assert isinstance(log_likelihood, xr.DataArray)
    assert log_likelihood.dims == mock_inputs.gt_data[mock_inputs.config["data_var"]].dims
    assert log_likelihood.coords.identical(
        mock_inputs.gt_data[mock_inputs.config["data_var"]].coords
    )
    assert np.all(
        (log_likelihood <= 0.0) | np.isclose(log_likelihood, 0.0) | np.isnan(log_likelihood)
    )
    if mock_inputs.config["likelihood"]["dist"] in {"rmse", "absolute_error"}:
        assert (log_likelihood == log_likelihood.values[0, 0]).all()
    assert log_likelihood.isnull().equals(
        mock_inputs.gt_data[mock_inputs.config["data_var"]].isnull()
        | mock_inputs.model_data[mock_inputs.config["sim_var"]].isnull()
    )


@pytest.mark.parametrize("factory", (invalid_misshaped_data_factory,))
def test_compute_logloss_data_misshape_value_error(
    factory: Callable[[], MockStatisticInput],
) -> None:
    """Test `compute_logloss` method raises `ValueError` when data are different shapes."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    model_data_shape = mock_inputs.model_data[mock_inputs.config["sim_var"]].shape
    gt_data_shape = mock_inputs.gt_data[mock_inputs.config["data_var"]].shape
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The `model_data` shape, {model_data_shape}, does not "
            f"match the `gt_data` shape, {gt_data_shape}."
        ),
    ):
        statistic.compute_logloss(mock_inputs.model_data, mock_inputs.gt_data)


@pytest.mark.parametrize("factory", ALL_VALID_FACTORIES)
def test_compute_logloss(factory: Callable[[], MockStatisticInput]) -> None:
    """Test `compute_logloss` method returns expected log-likelihood and regularization."""
    mock_inputs = factory()
    statistic = mock_inputs.create_statistic_instance()
    log_likelihood, regularization = statistic.compute_logloss(
        mock_inputs.model_data, mock_inputs.gt_data
    )
    regularization_config = mock_inputs.config.get("regularize", [])
    assert isinstance(log_likelihood, xr.DataArray)
    assert log_likelihood.coords.identical(
        xr.Coordinates(coords={"subpop": mock_inputs.gt_data.coords.get("subpop")})
    )
    assert isinstance(regularization, float)
    if not regularization_config:
        assert regularization == 0.0
    if mock_inputs.config.get("date_skipna", None) in {None, True}:
        if mock_inputs.config.get("date_min_count", None) is None:
            assert not log_likelihood.isnull().any()
        else:
            ndates, _ = mock_inputs.model_data[mock_inputs.config["sim_var"]].shape
            threshold = ndates - mock_inputs.config["date_min_count"]
            data_meets_threshold = (
                mock_inputs.model_data[mock_inputs.config["sim_var"]].isnull()
                | mock_inputs.gt_data[mock_inputs.config["data_var"]].isnull()
            ).sum("date", skipna=True) > threshold
            assert log_likelihood.isnull().equals(data_meets_threshold)
    else:
        assert log_likelihood.isnull().any().item() == (
            mock_inputs.model_data[mock_inputs.config["sim_var"]].isnull().any().item()
            or mock_inputs.gt_data[mock_inputs.config["data_var"]].isnull().any().item()
        )
