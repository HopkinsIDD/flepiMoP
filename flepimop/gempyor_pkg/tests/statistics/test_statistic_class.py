from datetime import date
from itertools import product
from typing import Any, Callable

import confuse
import numpy as np
import pandas as pd
import pytest
import scipy
import xarray as xr
import re

from gempyor.statistics import Statistic
from gempyor.testing import create_confuse_configview_from_dict


class MockStatisticInput:
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
        if self._confuse_subview is None:
            self._confuse_subview = create_confuse_configview_from_dict(
                self.config, name=self.name
            )
        return self._confuse_subview

    def create_statistic_instance(self) -> Statistic:
        return Statistic(self.name, self.create_confuse_subview())


def invalid_regularization_factory() -> MockStatisticInput:
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


def simple_valid_factory() -> MockStatisticInput:
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


def simple_valid_resample_factory() -> MockStatisticInput:
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


def simple_valid_scale_factory() -> MockStatisticInput:
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


def simple_valid_resample_and_scale_factory() -> MockStatisticInput:
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


def simple_valid_factory_with_pois() -> MockStatisticInput:
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


def simple_valid_factory_with_pois_with_some_zeros() -> MockStatisticInput:
    mock_input = simple_valid_factory_with_pois()

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


all_valid_factories = [
    (simple_valid_factory),
    (simple_valid_resample_factory),
    (simple_valid_scale_factory),
    (simple_valid_resample_and_scale_factory),
    (simple_valid_factory_with_pois),
    (simple_valid_factory_with_pois_with_some_zeros),
]


class TestStatistic:
    @pytest.mark.parametrize("factory", [invalid_regularization_factory])
    def test_unsupported_regularizations_value_error(
        self, factory: Callable[[], MockStatisticInput]
    ) -> None:
        mock_inputs = factory()
        unsupported_name = next(
            reg_name
            for reg_name in [
                reg["name"] for reg in mock_inputs.config.get("regularize", [])
            ]
            if reg_name not in ["forecast", "allsubpop"]
        )
        with pytest.raises(
            # ValueError, match=rf"^Unsupported regularization \[received: 'invalid'\]"
            ValueError,
            match=(
                f"Given an unsupported regularization name, "
                f"'{unsupported_name}', must be one of:"
            ),
        ):
            mock_inputs.create_statistic_instance()

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_statistic_instance_attributes(
        self, factory: Callable[[], MockStatisticInput]
    ) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # `data_var` attribute
        assert statistic.data_var == mock_inputs.config["data_var"]

        # `dist` attribute
        assert statistic.dist == mock_inputs.config["likelihood"]["dist"]

        # `name` attribute
        assert statistic.name == mock_inputs.name

        # `params` attribute
        assert statistic.params == mock_inputs.config["likelihood"].get("params", {})

        # `regularizations` attribute
        assert statistic.regularizations == [
            (r["name"], r) for r in mock_inputs.config.get("regularize", [])
        ]

        # `resample` attribute
        resample_config = mock_inputs.config.get("resample", {})
        assert statistic.resample == (resample_config != {})

        if resample_config:
            # `resample_aggregator_name` attribute
            assert statistic.resample_aggregator_name == resample_config.get(
                "aggregator", ""
            )

            # `resample_freq` attribute
            assert statistic.resample_freq == resample_config.get("freq", "")

            # `resample_skipna` attribute
            assert (
                statistic.resample_skipna == resample_config.get("skipna", False)
                if resample_config.get("aggregator") is not None
                else False
            )

        # `scale` attribute
        assert statistic.scale == (mock_inputs.config.get("scale") is not None)

        # `scale_func` attribute
        if (scale_func := mock_inputs.config.get("scale")) is not None:
            assert statistic.scale_func == getattr(np, scale_func)

        # `sim_var` attribute
        assert statistic.sim_var == mock_inputs.config["sim_var"]

        # `zero_to_one` attribute
        assert statistic.zero_to_one == mock_inputs.config.get("zero_to_one", False)

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_statistic_str_and_repr(
        self, factory: Callable[[], MockStatisticInput]
    ) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        statistic_str = (
            f"{mock_inputs.name}: {mock_inputs.config['likelihood']['dist']} between "
            f"{mock_inputs.config['sim_var']} (sim) and "
            f"{mock_inputs.config['data_var']} (data)."
        )
        assert str(statistic) == statistic_str
        assert repr(statistic) == f"A Statistic(): {statistic_str}"

    @pytest.mark.parametrize("factory,last_n,mult", [(simple_valid_factory, 4, 2.0)])
    def test_forecast_regularize(
        self, factory: Callable[[], MockStatisticInput], last_n: int, mult: int | float
    ) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        forecast_regularization = statistic._forecast_regularize(
            mock_inputs.model_data[mock_inputs.config["sim_var"]],
            mock_inputs.gt_data[mock_inputs.config["data_var"]],
            last_n=last_n,
            mult=mult,
        )
        assert isinstance(forecast_regularization, float)

    @pytest.mark.parametrize("factory,mult", [(simple_valid_factory, 2.0)])
    def test_allsubpop_regularize(
        self, factory: Callable[[], MockStatisticInput], mult: int | float
    ) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        forecast_regularization = statistic._allsubpop_regularize(
            mock_inputs.model_data[mock_inputs.config["sim_var"]],
            mock_inputs.gt_data[mock_inputs.config["data_var"]],
            mult=mult,
        )
        assert isinstance(forecast_regularization, float)

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_apply_resample(self, factory: Callable[[], MockStatisticInput]) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        resampled_data = statistic.apply_resample(
            mock_inputs.model_data[mock_inputs.config["sim_var"]]
        )
        if resample_config := mock_inputs.config.get("resample", {}):
            # Resample config
            expected_resampled_data = mock_inputs.model_data[
                mock_inputs.config["sim_var"]
            ].resample(date=resample_config.get("freq", ""))
            aggregation_func = getattr(
                expected_resampled_data, resample_config.get("aggregator", "")
            )
            expected_resampled_data = aggregation_func(
                skipna=(
                    resample_config.get("skipna", False)
                    if resample_config.get("aggregator") is not None
                    else False
                )
            )
            assert resampled_data.identical(expected_resampled_data)
        else:
            # No resample config, `apply_resample` returns our input
            assert resampled_data.identical(
                mock_inputs.model_data[mock_inputs.config["sim_var"]]
            )

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_apply_scale(self, factory: Callable[[], MockStatisticInput]) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        scaled_data = statistic.apply_scale(
            mock_inputs.model_data[mock_inputs.config["sim_var"]]
        )
        if (scale_func := mock_inputs.config.get("scale")) is not None:
            # Scale config
            expected_scaled_data = getattr(np, scale_func)(
                mock_inputs.model_data[mock_inputs.config["sim_var"]]
            )
            assert scaled_data.identical(expected_scaled_data)
        else:
            # No scale config, `apply_scale` is a no-op
            assert scaled_data.identical(
                mock_inputs.model_data[mock_inputs.config["sim_var"]]
            )

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_apply_transforms(self, factory: Callable[[], MockStatisticInput]) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        transformed_data = statistic.apply_transforms(
            mock_inputs.model_data[mock_inputs.config["sim_var"]]
        )
        expected_transformed_data = mock_inputs.model_data[
            mock_inputs.config["sim_var"]
        ].copy()
        if resample_config := mock_inputs.config.get("resample", {}):
            # Resample config
            expected_transformed_data = expected_transformed_data.resample(
                date=resample_config.get("freq", "")
            )
            aggregation_func = getattr(
                expected_transformed_data, resample_config.get("aggregator", "")
            )
            expected_transformed_data = aggregation_func(
                skipna=(
                    resample_config.get("skipna", False)
                    if resample_config.get("aggregator") is not None
                    else False
                )
            )
        if (scale_func := mock_inputs.config.get("scale")) is not None:
            # Scale config
            expected_transformed_data = getattr(np, scale_func)(expected_transformed_data)
        assert transformed_data.identical(expected_transformed_data)

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_llik(self, factory: Callable[[], MockStatisticInput]) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        # Tests
        log_likelihood = statistic.llik(
            mock_inputs.model_data[mock_inputs.config["sim_var"]],
            mock_inputs.gt_data[mock_inputs.config["data_var"]],
        )

        assert isinstance(log_likelihood, xr.DataArray)
        assert (
            log_likelihood.dims == mock_inputs.gt_data[mock_inputs.config["data_var"]].dims
        )
        assert log_likelihood.coords.identical(
            mock_inputs.gt_data[mock_inputs.config["data_var"]].coords
        )
        dist_name = mock_inputs.config["likelihood"]["dist"]
        if dist_name == "absolute_error":
            # MAE produces a single repeated number
            assert np.allclose(
                log_likelihood.values,
                -np.log(
                    np.nansum(
                        np.abs(
                            mock_inputs.model_data[mock_inputs.config["sim_var"]]
                            - mock_inputs.gt_data[mock_inputs.config["data_var"]]
                        )
                    )
                ),
            )
        elif dist_name == "rmse":
            assert np.allclose(
                log_likelihood.values,
                -np.log(
                    np.sqrt(
                        np.nansum(
                            np.power(
                                mock_inputs.model_data[mock_inputs.config["sim_var"]]
                                - mock_inputs.gt_data[mock_inputs.config["data_var"]],
                                2.0,
                            )
                        )
                    )
                ),
            )
        elif dist_name == "pois":
            assert np.allclose(
                log_likelihood.values,
                scipy.stats.poisson.logpmf(
                    np.where(
                        mock_inputs.config.get("zero_to_one", False)
                        & (mock_inputs.gt_data[mock_inputs.config["data_var"]].values == 0),
                        1,
                        mock_inputs.gt_data[mock_inputs.config["data_var"]].values,
                    ),
                    np.where(
                        mock_inputs.config.get("zero_to_one", False)
                        & (
                            mock_inputs.model_data[mock_inputs.config["data_var"]].values
                            == 0
                        ),
                        1,
                        mock_inputs.model_data[mock_inputs.config["data_var"]].values,
                    ),
                ),
            )
        elif dist_name in {"norm", "norm_cov"}:
            scale = mock_inputs.config["likelihood"]["params"]["scale"]
            if dist_name == "norm_cov":
                scale *= mock_inputs.model_data[mock_inputs.config["sim_var"]].where(
                    mock_inputs.model_data[mock_inputs.config["sim_var"]] > 5, 5
                )
            assert np.allclose(
                log_likelihood.values,
                scipy.stats.norm.logpdf(
                    mock_inputs.gt_data[mock_inputs.config["data_var"]].values,
                    mock_inputs.model_data[mock_inputs.config["sim_var"]].values,
                    scale=scale,
                ),
            )
        elif dist_name == "nbinom":
            alpha = mock_inputs.config["likelihood"]["params"]["alpha"]
            assert np.allclose(
                log_likelihood.values,
                scipy.stats.nbinom.logpmf(
                    k=mock_inputs.gt_data[mock_inputs.config["data_var"]].values,
                    n=1.0 / alpha,
                    p=1.0
                    / (
                        1.0
                        + alpha
                        * mock_inputs.model_data[mock_inputs.config["sim_var"]].values
                    ),
                ),
            )

    @pytest.mark.parametrize("factory", [invalid_misshaped_data_factory])
    def test_compute_logloss_data_misshape_value_error(
        self, factory: Callable[[], MockStatisticInput]
    ) -> None:
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()

        model_rows, model_cols = mock_inputs.model_data[mock_inputs.config["sim_var"]].shape
        gt_rows, gt_cols = mock_inputs.gt_data[mock_inputs.config["data_var"]].shape
        expected_match = re.escape(
            rf"`model_data` and `gt_data` do not have the same shape: "
            rf"`model_data.shape` = '{mock_inputs.model_data[mock_inputs.config['sim_var']].shape}' "
            rf"!= `gt_data.shape` = '{mock_inputs.gt_data[mock_inputs.config['data_var']].shape}'."
        )
        with pytest.raises(ValueError, match=expected_match):
            statistic.compute_logloss(mock_inputs.model_data, mock_inputs.gt_data)

    @pytest.mark.parametrize("factory", all_valid_factories)
    def test_compute_logloss(self, factory: Callable[[], MockStatisticInput]) -> None:
        # Setup
        mock_inputs = factory()
        statistic = mock_inputs.create_statistic_instance()
        log_likelihood, regularization = statistic.compute_logloss(
            mock_inputs.model_data, mock_inputs.gt_data
        )
        regularization_config = mock_inputs.config.get("regularize", [])

        # Assertions on log_likelihood
        assert isinstance(log_likelihood, xr.DataArray)
        assert log_likelihood.coords.identical(
            xr.Coordinates(coords={"subpop": mock_inputs.gt_data.coords.get("subpop")})
        )

        # Assertions on regularization
        assert isinstance(regularization, float)
        if regularization_config:
            # Regularizations on logistic loss
            assert regularization != 0.0
        else:
            # No regularizations on logistic loss
            assert regularization == 0.0
