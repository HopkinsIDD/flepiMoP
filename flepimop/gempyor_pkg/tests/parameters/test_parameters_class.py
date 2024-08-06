from datetime import date
from functools import partial
import pathlib
from typing import Any, Callable
from uuid import uuid4

import confuse
import numpy as np
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile

from gempyor.parameters import Parameters
from gempyor.testing import (
    create_confuse_subview_from_dict,
    partials_are_similar,
    sample_fits_distribution,
)
from gempyor.utils import random_distribution_sampler


class MockData:
    simple_timeseries_param_df = pd.DataFrame(
        data={
            "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
            "1": [1.2, 2.3, 3.4, 4.5, 5.6],
            "2": [2.3, 3.4, 4.5, 5.6, 6.7],
        }
    )

    simple_inputs = {
        "parameter_config": create_confuse_subview_from_dict(
            "parameters", {"sigma": {"value": 0.1}}
        ),
        "ti": date(2024, 1, 1),
        "tf": date(2024, 1, 10),
        "subpop_names": ["1", "2", "3"],
    }

    small_inputs = {
        "parameter_config": create_confuse_subview_from_dict(
            "parameters",
            {"sigma": {"value": 0.1}, "gamma": {"value": 0.2}, "eta": {"value": 0.3}},
        ),
        "ti": date(2024, 1, 1),
        "tf": date(2024, 1, 31),
        "subpop_names": ["1", "2"],
    }

    empty_param_overrides_df = pd.DataFrame(data={"parameter": [], "value": []})


class MockParametersInput:
    def __init__(
        self,
        config: dict[str, Any],
        ti: date,
        tf: date,
        subpop_names: list[str],
        path_prefix: str = ".",
    ) -> None:
        self.config = config
        self.ti = ti
        self.tf = tf
        self.subpop_names = subpop_names
        self.path_prefix = path_prefix
        self._timeseries_dfs = {}
        self._confuse_subview = None

    def create_confuse_subview(self) -> confuse.Subview:
        if self._confuse_subview is None:
            self._confuse_subview = create_confuse_subview_from_dict(
                "parameters", self.config
            )
        return self._confuse_subview

    def create_parameters_instance(self) -> Parameters:
        return Parameters(
            parameter_config=self.create_confuse_subview(),
            ti=self.ti,
            tf=self.tf,
            subpop_names=self.subpop_names,
            path_prefix=self.path_prefix,
        )

    def number_of_subpops(self) -> int:
        return len(self.subpop_names)

    def number_of_days(self) -> int:
        return (self.tf - self.ti).days

    def number_of_parameters(self) -> int:
        return len(self.config)

    def get_timeseries_parameters(self) -> list[str]:
        return [k for k, v in self.config.items() if "timeseries" in v]

    def get_nontimeseries_parameters(self) -> list[str]:
        return [k for k, v in self.config.items() if "timeseries" not in v]

    def number_of_nontimeseries_parameters(self) -> int:
        return len(self.get_nontimeseries_parameters())

    def has_timeseries_parameter(self) -> bool:
        for _, v in self.config.items():
            if "timeseries" in v:
                return True
        return False

    def get_timeseries_df(self, param_name: str) -> pd.DataFrame:
        df = self._timeseries_dfs.get(param_name)
        if df is not None:
            return df.copy()
        conf = self.config.get(param_name, {})
        df_file = conf.get("timeseries")
        if df_file is None:
            raise ValueError(
                f"The given param '{param_name}' does not have a timeseries dataframe."
            )
        df = pd.read_csv(df_file, index_col=None)
        df["date"] = pd.to_datetime(df["date"])
        self._timeseries_dfs[param_name] = df
        return df.copy()


def fixed_three_valid_parameter_factory(tmp_path: pathlib.Path) -> MockParametersInput:
    return MockParametersInput(
        config={"sigma": {"value": 0.1}, "eta": {"value": 0.2}, "nu": {"value": 0.3}},
        ti=date(2024, 1, 1),
        tf=date(2024, 1, 31),
        subpop_names=["1", "2", "3"],
    )


def distribution_three_valid_parameter_factory(
    tmp_path: pathlib.Path,
) -> MockParametersInput:
    return MockParametersInput(
        config={
            "sigma": {"value": {"distribution": "uniform", "low": 1.0, "high": 2.0}},
            "eta": {"value": {"distribution": "binomial", "n": 20, "p": 0.5}},
            "nu": {"value": {"distribution": "lognorm", "meanlog": 1.0, "sdlog": 1.0}},
        },
        ti=date(2024, 1, 1),
        tf=date(2024, 1, 31),
        subpop_names=["1", "2", "3"],
    )


def valid_parameters_factory(tmp_path: pathlib.Path) -> MockParametersInput:
    tmp_file = tmp_path / f"{uuid4().hex}.csv"
    df = MockData.simple_timeseries_param_df.copy()
    df.to_csv(tmp_file, index=False)
    return MockParametersInput(
        config={
            "sigma": {"timeseries": str(tmp_file.absolute())},
            "gamma": {"value": 0.1234, "stacked_modifier_method": "sum"},
            "Ro": {"value": {"distribution": "uniform", "low": 1.0, "high": 2.0}},
        },
        ti=df["date"].dt.date.min(),
        tf=df["date"].dt.date.max(),
        subpop_names=[c for c in df.columns.to_list() if c != "date"],
    )


class TestParameters:
    def test_nonunique_parameter_names_value_error(self) -> None:
        duplicated_parameters = create_confuse_subview_from_dict(
            "parameters",
            {"sigma": {"value": 0.1}, "gamma": {"value": 0.2}, "GAMMA": {"value": 0.3}},
        )
        with pytest.raises(
            ValueError,
            match=(
                r"Parameters of the SEIR model have the same name "
                r"\(remember that case is not sufficient\!\)"
            ),
        ):
            Parameters(
                duplicated_parameters,
                ti=date(2024, 1, 1),
                tf=date(2024, 12, 31),
                subpop_names=["1", "2"],
            )

    def test_timeseries_parameter_has_insufficient_columns_value_error(self) -> None:
        param_df = pd.DataFrame(
            data={
                "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
                "1": [1.2, 2.3, 3.4, 4.5, 5.6],
                "2": [2.3, 3.4, 4.5, 5.6, 6.7],
            }
        )
        with NamedTemporaryFile(suffix=".csv") as temp_file:
            param_df.to_csv(temp_file.name, index=False)
            invalid_timeseries_parameters = create_confuse_subview_from_dict(
                "parameters", {"sigma": {"timeseries": temp_file.name}}
            )
            with pytest.raises(
                ValueError,
                match=(
                    rf"ERROR loading file {temp_file.name} for parameter sigma\: "
                    rf"the number of non 'date'\s+columns are 2, expected 3 "
                    rf"\(the number of subpops\) or one\."
                ),
            ):
                Parameters(
                    invalid_timeseries_parameters,
                    ti=date(2024, 1, 1),
                    tf=date(2024, 1, 5),
                    subpop_names=["1", "2", "3"],
                )

    @pytest.mark.parametrize(
        "start_date,end_date,timeseries_df",
        [(date(2024, 1, 1), date(2024, 1, 6), MockData.simple_timeseries_param_df)],
    )
    def test_timeseries_parameter_has_insufficient_dates_value_error(
        self, start_date: date, end_date: date, timeseries_df: pd.DataFrame
    ) -> None:
        # First way to get at this error, purely a length difference
        with NamedTemporaryFile(suffix=".csv") as temp_file:
            timeseries_df.to_csv(temp_file.name, index=False)
            invalid_timeseries_parameters = create_confuse_subview_from_dict(
                "parameters", {"sigma": {"timeseries": temp_file.name}}
            )
            timeseries_start_date = timeseries_df["date"].dt.date.min()
            timeseries_end_date = timeseries_df["date"].dt.date.max()
            subpop_names = [c for c in timeseries_df.columns.to_list() if c != "date"]
            with pytest.raises(
                ValueError,
                match=(
                    rf"ERROR loading file {temp_file.name} for parameter sigma\:\s+"
                    rf"the \'date\' entries of the provided file do not include all the"
                    rf" days specified to be modeled by\s+the config\. the provided "
                    rf"file includes 5 days between {timeseries_start_date}"
                    rf"( 00\:00\:00)? to {timeseries_end_date}( 00\:00\:00)?,\s+while "
                    rf"there are 6 days in the config time span of {start_date}->"
                    rf"{end_date}\. The file must contain entries for the\s+the exact "
                    rf"start and end dates from the config\. "
                ),
            ):
                Parameters(
                    invalid_timeseries_parameters,
                    ti=start_date,
                    tf=end_date,
                    subpop_names=subpop_names,
                )

        # TODO: I'm not sure how to get to the second pathway to this error message.
        # 1) We subset the read in dataframe to `ti` to `tf` so if the dataframe goes
        # from 2024-01-01 through 2024-01-05 and the given date range is 2024-01-02
        # through 2024-01-06 the dataframe's date range will be subsetted to 2024-01-02
        # through 2024-01-05 which is a repeat of the above.
        # 2) Because of the subsetting you can't provide anything except a monotonic
        # increasing sequence of dates, pandas only allows subsetting on ordered date
        # indexes so you'll get a different error.
        # 3) If you provide a monotonic increasing sequence of dates but 'reverse' `ti`
        # and `tf` you get no errors (which I think is also bad) because the slice
        # operation returns an empty dataframe with the right columns & index and the
        # `pd.date_range` function only creates monotonic increasing sequences and
        # 0 == 0.

    @pytest.mark.parametrize("factory", [(valid_parameters_factory)])
    def test_parameters_instance_attributes(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
    ) -> None:
        # Setup
        mock_inputs = factory(tmp_path)
        params = mock_inputs.create_parameters_instance()

        # The `npar` attribute
        assert params.npar == mock_inputs.number_of_parameters()

        # The `pconfig` attribute
        assert params.pconfig == mock_inputs.create_confuse_subview()

        # The `pdata` attribute
        assert set(params.pdata.keys()) == set(mock_inputs.config.keys())
        for param_name, param_conf in mock_inputs.config.items():
            assert params.pdata[param_name]["idx"] == params.pnames2pindex[param_name]
            assert params.pdata[param_name][
                "stacked_modifier_method"
            ] == param_conf.get("stacked_modifier_method", "product")
            if "timeseries" in param_conf:
                assert params.pdata[param_name]["ts"].equals(
                    mock_inputs.get_timeseries_df(param_name).set_index("date")
                )
            elif isinstance(params.pdata[param_name]["dist"], partial):
                if isinstance(param_conf.get("value"), float):
                    expected = random_distribution_sampler(
                        "fixed", value=param_conf.get("value")
                    )
                else:
                    expected = random_distribution_sampler(
                        param_conf.get("value").get("distribution"),
                        **{
                            k: v
                            for k, v in param_conf.get("value").items()
                            if k != "distribution"
                        },
                    )
                assert partials_are_similar(params.pdata[param_name]["dist"], expected)
            else:
                expected = random_distribution_sampler(
                    param_conf.get("value").get("distribution"),
                    **{
                        k: v
                        for k, v in param_conf.get("value").items()
                        if k != "distribution"
                    },
                )
                assert (
                    params.pdata[param_name]["dist"].__self__.kwds
                    == expected.__self__.kwds
                )
                assert (
                    params.pdata[param_name]["dist"].__self__.support()
                    == expected.__self__.support()
                )

        # The `pnames` attribute
        assert params.pnames == list(mock_inputs.config.keys())

        # The `pnames2pindex` attribute
        assert params.pnames2pindex == {
            key: idx for idx, key in enumerate(mock_inputs.config.keys())
        }

        # # The `stacked_modifier_method` attribute
        expected_stacked_modifier_method = {
            "sum": [],
            "product": [],
            "reduction_product": [],
        }
        for param_name, param_conf in mock_inputs.config.items():
            modifier_type = param_conf.get("stacked_modifier_method", "product")
            expected_stacked_modifier_method[modifier_type].append(param_name.lower())
        assert params.stacked_modifier_method == expected_stacked_modifier_method

    @pytest.mark.parametrize(
        "parameters_inputs,alpha_val", [(MockData.simple_inputs, None)]
    )
    def test_picklable_lamda_alpha(
        self, parameters_inputs: dict[str, Any], alpha_val: Any
    ) -> None:
        # Setup
        params = Parameters(**parameters_inputs)

        # Attribute error if `alpha_val` is not set
        with pytest.raises(AttributeError):
            params.picklable_lamda_alpha()

        # We get the expected value when `alpha_val` is set
        params.alpha_val = alpha_val
        assert params.picklable_lamda_alpha() == alpha_val

    @pytest.mark.parametrize(
        "parameters_inputs,sigma_val", [(MockData.simple_inputs, None)]
    )
    def test_picklable_lamda_sigma(
        self, parameters_inputs: dict[str, Any], sigma_val: Any
    ) -> None:
        # Setup
        params = Parameters(**parameters_inputs)

        # Attribute error if `sigma_val` is not set
        with pytest.raises(AttributeError):
            params.picklable_lamda_sigma()

        # We get the expected value when `sigma_val` is set
        params.sigma_val = sigma_val
        assert params.picklable_lamda_sigma() == sigma_val

    @pytest.mark.parametrize(
        "parameters_inputs", [(MockData.simple_inputs), (MockData.small_inputs)]
    )
    def test_get_pnames2pindex(self, parameters_inputs: dict[str, Any]) -> None:
        params = Parameters(**parameters_inputs)
        assert params.get_pnames2pindex() == params.pnames2pindex
        assert params.pnames2pindex == {
            key: idx
            for idx, key in enumerate(parameters_inputs["parameter_config"].keys())
        }

    @pytest.mark.parametrize(
        "factory,n_days,nsubpops",
        [
            (fixed_three_valid_parameter_factory, None, None),
            (fixed_three_valid_parameter_factory, 4, 2),
            (distribution_three_valid_parameter_factory, None, None),
            (distribution_three_valid_parameter_factory, 5, 2),
        ],
    )
    def test_parameters_quick_draw(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
        n_days: None | int,
        nsubpops: None | int,
    ) -> None:
        # Setup
        mock_inputs = factory(tmp_path)
        params = mock_inputs.create_parameters_instance()
        n_days = mock_inputs.number_of_days() if n_days is None else n_days
        nsubpops = mock_inputs.number_of_subpops() if nsubpops is None else nsubpops

        if mock_inputs.has_timeseries_parameter() and (
            (n_days_expected := mock_inputs.number_of_days()) != n_days
            or (nsubpops_expected := mock_inputs.number_of_subpops()) != nsubpops
        ):
            # Incompatible shapes
            with pytest.raises(
                ValueError,
                match=(
                    rf"could not broadcast input array from shape \({n_days}\,"
                    rf"{nsubpops}\) into shape \({n_days_expected}\,"
                    rf"{nsubpops_expected}\)"
                ),
            ):
                params.parameters_quick_draw(n_days, nsubpops)
        else:
            # Compatible shapes
            p_draw = params.parameters_quick_draw(n_days, nsubpops)
            assert isinstance(p_draw, np.ndarray)
            assert p_draw.dtype == np.float64
            assert p_draw.shape == (
                mock_inputs.number_of_parameters(),
                n_days,
                nsubpops,
            )

            # Loop over each param and check it individually
            for param_name, conf in mock_inputs.config.items():
                i = params.pnames.index(param_name)
                if "timeseries" in conf:
                    # Check if the values in p_draw[i, :, :] match timeseries
                    timeseries_df = mock_inputs.get_timeseries_df(param_name)
                    assert np.allclose(p_draw[i, :, :], timeseries_df.values)
                elif isinstance((fixed_value := conf.get("value")), float):
                    # Check if all the values in p_draw[i, :, :] match a const
                    assert np.allclose(p_draw[i, :, :], fixed_value)
                else:
                    # Check if the values in p_draw[i, :, :] match the distribution
                    assert np.allclose(p_draw[i, :, :], p_draw[i, 0, 0])
                    value = float(p_draw[i, 0, 0])
                    assert sample_fits_distribution(
                        value, **{k: v for k, v in conf.get("value").items()}
                    )

    @pytest.mark.parametrize(
        "factory,param_df,n_days,nsubpops",
        [
            (
                fixed_three_valid_parameter_factory,
                MockData.empty_param_overrides_df,
                None,
                None,
            ),
            (
                fixed_three_valid_parameter_factory,
                MockData.empty_param_overrides_df,
                4,
                2,
            ),
            (
                distribution_three_valid_parameter_factory,
                MockData.empty_param_overrides_df,
                None,
                None,
            ),
            (
                distribution_three_valid_parameter_factory,
                MockData.empty_param_overrides_df,
                5,
                2,
            ),
        ],
    )
    def test_parameters_load(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
        param_df: pd.DataFrame,
        n_days: None | int,
        nsubpops: None | int,
    ) -> None:
        # Setup
        mock_inputs = factory(tmp_path)
        params = mock_inputs.create_parameters_instance()
        n_days = mock_inputs.number_of_days() if n_days is None else n_days
        nsubpops = mock_inputs.number_of_subpops() if nsubpops is None else nsubpops

        timeseries_parameters = set(mock_inputs.get_timeseries_parameters())
        override_parameters = set(param_df["parameter"].unique())
        timeseries_not_overridden = timeseries_parameters - override_parameters

        if len(timeseries_not_overridden) and (
            (n_days_expected := mock_inputs.number_of_days()) != n_days
            or (nsubpops_expected := mock_inputs.number_of_subpops()) != nsubpops
        ):
            # Incompatible shapes
            with pytest.raises(
                ValueError,
                match=(
                    rf"could not broadcast input array from shape \({n_days}\,"
                    rf"{nsubpops}\) into shape \({n_days_expected}\,"
                    rf"{nsubpops_expected}\)"
                ),
            ):
                params.parameters_load(param_df, n_days, nsubpops)
        else:
            # Compatible shapes
            p_draw = params.parameters_load(param_df, n_days, nsubpops)
            assert isinstance(p_draw, np.ndarray)
            assert p_draw.dtype == np.float64
            assert p_draw.shape == (
                mock_inputs.number_of_parameters(),
                n_days,
                nsubpops,
            )

            # Loop over each param and check it individually
            for param_name, conf in mock_inputs.config.items():
                i = params.pnames.index(param_name)
                if param_name in param_df["parameter"].values:
                    # Check that the values in p_draw[i, :, :] match override
                    assert np.allclose(
                        p_draw[i, :, :],
                        param_df[param_df["parameter"] == param_name]
                        .get("value")
                        .item(),
                    )
                elif "timeseries" in conf:
                    # Check if the values in p_draw[i, :, :] match timeseries
                    timeseries_df = mock_inputs.get_timeseries_df(param_name)
                    assert np.allclose(p_draw[i, :, :], timeseries_df.values)
                elif isinstance((fixed_value := conf.get("value")), float):
                    # Check if all the values in p_draw[i, :, :] match a const
                    assert np.allclose(p_draw[i, :, :], fixed_value)
                else:
                    # Check if the values in p_draw[i, :, :] match the distribution
                    assert np.allclose(p_draw[i, :, :], p_draw[i, 0, 0])
                    value = float(p_draw[i, 0, 0])
                    assert sample_fits_distribution(
                        value, **{k: v for k, v in conf.get("value").items()}
                    )

    @pytest.mark.parametrize(
        "factory,n_days,nsubpops",
        [
            (fixed_three_valid_parameter_factory, None, None),
            (fixed_three_valid_parameter_factory, 4, 2),
            (distribution_three_valid_parameter_factory, None, None),
            (distribution_three_valid_parameter_factory, 5, 2),
        ],
    )
    def test_getParameterDF(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
        n_days: None | int,
        nsubpops: None | int,
    ) -> None:
        # Setup
        mock_inputs = factory(tmp_path)
        params = mock_inputs.create_parameters_instance()
        n_days = mock_inputs.number_of_days() if n_days is None else n_days
        nsubpops = mock_inputs.number_of_subpops() if nsubpops is None else nsubpops

        p_draw = params.parameters_quick_draw(n_days, nsubpops)
        df = params.getParameterDF(p_draw)

        # Go through assertions on the structure of the DataFrame
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (mock_inputs.number_of_nontimeseries_parameters(), 2)
        assert df.columns.to_list() == ["value", "parameter"]
        assert (df.index.to_series() == df["parameter"]).all()
        assert not df["parameter"].duplicated().any()
        assert set(df["parameter"].to_list()) == set(
            mock_inputs.get_nontimeseries_parameters()
        )
        for row in df.itertuples(index=False):
            i = params.pnames.index(row.parameter)
            assert np.isclose(row.value, p_draw[i, 0, 0])

    def test_parameters_reduce(self) -> None:
        # TODO: Come back and unit test this method after getting a better handle on
        # these NPI objects.
        pass
