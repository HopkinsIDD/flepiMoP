from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import partial
from itertools import repeat
import multiprocessing as mp
import pathlib
from typing import Any, Callable
from uuid import uuid4

import confuse
import numpy as np
import pandas as pd
import pytest

from gempyor.parameters import Parameters
from gempyor.distributions import (
    DistributionABC,
    BetaDistribution,
    BinomialDistribution,
    FixedDistribution,
    GammaDistribution,
    LognormalDistribution,
    NormalDistribution,
    PoissonDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
    WeibullDistribution,
)
from gempyor.testing import (
    create_confuse_configview_from_dict,
    partials_are_similar,
    sample_fits_distribution,
)
from gempyor.utils import random_distribution_sampler


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
            self._confuse_subview = create_confuse_configview_from_dict(
                self.config, name="parameters"
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
        return (self.tf - self.ti).days + 1

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

    def get_timeseries_df(
        self, param_name: str, subset_by_subpops: bool = True
    ) -> pd.DataFrame:
        df = self._timeseries_dfs.get(param_name)
        if df is not None:
            return df[self.subpop_names].copy() if subset_by_subpops else df.copy()
        conf = self.config.get(param_name, {})
        df_file = conf.get("timeseries")
        if df_file is None:
            raise ValueError(
                f"The given param '{param_name}' does not have a timeseries dataframe."
            )
        df = pd.read_csv(df_file, index_col=None)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        self._timeseries_dfs[param_name] = df
        return df[self.subpop_names].copy() if subset_by_subpops else df.copy()


def fixed_three_valid_parameter_factory(tmp_path: pathlib.Path) -> MockParametersInput:
    return MockParametersInput(
        config={
            "sigma": {"value": {"distribution": "fixed", "value": 0.1}},
            "eta": {"value": {"distribution": "fixed", "value": 0.2}},
            "nu": {"value": {"distribution": "fixed", "value": 0.3}},
        },
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
    df = pd.DataFrame(
        data={
            "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
            "1": [1.2, 2.3, 3.4, 4.5, 5.6],
            "2": [2.3, 3.4, 4.5, 5.6, 6.7],
        }
    )
    df.to_csv(tmp_file, index=False)
    return MockParametersInput(
        config={
            "sigma": {"timeseries": str(tmp_file.absolute())},
            "gamma": {
                "value": {"distribution": "fixed", "value": 0.1234},
                "stacked_modifier_method": "sum",
            },
            "Ro": {"value": {"distribution": "uniform", "low": 1.0, "high": 2.0}},
        },
        ti=df["date"].dt.date.min(),
        tf=df["date"].dt.date.max(),
        subpop_names=[c for c in df.columns.to_list() if c != "date"],
    )


def nonunique_invalid_parameter_factory(tmp_path: pathlib.Path) -> MockParametersInput:
    return MockParametersInput(
        config={
            "sigma": {"value": 0.1},
            "eta": {"value": 0.2},
            "SIGMA": {"value": 0.3},
        },
        ti=date(2024, 1, 1),
        tf=date(2024, 1, 3),
        subpop_names=["1", "2", "3"],
    )


def insufficient_columns_parameter_factory(
    tmp_path: pathlib.Path,
) -> MockParametersInput:
    df = pd.DataFrame(
        data={
            "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
            "1": [1.2, 2.3, 3.4, 4.5, 5.6],
            "2": [2.3, 3.4, 4.5, 5.6, 6.7],
        }
    )
    tmp_file = tmp_path / f"{uuid4().hex}.csv"
    df.to_csv(tmp_file, index=False)
    return MockParametersInput(
        config={"sigma": {"timeseries": str(tmp_file.absolute())}},
        ti=date(2024, 1, 1),
        tf=date(2024, 1, 5),
        subpop_names=["1", "2", "3"],
    )


def insufficient_dates_parameter_factory(tmp_path: pathlib.Path) -> MockParametersInput:
    df = pd.DataFrame(
        data={
            "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
            "1": [1.2, 2.3, 3.4, 4.5, 5.6],
            "2": [2.3, 3.4, 4.5, 5.6, 6.7],
        }
    )
    tmp_file = tmp_path / f"{uuid4().hex}.csv"
    df.to_csv(tmp_file, index=False)
    return MockParametersInput(
        config={"sigma": {"timeseries": str(tmp_file.absolute())}},
        ti=date(2024, 1, 1),
        tf=date(2024, 1, 6),
        subpop_names=["1", "2"],
    )


def sample_params(params: Parameters, reinit: bool) -> np.ndarray:
    """
    Helper method for unit testing.

    Args:
        params: The instance of the Parameters class to sample from.
        reinit: Whether to reinitialize the parameters.

    Returns:
        The sampled parameters as a flattened numpy array.
    """
    if reinit:
        params.reinitialize_distributions()
    return params.parameters_quick_draw(1, 1).flatten()


class TestParameters:
    @pytest.mark.parametrize("factory", [nonunique_invalid_parameter_factory])
    def test_nonunique_parameter_names_value_error(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
    ) -> None:
        mock_inputs = factory(tmp_path)
        with pytest.raises(
            ValueError,
            match=(
                r"Parameters of the SEIR model have the same name "
                r"\(remember that case is not sufficient\!\)"
            ),
        ):
            mock_inputs.create_parameters_instance()

    @pytest.mark.parametrize("factory", [insufficient_columns_parameter_factory])
    def test_timeseries_parameter_has_insufficient_columns_value_error(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
    ) -> None:
        mock_inputs = factory(tmp_path)
        tmp_file = None
        for param_name, conf in mock_inputs.config.items():
            if "timeseries" in conf:
                df = mock_inputs.get_timeseries_df(param_name, subset_by_subpops=False)
                actual_columns = len(df.columns)
                if (
                    actual_columns != mock_inputs.number_of_subpops()
                    and actual_columns != 1
                ):
                    tmp_file = conf.get("timeseries")
                    break
        if tmp_file is None:
            raise RuntimeError(
                (
                    "The given factory does not produce a timeseries "
                    "with an insufficient number of columns."
                )
            )
        with pytest.raises(
            ValueError,
            match=(
                rf"^Issue loading file '{tmp_file}' for parameter 'sigma': "
                rf"the number of non-'date' columns is '{actual_columns}', expected "
                rf"'{mock_inputs.number_of_subpops()}' "
                rf"\(number of subpopulations\) or one\.$"
            ),
        ):
            mock_inputs.create_parameters_instance()

    @pytest.mark.parametrize("factory", [insufficient_dates_parameter_factory])
    def test_timeseries_parameter_has_insufficient_dates_value_error(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
    ) -> None:
        mock_inputs = factory(tmp_path)

        tmp_file = None
        for param_name, conf in mock_inputs.config.items():
            if "timeseries" in conf:
                df = mock_inputs.get_timeseries_df(param_name)
                timeseries_start_date = df.index.to_series().dt.date.min()
                timeseries_end_date = df.index.to_series().dt.date.max()
                if (
                    (timeseries_start_date > mock_inputs.ti)
                    or (timeseries_end_date < mock_inputs.tf)
                    or (
                        not pd.date_range(mock_inputs.ti, mock_inputs.tf)
                        .isin(df.index)
                        .all()
                    )
                ):
                    tmp_file = conf.get("timeseries")
                    break

        if tmp_file is None:
            raise RuntimeError(
                (
                    "The given factory does not produce a timeseries with an "
                    "insufficient date range."
                )
            )

        file_days = (timeseries_end_date - timeseries_start_date).days + 1
        with pytest.raises(
            ValueError,
            match=(
                f"Issue loading file '{tmp_file}' for parameter 'sigma': "
                f"Provided file dates span '{timeseries_start_date}( 00:00:00)?' to "
                rf"'{timeseries_end_date}( 00:00:00)?', "
                f"but the config dates span '{mock_inputs.ti}' to '{mock_inputs.tf}'.$"
            ),
        ):
            mock_inputs.create_parameters_instance()

    @pytest.mark.parametrize(
        "factory",
        [
            (fixed_three_valid_parameter_factory),
            (distribution_three_valid_parameter_factory),
            (valid_parameters_factory),
        ],
    )
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
            assert params.pdata[param_name]["stacked_modifier_method"] == param_conf.get(
                "stacked_modifier_method", "product"
            )
            if "timeseries" in param_conf:
                assert params.pdata[param_name]["ts"].equals(
                    mock_inputs.get_timeseries_df(param_name)
                )
            elif "dist" in params.pdata[param_name]:
                dist_obj = params.pdata[param_name]["dist"]
                value_conf = param_conf.get("value")
                assert isinstance(dist_obj, DistributionABC)

        # The `pnames` attribute
        assert set(params.pnames) == set(mock_inputs.config.keys())

        # The `pnames2pindex` attribute
        assert params.pnames2pindex == {
            p: params.pnames.index(p) for p in params.pnames2pindex
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
        "factory",
        [
            (fixed_three_valid_parameter_factory),
            (distribution_three_valid_parameter_factory),
            (valid_parameters_factory),
        ],
    )
    def test_get_pnames2pindex(
        self,
        tmp_path: pathlib.Path,
        factory: Callable[[pathlib.Path], MockParametersInput],
    ) -> None:
        # Setup
        mock_inputs = factory(tmp_path)
        params = mock_inputs.create_parameters_instance()

        # Assertions
        assert params.get_pnames2pindex() == params.pnames2pindex
        assert params.pnames2pindex == {p: params.pnames.index(p) for p in params.pnames}

    @pytest.mark.parametrize(
        "factory,n_days,nsubpops",
        [
            (fixed_three_valid_parameter_factory, None, None),
            (fixed_three_valid_parameter_factory, 4, 2),
            (distribution_three_valid_parameter_factory, None, None),
            (distribution_three_valid_parameter_factory, 5, 2),
            (valid_parameters_factory, None, None),
            (valid_parameters_factory, 13, 3),
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
        n_days_expected = mock_inputs.number_of_days()
        nsubpops_expected = mock_inputs.number_of_subpops()
        n_days = mock_inputs.number_of_days() if n_days is None else n_days
        nsubpops = mock_inputs.number_of_subpops() if nsubpops is None else nsubpops

        if mock_inputs.has_timeseries_parameter() and (
            n_days_expected != n_days or nsubpops_expected != nsubpops
        ):
            # Incompatible shapes
            with pytest.raises(
                ValueError,
                match=(
                    rf"^could not broadcast input array from shape \({n_days_expected}"
                    rf"\,{nsubpops_expected}\) into shape \({n_days}\,{nsubpops}\)$"
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
                    timeseries_df = mock_inputs.get_timeseries_df(param_name)
                    assert np.allclose(p_draw[i, :, :], timeseries_df.values)

                else:
                    dist_obj = params.pdata[param_name]["dist"]
                    drawn_values = p_draw[i, :, :]

    @pytest.mark.parametrize(
        "factory,param_df,n_days,nsubpops",
        [
            (
                fixed_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": [], "value": []}),
                None,
                None,
            ),
            (
                fixed_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": [], "value": []}),
                4,
                2,
            ),
            (
                fixed_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": ["sigma"], "value": [-0.123]}),
                None,
                None,
            ),
            (
                fixed_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": ["sigma"], "value": [-0.123]}),
                4,
                2,
            ),
            (
                distribution_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": [], "value": []}),
                None,
                None,
            ),
            (
                distribution_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": [], "value": []}),
                5,
                2,
            ),
            (
                distribution_three_valid_parameter_factory,
                pd.DataFrame(data={"parameter": ["nu", "alpha"], "value": [-9.9, 0.0]}),
                None,
                None,
            ),
            (
                valid_parameters_factory,
                pd.DataFrame(data={"parameter": [], "value": []}),
                None,
                None,
            ),
            (
                valid_parameters_factory,
                pd.DataFrame(data={"parameter": [], "value": []}),
                13,
                2,
            ),
            (
                valid_parameters_factory,
                pd.DataFrame(data={"parameter": ["Ro", "Ro"], "value": [2.5, 3.6]}),
                None,
                None,
            ),
            (
                valid_parameters_factory,
                pd.DataFrame(data={"parameter": ["Ro", "Ro"], "value": [2.5, 3.6]}),
                13,
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
        n_days_expected = mock_inputs.number_of_days()
        nsubpops_expected = mock_inputs.number_of_subpops()
        n_days = n_days_expected if n_days is None else n_days
        nsubpops = nsubpops_expected if nsubpops is None else nsubpops

        timeseries_parameters = set(mock_inputs.get_timeseries_parameters())
        override_parameters = set(param_df["parameter"].unique())
        timeseries_not_overridden = timeseries_parameters - override_parameters

        if len(timeseries_not_overridden) and (
            n_days_expected != n_days or nsubpops_expected != nsubpops
        ):
            # Incompatible shapes
            with pytest.raises(
                ValueError,
                match=(
                    rf"^could not broadcast input array from shape \({n_days_expected}"
                    rf"\,{nsubpops_expected}\) into shape \({n_days}\,{nsubpops}\)$"
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
                    assert np.allclose(
                        p_draw[i, :, :],
                        param_df[param_df["parameter"] == param_name]
                        .iloc[0]["value"]
                        .item(),
                    )

                elif "timeseries" in conf:
                    timeseries_df = mock_inputs.get_timeseries_df(param_name)
                    assert np.allclose(p_draw[i, :, :], timeseries_df.values)

                else:
                    dist_obj = params.pdata[param_name]["dist"]
                    drawn_values = p_draw[i, :, :]

                    # For a fixed distribution, all values must be equal to the specified value
                    if isinstance(dist_obj, FixedDistribution):
                        assert np.allclose(drawn_values, dist_obj.value)

                    # For bounded distributions, check that all values are within the domain
                    elif isinstance(dist_obj, UniformDistribution):
                        assert np.all(drawn_values >= dist_obj.low)
                        assert np.all(drawn_values <= dist_obj.high)
                    elif isinstance(dist_obj, TruncatedNormalDistribution):
                        assert np.all(drawn_values >= dist_obj.a)
                        assert np.all(drawn_values <= dist_obj.b)
                    elif isinstance(dist_obj, BetaDistribution):
                        assert np.all(drawn_values >= 0)
                        assert np.all(drawn_values <= 1)

                    # For discrete distributions, check for integer type and domain
                    elif isinstance(dist_obj, BinomialDistribution):
                        assert np.all(drawn_values >= 0)
                        assert np.all(drawn_values <= dist_obj.n)
                    elif isinstance(dist_obj, PoissonDistribution):
                        assert np.all(drawn_values >= 0)

                    # For distributions with a non-negative domain
                    elif isinstance(
                        dist_obj,
                        (LognormalDistribution, GammaDistribution, WeibullDistribution),
                    ):
                        assert np.all(drawn_values >= 0)

                    # For distributions without a simple domain to check (Normal)
                    else:
                        assert isinstance(dist_obj, NormalDistribution)
                        assert not np.isnan(drawn_values).any()

    @pytest.mark.parametrize(
        "factory,n_days,nsubpops",
        [
            (fixed_three_valid_parameter_factory, None, None),
            (fixed_three_valid_parameter_factory, 4, 2),
            (distribution_three_valid_parameter_factory, None, None),
            (distribution_three_valid_parameter_factory, 5, 2),
            (valid_parameters_factory, None, None),
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

    @pytest.mark.parametrize("do_reinit", [True, False])
    def test_reinitialize_parameters(self, tmp_path: pathlib.Path, do_reinit: bool) -> None:
        """Reinitialization of distributions required for multiprocessing under spawn."""
        mock_inputs = distribution_three_valid_parameter_factory(tmp_path)
        params = mock_inputs.create_parameters_instance()
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as ex:
            results = list(
                ex.map(
                    sample_params,
                    repeat(params, times=6),
                    repeat(do_reinit, times=6),
                )
            )
        for i in range(1, len(results)):
            assert np.allclose(results[i - 1], results[i]) == (not do_reinit)
