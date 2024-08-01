from datetime import date
from functools import partial

import numpy as np
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile

from gempyor.parameters import Parameters
from gempyor.testing import create_confuse_subview_from_dict, partials_are_similar


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

    def test_timeseries_parameter_has_insufficient_dates_value_error(self) -> None:
        # First way to get at this error, purely a length difference
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
                    rf"ERROR loading file {temp_file.name} for parameter sigma\:\s+"
                    rf"the \'date\' entries of the provided file do not include all the"
                    rf" days specified to be modeled by\s+the config\. the provided "
                    rf"file includes 5 days between 2024-01-01( 00\:00\:00)? to "
                    rf"2024-01-05( 00\:00\:00)?,\s+while there are 6 days in the config"
                    rf" time span of 2024-01-01->2024-01-06\. The file must contain "
                    rf"entries for the\s+the exact start and end dates from the "
                    rf"config\. "
                ),
            ):
                Parameters(
                    invalid_timeseries_parameters,
                    ti=date(2024, 1, 1),
                    tf=date(2024, 1, 6),
                    subpop_names=["1", "2"],
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

    def test_parameters_instance_attributes(self) -> None:
        # Setup
        param_df = pd.DataFrame(
            data={
                "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
                "1": [1.2, 2.3, 3.4, 4.5, 5.6],
                "2": [2.3, 3.4, 4.5, 5.6, 6.7],
            }
        )
        with NamedTemporaryFile(suffix=".csv") as temp_file:
            param_df.to_csv(temp_file.name, index=False)
            valid_parameters = create_confuse_subview_from_dict(
                "parameters",
                {
                    "sigma": {"timeseries": temp_file.name},
                    "gamma": {"value": 0.1234, "stacked_modifier_method": "sum"},
                    "Ro": {
                        "value": {"distribution": "uniform", "low": 1.0, "high": 2.0}
                    },
                },
            )
            params = Parameters(
                valid_parameters,
                ti=date(2024, 1, 1),
                tf=date(2024, 1, 5),
                subpop_names=["1", "2"],
            )

            # The `npar` attribute
            assert params.npar == 3

            # The `pconfig` attribute
            assert params.pconfig == valid_parameters

            # The `pdata` attribute
            assert set(params.pdata.keys()) == {"sigma", "gamma", "Ro"}
            assert set(params.pdata["sigma"].keys()) == {
                "idx",
                "ts",
                "stacked_modifier_method",
            }
            assert params.pdata["sigma"]["idx"] == 0
            assert params.pdata["sigma"]["ts"].equals(param_df.set_index("date"))
            assert params.pdata["sigma"]["stacked_modifier_method"] == "product"
            assert set(params.pdata["gamma"].keys()) == {
                "idx",
                "dist",
                "stacked_modifier_method",
            }
            assert params.pdata["gamma"]["idx"] == 1
            assert isinstance(params.pdata["gamma"]["dist"], partial)
            assert partials_are_similar(
                params.pdata["gamma"]["dist"],
                partial(np.random.uniform, 0.1234, 0.1234),
            )
            assert params.pdata["gamma"]["stacked_modifier_method"] == "sum"
            assert set(params.pdata["Ro"].keys()) == {
                "idx",
                "dist",
                "stacked_modifier_method",
            }
            assert params.pdata["Ro"]["idx"] == 2
            assert isinstance(params.pdata["Ro"]["dist"], partial)
            assert partials_are_similar(
                params.pdata["Ro"]["dist"], partial(np.random.uniform, 1.0, 2.0)
            )
            assert params.pdata["Ro"]["stacked_modifier_method"] == "product"

            # The `pnames` attribute
            assert params.pnames == ["sigma", "gamma", "Ro"]

            # The `pnames2pindex` attribute
            assert params.pnames2pindex == {"sigma": 0, "gamma": 1, "Ro": 2}

            # The `stacked_modifier_method` attribute
            assert params.stacked_modifier_method == {
                "sum": ["gamma"],
                "product": ["sigma", "ro"],
                "reduction_product": [],
            }

    def test_picklable_lamda_alpha(self) -> None:
        # Setup
        simple_parameters = create_confuse_subview_from_dict(
            "parameters", {"sigma": {"value": 0.1}}
        )
        params = Parameters(
            simple_parameters,
            ti=date(2024, 1, 1),
            tf=date(2024, 1, 10),
            subpop_names=["1", "2"],
        )

        # Attribute error if `alpha_val` is not set
        with pytest.raises(AttributeError):
            params.picklable_lamda_alpha()

        # We get the expected value when `alpha_val` is set
        params.alpha_val = None
        assert params.picklable_lamda_alpha() == None

    def test_picklable_lamda_sigma(self) -> None:
        # Setup
        simple_parameters = create_confuse_subview_from_dict(
            "parameters", {"sigma": {"value": 0.1}}
        )
        params = Parameters(
            simple_parameters,
            ti=date(2024, 1, 1),
            tf=date(2024, 1, 10),
            subpop_names=["1", "2"],
        )

        # Attribute error if `sigma_val` is not set
        with pytest.raises(AttributeError):
            params.picklable_lamda_sigma()

        # We get the expected value when `sigma_val` is set
        params.sigma_val = None
        assert params.picklable_lamda_sigma() == None

    def test_get_pnames2pindex(self) -> None:
        simple_parameters = create_confuse_subview_from_dict(
            "parameters",
            {"sigma": {"value": 0.1}, "gamma": {"value": 0.2}, "eta": {"value": 0.3}},
        )
        params = Parameters(
            simple_parameters,
            ti=date(2024, 1, 1),
            tf=date(2024, 1, 10),
            subpop_names=["1", "2"],
        )
        assert params.get_pnames2pindex() == params.pnames2pindex
        assert params.pnames2pindex == {"sigma": 0, "gamma": 1, "eta": 2}

    def test_parameters_quick_draw(self) -> None:
        # First with a time series param, fixed size draws
        param_df = pd.DataFrame(
            data={
                "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
                "1": [1.2, 2.3, 3.4, 4.5, 5.6],
                "2": [2.3, 3.4, 4.5, 5.6, 6.7],
            }
        )
        with NamedTemporaryFile(suffix=".csv") as temp_file:
            param_df.to_csv(temp_file.name, index=False)
            valid_parameters = create_confuse_subview_from_dict(
                "parameters",
                {
                    "sigma": {"timeseries": temp_file.name},
                    "gamma": {"value": 0.1234, "stacked_modifier_method": "sum"},
                    "Ro": {
                        "value": {"distribution": "uniform", "low": 1.0, "high": 2.0}
                    },
                },
            )
            params = Parameters(
                valid_parameters,
                ti=date(2024, 1, 1),
                tf=date(2024, 1, 5),
                subpop_names=["1", "2"],
            )

            # Test the exception
            with pytest.raises(
                ValueError,
                match=(
                    r"could not broadcast input array from shape "
                    r"\(5\,2\) into shape \(4\,2\)"
                ),
            ):
                params.parameters_quick_draw(4, 2)

            # Test our result
            p_draw = params.parameters_quick_draw(5, 2)
            assert isinstance(p_draw, np.ndarray)
            assert p_draw.dtype == np.float64
            assert p_draw.shape == (3, 5, 2)
            assert np.allclose(
                p_draw[0, :, :],
                np.array([[1.2, 2.3], [2.3, 3.4], [3.4, 4.5], [4.5, 5.6], [5.6, 6.7]]),
            )
            assert np.allclose(p_draw[1, :, :], 0.1234 * np.ones((5, 2)))
            assert np.greater_equal(p_draw[2, :, :], 1.0).all()
            assert np.less(p_draw[2, :, :], 2.0).all()
            assert np.allclose(p_draw[2, :, :], p_draw[2, 0, 0])

        # Second without a time series param, arbitrary sized draws
        valid_parameters = create_confuse_subview_from_dict(
            "parameters",
            {
                "eta": {"value": 2.2},
                "nu": {
                    "value": {
                        "distribution": "truncnorm",
                        "mean": 0.0,
                        "sd": 2.0,
                        "a": -2.0,
                        "b": 2.0,
                    }
                },
            },
        )
        params = Parameters(
            valid_parameters,
            ti=date(2024, 1, 1),
            tf=date(2024, 1, 5),
            subpop_names=["1", "2"],
        )

        p_draw = params.parameters_quick_draw(5, 2)
        assert isinstance(p_draw, np.ndarray)
        assert p_draw.dtype == np.float64
        assert p_draw.shape == (2, 5, 2)
        assert np.allclose(p_draw[0, :, :], 2.2)
        assert np.greater_equal(p_draw[1, :, :], -2.0).all()
        assert np.less_equal(p_draw[1, :, :], 2.0).all()
        assert np.allclose(p_draw[1, :, :], p_draw[1, 0, 0])

        p_draw = params.parameters_quick_draw(4, 3)
        assert isinstance(p_draw, np.ndarray)
        assert p_draw.dtype == np.float64
        assert p_draw.shape == (2, 4, 3)
        assert np.allclose(p_draw[0, :, :], 2.2)
        assert np.greater_equal(p_draw[1, :, :], -2.0).all()
        assert np.less_equal(p_draw[1, :, :], 2.0).all()
        assert np.allclose(p_draw[1, :, :], p_draw[1, 0, 0])

    def test_parameters_load(self) -> None:
        # Setup
        param_overrides_df = pd.DataFrame(
            {"parameter": ["nu", "gamma", "nu"], "value": [0.1, 0.2, 0.3]}
        )
        param_empty_df = pd.DataFrame({"parameter": [], "value": []})

        # With time series
        param_df = pd.DataFrame(
            data={
                "date": pd.date_range(date(2024, 1, 1), date(2024, 1, 5)),
                "1": [1.2, 2.3, 3.4, 4.5, 5.6],
                "2": [2.3, 3.4, 4.5, 5.6, 6.7],
            }
        )
        with NamedTemporaryFile(suffix=".csv") as temp_file:
            param_df.to_csv(temp_file.name, index=False)
            valid_parameters = create_confuse_subview_from_dict(
                "parameters",
                {
                    "sigma": {"timeseries": temp_file.name},
                    "gamma": {"value": 0.1234, "stacked_modifier_method": "sum"},
                    "Ro": {
                        "value": {"distribution": "uniform", "low": 1.0, "high": 2.0}
                    },
                },
            )
            params = Parameters(
                valid_parameters,
                ti=date(2024, 1, 1),
                tf=date(2024, 1, 5),
                subpop_names=["1", "2"],
            )

            # Test the exception
            with pytest.raises(
                ValueError,
                match=(
                    r"could not broadcast input array from shape "
                    r"\(5\,2\) into shape \(4\,2\)"
                ),
            ):
                params.parameters_load(param_empty_df, 4, 2)

            # Empty overrides
            p_draw = params.parameters_load(param_empty_df, 5, 2)
            assert isinstance(p_draw, np.ndarray)
            assert p_draw.dtype == np.float64
            assert p_draw.shape == (3, 5, 2)
            assert np.allclose(
                p_draw[0, :, :],
                np.array([[1.2, 2.3], [2.3, 3.4], [3.4, 4.5], [4.5, 5.6], [5.6, 6.7]]),
            )
            assert np.allclose(p_draw[1, :, :], 0.1234 * np.ones((5, 2)))
            assert np.greater_equal(p_draw[2, :, :], 1.0).all()
            assert np.less(p_draw[2, :, :], 2.0).all()
            assert np.allclose(p_draw[2, :, :], p_draw[2, 0, 0])

            # But if we override time series no exception
            p_draw = params.parameters_load(
                pd.DataFrame({"parameter": ["sigma"], "value": [12.34]}), 4, 2
            )
            assert isinstance(p_draw, np.ndarray)
            assert p_draw.dtype == np.float64
            assert p_draw.shape == (3, 4, 2)
            assert np.allclose(p_draw[0, :, :], 12.34)
            assert np.allclose(p_draw[1, :, :], 0.1234 * np.ones((4, 2)))
            assert np.greater_equal(p_draw[2, :, :], 1.0).all()
            assert np.less(p_draw[2, :, :], 2.0).all()
            assert np.allclose(p_draw[2, :, :], p_draw[2, 0, 0])

            # If not overriding time series then must conform
            p_draw = params.parameters_load(param_overrides_df, 5, 2)
            assert isinstance(p_draw, np.ndarray)
            assert p_draw.dtype == np.float64
            assert p_draw.shape == (3, 5, 2)
            assert np.allclose(
                p_draw[0, :, :],
                np.array([[1.2, 2.3], [2.3, 3.4], [3.4, 4.5], [4.5, 5.6], [5.6, 6.7]]),
            )
            assert np.allclose(p_draw[1, :, :], 0.2 * np.ones((5, 2)))
            assert np.greater_equal(p_draw[2, :, :], 1.0).all()
            assert np.less(p_draw[2, :, :], 2.0).all()
            assert np.allclose(p_draw[2, :, :], p_draw[2, 0, 0])

        # Without time series
        valid_parameters = create_confuse_subview_from_dict(
            "parameters",
            {
                "eta": {"value": 2.2},
                "nu": {
                    "value": {
                        "distribution": "truncnorm",
                        "mean": 0.0,
                        "sd": 2.0,
                        "a": -2.0,
                        "b": 2.0,
                    }
                },
            },
        )
        params = Parameters(
            valid_parameters,
            ti=date(2024, 1, 1),
            tf=date(2024, 1, 5),
            subpop_names=["1", "2"],
        )

        # Takes an 'empty' DataFrame
        p_draw = params.parameters_load(param_empty_df, 5, 2)
        assert isinstance(p_draw, np.ndarray)
        assert p_draw.dtype == np.float64
        assert p_draw.shape == (2, 5, 2)
        assert np.allclose(p_draw[0, :, :], 2.2)
        assert np.greater_equal(p_draw[1, :, :], -2.0).all()
        assert np.less_equal(p_draw[1, :, :], 2.0).all()

        # Takes a DataFrame with values, only takes the first
        p_draw = params.parameters_load(param_overrides_df, 4, 3)
        assert isinstance(p_draw, np.ndarray)
        assert p_draw.dtype == np.float64
        assert p_draw.shape == (2, 4, 3)
        assert np.allclose(p_draw[0, :, :], 2.2)
        assert np.allclose(p_draw[1, :, :], 0.1)
