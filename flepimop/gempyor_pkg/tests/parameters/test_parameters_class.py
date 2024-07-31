from datetime import date
from functools import partial

import numpy as np
import pandas as pd
import pytest
from tempfile import NamedTemporaryFile

from gempyor.parameters import Parameters
from gempyor.testing import create_confuse_subview_from_dict


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
            assert params.npar == 3
            assert params.pconfig == valid_parameters
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
            assert params.pdata["gamma"]["dist"].func == np.random.uniform
            assert params.pdata["gamma"]["dist"].args == (0.1234, 0.1234)
            assert params.pdata["gamma"]["stacked_modifier_method"] == "sum"
            assert set(params.pdata["Ro"].keys()) == {
                "idx",
                "dist",
                "stacked_modifier_method",
            }
            assert params.pdata["Ro"]["idx"] == 2
            assert isinstance(params.pdata["Ro"]["dist"], partial)
            assert params.pdata["Ro"]["dist"].func == np.random.uniform
            assert params.pdata["Ro"]["dist"].args == (1.0, 2.0)
            assert params.pdata["Ro"]["stacked_modifier_method"] == "product"
            assert params.pnames == ["sigma", "gamma", "Ro"]
            assert params.pnames2pindex == {"sigma": 0, "gamma": 1, "Ro": 2}
            assert params.stacked_modifier_method == {
                "sum": ["gamma"],
                "product": ["sigma", "ro"],
                "reduction_product": [],
            }
