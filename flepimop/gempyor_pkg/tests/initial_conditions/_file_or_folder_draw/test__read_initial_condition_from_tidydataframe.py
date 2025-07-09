"""
Unit tests for `gempyor.initial_conditions.read_initial_condition_from_tidydataframe`.
"""

from typing import Generator, NamedTuple
from unittest.mock import Mock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from gempyor.initial_conditions._file_or_folder_draw import (
    _read_initial_condition_from_tidydataframe,
)


class CompartmentsMock(Mock):
    def subset_dataframe(
        self,
        dataframe: pd.DataFrame,
        skip_name: bool = False,
        raise_on_empty: bool = True,
        yield_compartment: bool = False,
    ) -> Generator[pd.DataFrame | tuple[NamedTuple, pd.DataFrame], None, None]:
        mc_name = (not skip_name) and ("mc_name" in dataframe.columns)
        row_names = ["Index"] + self.compartments.columns.tolist()
        for row in self.compartments.itertuples():
            query = (
                f"mc_name=='{row.name}'"
                if mc_name
                else " & ".join(
                    f"mc_{k}=='{v}'"
                    for k, v in zip(row_names, row)
                    if k not in {"Index", "name"}
                )
            )
            dataframe_subset = dataframe.query(query)
            if raise_on_empty and dataframe_subset.empty:
                raise ValueError(
                    "There were no matches found in `dataframe` for "
                    f"compartment filters matching the query: {query}"
                )
            yield (row, dataframe_subset) if yield_compartment else dataframe_subset


def create_mock_model_info(
    compartments: pd.DataFrame | None = None,
    subpop_names: list[str] | None = None,
    subpop_pop: list[int] | None = None,
) -> Mock:
    """
    Create a mock model info object for unit testing purposes.

    Args:
        compartments: A DataFrame representing compartments of the model, or `None` if
            not applicable.
        subpop_names: A list of subpopulation names, or `None` if not applicable.
        subpop_pop: A list of subpopulation populations, or `None` if not applicable.

    Returns:
        A mock model info object that can be used in unit testing.
    """
    model_info = Mock()
    if compartments is not None:
        model_info.compartments = CompartmentsMock()
        model_info.compartments.compartments = compartments.copy()
        model_info.compartments.compartments["name"] = (
            model_info.compartments.compartments.astype(str).apply(
                lambda x: "_".join(x), axis=1
            )
        )
    if any(v is not None for v in (subpop_names, subpop_pop)):
        model_info.subpop_struct = Mock()
        if subpop_names is not None:
            model_info.subpop_struct.subpop_names = subpop_names
            model_info.subpop_struct.nsubpops = len(subpop_names)
            model_info.nsubpops = len(subpop_names)
        if subpop_pop is not None:
            model_info.subpop_struct.subpop_pop = subpop_pop
            model_info.subpop_struct.nsubpops = len(subpop_pop)
            model_info.nsubpops = len(subpop_pop)
            model_info.subpop_pop = subpop_pop
    return model_info


@pytest.mark.parametrize(
    ("set_initial_conditions", "compartments", "subpop_names", "subpop_pop"),
    [
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "E", "I", "R"],
                    "subpop": ["A", "A", "A", "A"],
                    "amount": [90, 10, 0, 0],
                }
            ),
            pd.DataFrame(data={"infection_stage": ["S", "E", "I", "R"]}),
            ["A", "B"],
            [100, 200],
        ),
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "E", "I", "R"],
                    "subpop": ["B", "B", "B", "B"],
                    "amount": [190, 10, 0, 0],
                }
            ),
            pd.DataFrame(data={"infection_stage": ["S", "E", "I", "R"]}),
            ["A", "B"],
            [100, 200],
        ),
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "E", "I", "R", "S", "E", "I", "R"],
                    "subpop": ["A", "A", "A", "A", "B", "B", "B", "B"],
                    "amount": [90, 10, 0, 0, 190, 10, 0, 0],
                }
            ),
            pd.DataFrame(data={"infection_stage": ["S", "E", "I", "R"]}),
            ["A", "B", "C", "D"],
            [100, 200, 300, 400],
        ),
    ],
)
def test_missing_subpops_value_error(
    set_initial_conditions: pd.DataFrame,
    compartments: pd.DataFrame,
    subpop_names: list[str],
    subpop_pop: list[int],
) -> None:
    """The function raises an error if `allow_missing_subpops` is set to False."""
    ic_unique_subpops = set_initial_conditions["subpop"].unique()
    missing_subpop = next((s for s in subpop_names if s not in ic_unique_subpops), None)
    if missing_subpop is None:
        pytest.fail(
            "Test setup is incorrect, all subpopulations "
            "are present in the initial conditions."
        )
    model_info = create_mock_model_info(
        compartments=compartments,
        subpop_names=subpop_names,
        subpop_pop=subpop_pop,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^The following subpopulations are missing from the initial conditions "
            r"dataframe: .*$"
        ),
    ):
        _read_initial_condition_from_tidydataframe(
            set_initial_conditions,
            model_info.compartments,
            model_info.subpop_struct,
            False,
            False,
            False,
        )


@pytest.mark.parametrize(
    (
        "set_initial_conditions",
        "compartments",
        "subpop_names",
        "subpop_pop",
        "allow_missing_compartments",
        "proportional_ic",
    ),
    [
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "E", "I", "R", "S", "E", "I", "R"],
                    "subpop": ["A", "A", "A", "A", "B", "B", "B", "B"],
                    "amount": [90, 10, 0, 0, 190, 10, 0, 0],
                }
            ),
            pd.DataFrame(data={"infection_stage": ["S", "E", "I", "R"]}),
            ["A", "B"],
            [100, 200],
            False,
            False,
        ),
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "E", "S", "E"],
                    "subpop": ["A", "A", "B", "B"],
                    "amount": [90, 10, 190, 10],
                }
            ),
            pd.DataFrame(data={"infection_stage": ["S", "E", "I", "R"]}),
            ["A", "B"],
            [100, 200],
            True,
            False,
        ),
    ],
)
def test_exact_results_for_select_inputs(
    set_initial_conditions: pd.DataFrame,
    compartments: pd.DataFrame,
    subpop_names: list[str],
    subpop_pop: list[int],
    allow_missing_compartments: bool,
    proportional_ic: bool,
) -> None:
    """The function produces valid starting initial conditions given valid inputs."""
    model_info = create_mock_model_info(
        compartments=compartments,
        subpop_names=subpop_names,
        subpop_pop=subpop_pop,
    )
    y0 = _read_initial_condition_from_tidydataframe(
        set_initial_conditions,
        model_info.compartments,
        model_info.subpop_struct,
        False,
        allow_missing_compartments,
        proportional_ic,
    )
    assert y0.shape == (len(compartments), len(subpop_names))
    assert np.allclose(y0.sum(axis=0), subpop_pop)


@pytest.mark.parametrize(
    (
        "set_initial_conditions",
        "compartments",
        "subpop_names",
        "subpop_pop",
        "proportional_ic",
        "y0_expected",
    ),
    [
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "S", "I", "I", "R", "R"],
                    "subpop": ["A", "B", "A", "B", "A", "B"],
                    "amount": [90, 90, "rest", "rest", "rest", 0],
                },
            ),
            pd.DataFrame(data={"infection_stage": ["S", "I", "R"]}),
            ["A", "B"],
            [100, 100],
            False,
            np.array([[90, 90], [5, 10], [5, 0]], dtype=np.float64),
        ),
        (
            pd.DataFrame(
                data={
                    "mc_name": ["S", "S", "I", "I", "R", "R"],
                    "subpop": ["A", "B", "A", "B", "A", "B"],
                    "amount": [0.6, 0.9, "rest", "rest", "rest", 0],
                },
            ),
            pd.DataFrame(data={"infection_stage": ["S", "I", "R"]}),
            ["A", "B"],
            [100, 300],
            True,
            np.array([[60, 270], [20, 30], [20, 0]], dtype=np.float64),
        ),
    ],
)
def test_initial_conditions_with_rest_allocations(
    set_initial_conditions: pd.DataFrame,
    compartments: pd.DataFrame,
    subpop_names: list[str],
    subpop_pop: list[int],
    proportional_ic: bool,
    y0_expected: npt.NDArray[np.float64],
) -> None:
    """Test initial conditions with 'rest' allocations."""
    model_info = create_mock_model_info(
        compartments=compartments,
        subpop_names=subpop_names,
        subpop_pop=subpop_pop,
    )
    y0 = _read_initial_condition_from_tidydataframe(
        set_initial_conditions,
        model_info.compartments,
        model_info.subpop_struct,
        True,
        True,
        proportional_ic,
    )
    assert y0.shape == (len(compartments), len(subpop_names))
    assert np.allclose(y0.sum(axis=0), subpop_pop)
    assert np.allclose(y0, y0_expected)
