"""
Unit tests for `gempyor.initial_conditions.read_initial_condition_from_tidydataframe`.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gempyor.initial_conditions._file_or_folder_draw import (
    _read_initial_condition_from_tidydataframe,
)


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
        model_info.compartments = Mock()
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


def test_setting_allow_missing_subpops_to_true_is_not_supported() -> None:
    """The function raises an error if `allow_missing_subpops` is set to True."""
    ic_df = pd.DataFrame(
        data={
            "mc_name": ["S", "E", "I", "R"],
            "subpop": ["A", "A", "A", "A"],
            "amount": [90, 10, 0, 0],
        }
    )
    compartments = pd.DataFrame(data={"infection_stage": ["S", "E", "I", "R"]})
    subpop_names = ["A", "B"]
    subpop_pop = [100, 200]
    model_info = create_mock_model_info(
        compartments=compartments,
        subpop_names=subpop_names,
        subpop_pop=subpop_pop,
    )
    with pytest.raises(
        RuntimeError,
        match=r"^There is a bug; report this message. Past implementation was buggy.$",
    ):
        _read_initial_condition_from_tidydataframe(
            ic_df,
            model_info.compartments.compartments,
            model_info.subpop_struct,
            True,
            False,
            False,
        )


@pytest.mark.parametrize(
    ("ic_df", "compartments", "subpop_names", "subpop_pop"),
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
    ic_df: pd.DataFrame,
    compartments: pd.DataFrame,
    subpop_names: list[str],
    subpop_pop: list[int],
) -> None:
    """The function raises an error if `allow_missing_subpops` is set to False."""
    ic_unique_subpops = ic_df["subpop"].unique()
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
            r"^Subpop '.*' does not exist in `initial_conditions::states_file`. "
            r"You can set `allow_missing_subpops=TRUE` to bypass this error.$"
        ),
    ):
        _read_initial_condition_from_tidydataframe(
            ic_df,
            model_info.compartments.compartments,
            model_info.subpop_struct,
            False,
            False,
            False,
        )


@pytest.mark.parametrize(
    (
        "ic_df",
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
    ic_df: pd.DataFrame,
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
        ic_df,
        model_info.compartments.compartments,
        model_info.subpop_struct,
        False,
        allow_missing_compartments,
        proportional_ic,
    )
    assert y0.shape == (len(compartments), len(subpop_names))
    assert np.allclose(y0.sum(axis=0), subpop_pop)
