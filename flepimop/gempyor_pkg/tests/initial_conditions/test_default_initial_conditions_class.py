"""Unit tests for the `gempyor.initial_conditions.DefaultInitialConditions` class."""

from datetime import date
from string import ascii_lowercase
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gempyor.initial_conditions import DefaultInitialConditions
from gempyor.parameters import Parameters
from gempyor.testing import create_confuse_config_from_dict


@pytest.mark.parametrize("sim_id", [i for i in range(1, 102, 20)])
@pytest.mark.parametrize(
    "compartments_dataframe",
    [
        pd.DataFrame(data={"infection_stage": ["S", "I", "R"]}),
        pd.DataFrame(
            data={"infection_stage": ["S", "E1", "E2", "I1", "I2", "R", "H", "D"]}
        ),
        pd.DataFrame(
            data={
                "infection_stage": [
                    "S",
                    "E",
                    "I",
                    "R",
                    "S",
                    "E",
                    "I",
                    "R",
                    "S",
                    "E",
                    "I",
                    "R",
                ],
                "vaccines": [
                    "no_dose",
                    "no_dose",
                    "no_dose",
                    "no_dose",
                    "one_dose",
                    "one_dose",
                    "one_dose",
                    "one_dose",
                    "two_dose",
                    "two_dose",
                    "two_dose",
                    "two_dose",
                ],
            }
        ),
    ],
)
@pytest.mark.parametrize("subpop_pop", [[123], [12, 23, 34, 45], [987, 654, 321]])
def test_default_create_initial_conditions(
    sim_id: int, compartments_dataframe: pd.DataFrame, subpop_pop: list[int]
) -> None:
    # Setup work
    compartments = Mock()
    compartments.compartments = compartments_dataframe.copy()
    compartments.compartments["name"] = compartments.compartments.astype(str).apply(
        lambda x: "_".join(x), axis=1
    )

    subpopulation_structure = Mock()
    subpopulation_structure.nsubpops = len(subpop_pop)
    subpopulation_structure.subpop_pop = subpop_pop

    # Do the test
    initial_conditions = DefaultInitialConditions(path_prefix=None)
    y0 = initial_conditions.create_initial_conditions(
        sim_id, compartments, subpopulation_structure
    )

    # Assertions
    assert y0.shape == (len(compartments_dataframe), len(subpop_pop))
    assert y0.dtype == np.float64
    assert np.allclose(y0[0, :].T, np.array(subpop_pop))
    assert np.allclose(y0[1:, :], 0.0)
