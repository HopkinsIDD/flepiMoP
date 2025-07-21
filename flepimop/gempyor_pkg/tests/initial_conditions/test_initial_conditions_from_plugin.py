"""Test the `initial_conditions_from_plugin` functionality"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from gempyor.initial_conditions import (
    DefaultInitialConditions,
    initial_conditions_from_plugin,
)
from gempyor.initial_conditions._plugins import _initial_conditions_plugins
from gempyor.testing import create_confuse_configview_from_dict
from gempyor.warnings import ConfigurationWarning


def test_no_method_specified_will_return_default_with_warning() -> None:
    """Not specifying method will return `DefaultInitialConditions` with warning"""
    with pytest.warns(
        ConfigurationWarning,
        match=(
            r"^Initial conditions plugin 'method' "
            r"was not specified, assuming 'Default'.$"
        ),
    ):
        initial_conditions = initial_conditions_from_plugin(
            create_confuse_configview_from_dict({}, "initial_conditions")
        )
    assert isinstance(initial_conditions, DefaultInitialConditions)


@pytest.mark.parametrize("method", ["nope", "not valid", "WillNotWork"])
def test_no_matching_plugin_found(method: str) -> None:
    """Specifying an invalid 'method' will raise a `ValueError`."""
    assert method not in _initial_conditions_plugins
    with pytest.raises(
        ValueError,
        match=(
            r"^There is no initial conditions plugin matching 'method' "
            rf"name of '{method}'. Instead the available options are: .*.$"
        ),
    ):
        initial_conditions_from_plugin(
            create_confuse_configview_from_dict({"method": method}, "initial_conditions")
        )


def test_external_module_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that an external module plugin can be loaded."""
    # Setup
    monkeypatch.chdir(tmp_path)
    plugin_path = tmp_path / "foobar.py"
    plugin_path.write_text(
        """
from typing import Literal

from gempyor.compartments import Compartments
from gempyor.initial_conditions import (
    InitialConditionsABC,
    register_initial_conditions_plugin,
)
from gempyor.parameters import Parameters
from gempyor.subpopulation_structure import SubpopulationStructure
import numpy as np
import numpy.typing as npt

class FoobarInitialConditions(InitialConditionsABC):
    method: Literal["Foobar"] = "Foobar"
    first_compartment: float
    
    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
    ) -> npt.NDArray[np.float64]:
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        y0[0, :] = self.first_compartment * subpopulation_structure.subpop_pop
        y0[1, :] = (1.0 - self.first_compartment) * subpopulation_structure.subpop_pop
        return y0

register_initial_conditions_plugin(FoobarInitialConditions)
"""
    )
    config = create_confuse_configview_from_dict(
        {
            "method": "Foobar",
            "module": str(plugin_path),
            "first_compartment": "0.9",
        },
        "initial_conditions",
    )
    # Test
    foobar_initial_conditions = initial_conditions_from_plugin(config)
    assert foobar_initial_conditions.method == "Foobar"
    assert foobar_initial_conditions.first_compartment == 0.9


def test_external_module_plugin_with_requested_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that an external module plugin can be loaded with requested parameters."""
    # Setup
    monkeypatch.chdir(tmp_path)
    plugin_path = tmp_path / "fizzbuzz.py"
    plugin_path.write_text(
        """
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from gempyor.compartments import Compartments
from gempyor.initial_conditions import (
    InitialConditionsABC,
    register_initial_conditions_plugin,
)
from gempyor.parameters import Parameters
from gempyor.subpopulation_structure import SubpopulationStructure


class FizzbuzzInitialConditions(InitialConditionsABC):
    method: Literal["Fizzbuzz"] = "Fizzbuzz"
    expit: bool
    
    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        w = expit(alpha).item() if self.expit else alpha
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        y0[0, :] = [w * pop for pop in subpopulation_structure.subpop_pop]
        y0[1, :] = [(1.0 - w) * pop for pop in subpopulation_structure.subpop_pop]
        return y0

register_initial_conditions_plugin(FizzbuzzInitialConditions)
"""
    )
    config = create_confuse_configview_from_dict(
        {
            "method": "Fizzbuzz",
            "module": str(plugin_path),
            "expit": True,
        },
        "initial_conditions",
    )

    # Test
    fizzbuzz_initial_conditions = initial_conditions_from_plugin(config)
    assert fizzbuzz_initial_conditions.method == "Fizzbuzz"
    assert fizzbuzz_initial_conditions.expit is True

    # Test with requested parameters
    compartments = Mock()
    compartments.compartments = pd.DataFrame.from_records(
        [
            {"infection_stage": compartment, "name": compartment}
            for compartment in ["S", "E", "I", "R"]
        ]
    )
    subpopulation_structure = Mock()
    subpopulation_structure.subpop_pop = [100, 200]
    subpopulation_structure.nsubpops = len(subpopulation_structure.subpop_pop)
    parameters = Mock()
    parameters.pdata = {
        "humidity": {"idx": 0, "ts": True},
        "alpha": {"idx": 1, "dist": True},
        "beta": {"idx": 2, "dist": True},
    }
    p_draw = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # humidity
            [[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]],  # alpha
            [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]],  # beta
        ],
        dtype=np.float64,
    )
    y0 = fizzbuzz_initial_conditions.get_initial_conditions(
        0, compartments, subpopulation_structure, parameters, p_draw
    )
    assert y0.shape == (4, 2)
    assert np.all(y0[0, :] > y0[1, :])
    assert np.allclose(y0[2:, :], 0.0)
