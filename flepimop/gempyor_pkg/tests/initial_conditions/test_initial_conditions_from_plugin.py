"""Test the `initial_conditions_from_plugin` functionality"""

from pathlib import Path

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
        parameters: Parameters,
        p_draw: npt.NDArray[np.float64],
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
