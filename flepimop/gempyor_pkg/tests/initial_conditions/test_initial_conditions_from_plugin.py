"""Test the `initial_conditions_from_plugin` functionality"""

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
