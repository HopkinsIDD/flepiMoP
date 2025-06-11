"""Provides functionality for handling initial conditions."""

__all__ = (
    "InitialConditions",
    "check_population",
    "initial_conditions_factory",
    "read_initial_condition_from_seir_output",
    "read_initial_condition_from_tidydataframe",
)


from ._factory import initial_conditions_factory
from ._initial_conditions import InitialConditions
from ._utils import check_population
from ._readers import (
    read_initial_condition_from_seir_output,
    read_initial_condition_from_tidydataframe,
)
