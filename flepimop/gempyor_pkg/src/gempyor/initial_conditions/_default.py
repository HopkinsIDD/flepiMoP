"""Default initial conditions implementation."""

__all__: tuple[str, ...] = ()


from typing import Literal

import numpy as np
import numpy.typing as npt

from ..compartments import Compartments
from ..subpopulation_structure import SubpopulationStructure
from ._base import InitialConditionsABC
from ._plugins import register_initial_conditions_plugin


class DefaultInitialConditions(InitialConditionsABC):
    """
    Default implementation of initial conditions that uses a zero array.

    This class is used when no specific initial conditions are provided in the
    configuration.
    """

    method: Literal["Default"] = "Default"

    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
    ) -> npt.NDArray[np.float64]:
        """
        Produce an array of initial conditions as a zero array.

        Returns:
            A numpy array of initial conditions initialized to zero.
        """
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        y0[0, :] = subpopulation_structure.subpop_pop
        return y0


register_initial_conditions_plugin(DefaultInitialConditions)
