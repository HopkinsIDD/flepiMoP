from typing import Literal
from gempyor.compartments import Compartments
from gempyor.initial_conditions import (
    InitialConditionsABC,
    register_initial_conditions_plugin,
)
from gempyor.subpopulation_structure import SubpopulationStructure
import numpy as np
import numpy.typing as npt
from scipy.special import expit


class TwoCompartmentInitialConditions(InitialConditionsABC):
    method: Literal["TwoCompartment"] = "TwoCompartment"

    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
        alpha: float,
    ) -> npt.NDArray[np.float64]:
        weight = expit(alpha)
        subpopulation_population = np.array(
            subpopulation_structure.subpop_pop, dtype=np.float64
        )
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        y0[0, :] = weight * subpopulation_population
        y0[1, :] = (1.0 - weight) * subpopulation_population
        return y0


register_initial_conditions_plugin(TwoCompartmentInitialConditions)
