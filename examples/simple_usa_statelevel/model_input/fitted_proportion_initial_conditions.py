from typing import Literal

from gempyor.compartments import Compartments
from gempyor.initial_conditions import (
    InitialConditionsABC,
    register_initial_conditions_plugin,
)
from gempyor.subpopulation_structure import SubpopulationStructure
import numpy as np
import numpy.typing as npt
from scipy.special import softmax


class FittedProportionInitialConditions(InitialConditionsABC):
    method: Literal["FittedProportion"] = "FittedProportion"

    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
        So: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        S_idx = compartments.get_comp_idx({"infection_stage": "S"})
        I_idx = compartments.get_comp_idx({"infection_stage": "I"})
        subpop_pop = np.array(subpopulation_structure.subpop_pop, dtype=np.float64)
        ones_like = np.ones_like(subpop_pop)
        y0[[S_idx, I_idx], :] = subpop_pop * softmax(
            np.stack((ones_like, So), axis=1).T, axis=0
        )
        return y0


register_initial_conditions_plugin(FittedProportionInitialConditions)
