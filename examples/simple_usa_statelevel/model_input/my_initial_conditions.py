import gempyor.initial_conditions
import numpy as np


class InitialConditions(gempyor.initial_conditions.InitialConditions):
    def get_from_config(self, sim_id: int, modinf) -> np.ndarray:
        y0 = np.zeros((modinf.compartments.compartments.shape[0], modinf.nsubpops))
        S_idx = modinf.compartments.get_comp_idx({"infection_stage": "S"})
        I_idx = modinf.compartments.get_comp_idx({"infection_stage": "I"})
        prop_inf = 0.005  # np.random.uniform(low=0, high=0.01, size=modinf.nsubpops)
        # TODO cause this is another example
        y0[S_idx, :] = modinf.subpop_pop * (1 - prop_inf)
        y0[I_idx, :] = modinf.subpop_pop * prop_inf

        return y0

    def get_from_file(self, sim_id: int, modinf) -> np.ndarray:
        return self.draw(sim_id=sim_id, modinf=modinf)
