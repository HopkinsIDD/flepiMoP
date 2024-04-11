import gempyor.seeding_ic
import numpy as np

class InitialConditions(gempyor.seeding_ic.InitialConditions):

    def draw(self, sim_id: int, setup) -> np.ndarray:
        y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nsubpops))
        S_idx = setup.compartments.get_comp_idx({"infection_stage":"S"})
        I_idx = setup.compartments.get_comp_idx({"infection_stage":"I"})
        prop_inf = np.random.uniform(low=0,high=.01, size=setup.nsubpops)
        y0[S_idx, :] = setup.subpop_pop * (1-prop_inf)
        y0[I_idx, :] = setup.subpop_pop * prop_inf

        return y0
    
    def load(self, sim_id: int, setup) -> np.ndarray:
        return self.draw(sim_id=sim_id, setup=setup)
