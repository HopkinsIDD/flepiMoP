---
description: How to plug-in your code/data directly into flepiMoP
---

# Using plug-ins 🧩\[experimental]

Sometimes, the default modules, such as seeding, or initial condition, do not provide the desired functionality. Thankfully, it is possible to replace a gempyor module with your own code, using plug-ins. This works only for initial conditions and seeding at the moment, reach out to us if you are interested in having it works on parameters, modifiers, ...

Here is an example, to set a random initial condition, where each subpopulation a random proportion of individuals is infected. For this, simply set the `method` of a block to `plugin` and provide the path of your file.

```yaml
initial_conditions:
  method: plugin
  plugin_file_path: model_input/my_initial_conditions.py
  # you can also include some configuration for your plugin:
  ub_prop_infected: 0.001 # upper bound of the uniform distribution
```

This file contains a class that inherits from a gempyor class, which means that everything already defined in gempyor is available but you can overwrite any single method. Here, we will rewrite the load and draw methods of the initial conditions methods

```python
import gempyor.seeding_ic
import numpy as np

class InitialConditions(gempyor.seeding_ic.InitialConditions):

    def get_from_config(self, sim_id: int, setup) -> np.ndarray:
        y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nsubpops))
        S_idx = setup.compartments.get_comp_idx({"infection_stage":"S"})
        I_idx = setup.compartments.get_comp_idx({"infection_stage":"I"})
        prop_inf = np.random.uniform(low=0,high=self.config["ub_prop_infected"].get(), size=setup.nsubpops)
        y0[S_idx, :] = setup.subpop_pop * (1-prop_inf)
        y0[I_idx, :] = setup.subpop_pop * prop_inf
        
        return y0
    
    def get_from_file(self, sim_id: int, setup) -> np.ndarray:
        return self.get_from_config(sim_id=sim_id, setup=setup)
```

You can use any code within these functions, as long as the return object has the shape and type that gempyor expect (and that is _undocumented_ and still subject to change, but as you see in this case gempyor except an array (a matrix) of shape: number of compartments X number of subpopulations). You can e.g call bash functions or excute R scripts such as below

```python
import gempyor.seeding_ic
import numpy as np

class InitialConditions(gempyor.seeding_ic.InitialConditions):

    def get_from_config(self, sim_id: int, setup) -> np.ndarray:
        import rpy2.robjects as robjects
        robjects.r.source("path_to_your_Rscript.R", encoding="utf-8")
        y0 = robjects.r["initial_condition_fromR"]
        return y0
    
    def get_from_file(self, sim_id: int, setup) -> np.ndarray:
        return self.get_from_config(sim_id=sim_id, setup=setup)
```
