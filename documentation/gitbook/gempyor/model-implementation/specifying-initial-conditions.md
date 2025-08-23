---
description: |
  This section describes how to specify the values
  of each model state at the simulation start time.
---

# Specifying Initial Conditions

## Overview

In order for the previously specified models to be dynamically simulated the state of the system at the initial time must be specified, which is done via initial conditions. Initial conditions describe the value of each variable in the model at the time point that the simulation is to start. For example, on day zero of an outbreak, we may assume that the entire population is susceptible except for one infected individual. Alternatively, we could assume that some portion of the population already has prior immunity due to vaccination or previous infection. Different initial conditions lead to different model trajectories.

The `initial_conditions` section of the configuration file is detailed below. Note that in some cases, [the `seeding` section](specifying-seeding.md) can replace or complement the initial condition. To help you decide which to use please refer to the [Initial Conditions vs Seeding](./initial-conditions-vs-seeding.md) guide.

## Builtin Initial Condition Methods

Initial conditions are specified by the user in a configuration section named `initial_conditions`. It's not required that a user add this to their configuration file, but if this is not present a warning will be issued and it is assumed that the 'Default' method will be used for initial conditions. All initial conditions require a key called `method` that is a string corresponding to the type of initial conditions to use. For example, the minimum initial conditions configuration would be:

```yaml
initial_conditions:
  method: Default
```

Further configuration placed under the `initial_conditions` section is passed on to the method used and depends on the method. The methods included with `gempyor` are 'Default', 'SetInitialConditions', 'SetInitialConditionsFolderDraw', 'FromFile', 'InitialConditionsFolderDraw'.

### 'Default' Method

The 'Default' initial conditions will place the entire subpopulation into the first compartment specified by the model. For example:

```yaml
compartments:
  infection_stage: ["S", "I", "R"]
 
initial_conditions:
  method: Default
```

Would place the entire subpopulation into the 'S' compartment. For additional layers of stratification, the same logic applies. For example:

```yaml
compartments:
  infection_stage: ["S", "I", "R"]
  age_group: ["child", "adult"]
  vaccination_status: ["unvaxxed", "vaxxed"]
 
initial_conditions:
  method: Default
```

Would place the entire subpopulation into the 'S', 'child', 'unvaxxed' compartment. The Default behavior is typically only useful when doing initial configuration troubleshooting (e.g. when correcting parsing errors): because all population is placed in the "first" compartment, there often won't be any dynamics (e.g. because all 'S' is the disease free equilibrium) and the population distribution may be nonsensical (e.g. creating an all-'child' Lord of the Flies situation).

### 'SetInitialConditions' Method

With this method users can specify arbitrary initial conditions in a conveniently formatted input CSV or parquet file. For example, for a model with the following `compartments` and `initial_conditions` sections:

```yaml
compartments:
  infection_stage: ["S", "I", "R"]
  age_group: ["child", "adult"]
  vaccination_status: ["unvaxxed", "vaxxed"]
   
initial_conditions:
  method: SetInitialConditions
  initial_conditions_file: initial_conditions.csv
  allow_missing_subpops: TRUE
  allow_missing_compartments: TRUE
```

With the accompanying geodata file:

```csv
        subpop, population
large_province,      10000
small_province,       1000
```

Where `initial_conditions.csv` contains:

```csv
        subpop,          mc_name, amount
small_province, S_child_unvaxxed,    500
small_province, S_adult_unvaxxed,    500
large_province, S_child_unvaxxed,   5000
large_province, E_adult_unvaxxed,      5
large_province, S_adult_unvaxxed,   rest
```

The model will be started with half of the population of both subpopulations, consisting of children and the other half of adults, everyone unvaccinated, and 5 infections (in exposed-but-not-yet-infectious class) among the unvaccinated adults in the large province, with the remaining individuals susceptible (4995). All other compartments will contain zero individuals initially.

The initial conditions file must contain the following columns:

* `subpop` – The name of the subpopulation for which the initial condition is being specified. By default, all subpopulations must be listed in this file, unless the `allow_missing_subpops` option is set to `TRUE`.
* `mc_name` – The concatenated name of the compartment for which an initial condition is being specified. The order of the compartment groups in the name must be the same as the order in which these groups are defined in the config for the model, e.g., you cannot say 'unvaccinated_S'.
* `amount` – The value of the initial condition; either a numeric value or the string 'rest'.

For each subpopulation, if there are compartments that are not listed in `SetInitialConditions`, an error will be thrown unless `allow_missing_compartments` is set to `TRUE`, in which case it will be assumed there are zero individuals in them. If the sum of the values of the initial conditions in all compartments in a location does not add up to the total population of that location (specified in the geodata file), an error will be thrown. To allocate all remaining individuals in a subpopulation (the difference between the total population size and those allocated by defined initial conditions) to a single pre-specified compartment, include this compartment in the `initial_conditions_file` but instead of a number in the `amount` column, put the word 'rest'.

If `allow_missing_subpops` is `FALSE` or unspecified, an error will occur if initial conditions for some subpopulations are missing. If `TRUE`, then for subpopulations missing from the `initial_conditions` file, it will be assumed that all individuals begin in the first compartment. The first compartment depends on how the model was specified, and will be the compartment that contains the first named category in each compartment group.

### 'FromFile' Method

Similar to 'SetInitialConditions', with this method users can specify arbitrary initial conditions in a formatted CSV or parquet input file. However, the format of the input file is different. The required file format is consistent with the [output "seir" file ](../output-files.md#seir-infection-model-output) from the compartmental model, so the user could take output from one simulation and use it as input into another simulation with the same model structure.

For example, for an input configuration file containing:

```yaml
compartments:
  infection_stage: ["S", "I", "R"]
  age_group: ["child", "adult"]
  vaccination_status: ["unvaxxed", "vaxxed"]
   
initial_conditions:
  method: FromFile
  initial_conditions_file: initial_conditions_from_previous.csv
  allow_missing_compartments: FALSE
  allow_missing_subpops: FALSE
```

With the accompanying geodata file:

```csv
        subpop, population
large_province,      10000
small_province,       1000
```

Where `initial_conditions_from_previous.csv` contains:

```csv
mc_value_type, mc_infection_stage, mc_age, mc_vaccination_status, small_province, large_province,       date
...
   prevalence,                  S,  child,              unvaxxed,            400,            900, 2021-06-01
   prevalence,                  S,  child,                vaxxed,              0,              0, 2021-06-01
   prevalence,                  I,  child,              unvaxxed,              5,            100, 2021-06-01
   prevalence,                  I,  child,                vaxxed,              0,              0, 2021-06-01
   prevalence,                  R,  child,              unvaxxed,             95,           4000, 2021-06-01
   prevalence,                  R,  child,                vaxxed,              0,              0, 2021-06-01
   prevalence,                  S,  adult,              unvaxxed,             50,            900, 2021-06-01
   prevalence,                  S,  adult,                vaxxed,            400,              0, 2021-06-01
   prevalence,                  I,  adult,              unvaxxed,              4,            100, 2021-06-01
   prevalence,                  I,  adult,                vaxxed,              1,              0, 2021-06-01
   prevalence,                  R,  adult,              unvaxxed,             75,           4000, 2021-06-01
   prevalence,                  R,  adult,                vaxxed,             20,              0, 2021-06-01
...
```

The simulation would be initiated on 2021-06-01 with these values in each compartment.

The initial conditions file must contain the following columns:

* `mc_value_type` – In model output files, this is either 'prevalence' or 'incidence', but only prevalence values are selected to be used as initial conditions, since compartmental models described the prevalence (number of individuals at any given time) in each compartment.
* `mc_name` – The name of the compartment for which the value is reported, which is a concatenation of the compartment status in each state type, e.g. "S\_adult\_unvaxxed" and must be in the same order as these groups are defined in the config for the model, e.g., you cannot say `unvaxxed_S_adult`.
* `subpop_1`, `subpop_2`, etc. – One column for each different subpopulation, containing the value of the number of individuals in the described compartment in that subpopulation at the given date. Note that these are named after the node names defined by the user in the `geodata` file.
* `date` – The calendar date in the simulation, in YYYY-MM-DD format. Only values with a date that matches to the simulation `start_date` will be used.

### 'SetInitialConditionsFolderDraw' and 'InitialConditionsFolderDraw' Methods

The way that initial conditions is specified with 'SetInitialConditions' and 'FromFile' results in a single value for each compartment and does not easily allow the user to instead specify a distribution, like is possible for compartmental or outcome model parameters. If a user wants to use different possible initial condition values each time the model is run, the way to do this is to instead specify a folder containing a set of file with initial condition values for each simulation that will be run. The user can do this using files with the format described in 'SetInitialConditions' using instead 'method::SetInitialConditionsFolder' draw, and similarly for 'FromFile' using instead 'FromFileFolderDraw'.

Each file in the folder needs to be named according to the same naming conventions as the model output files, `run_number.run_id.file_type.{csv,parquet}`, where as it is now taking the place of the seeding files the model would normally output. Only one additional config argument is needed to use a FolderDraw method for initial conditions:

* `initial_file_type`: Either 'seir' or 'seed'.

### Overview Of Methods

Below is a table describing the configuration options that each method provides, except `method` which determines the method.

| Method                         | Configuration              | Type/Format                     | Description                                                                                               |
|--------------------------------|----------------------------|---------------------------------|-----------------------------------------------------------------------------------------------------------|
| Default                        | _N/A_                      | _N/A_                           | The Default method uses no other configuration options.                                                   |
| SetInitialConditions           | initial_conditions_file    | File path                       | The file to use for the SetInitialConditions method (format described above).                             |
|                                | allow_missing_subpops      | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all subpopulations.                   |
|                                | allow_missing_compartments | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all compartments.                     |
|                                | proportional_ic            | Boolean, defaults to `False`    | If set to `True` the input amount is treated as a percentage of the subpopulation population to allocate. |
| FromFile                       | initial_conditions_file    | File path                       | The file to use for the FromFile method (format described above).                                         |
|                                | allow_missing_subpops      | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all subpopulations.                   |
|                                | allow_missing_compartments | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all compartments.                     |
| SetInitialConditionsFolderDraw | initial_file_type          | String, either 'seir' or 'seed' | Whether to pull the initial conditions from the 'seir' or 'seed' model outputs.                           |
|                                | allow_missing_subpops      | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all subpopulations.                   |
|                                | allow_missing_compartments | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all compartments.                     |
|                                | proportional_ic            | Boolean, defaults to `False`    | If set to `True` the input amount is treated as a percentage of the subpopulation population to allocate. |
| InitialConditionsFolderDraw    | initial_file_type          | String, either 'seir' or 'seed' | Whether to pull the initial conditions from the 'seir' or 'seed' model outputs.                           |
|                                | allow_missing_subpops      | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all subpopulations.                   |
|                                | allow_missing_compartments | Boolean, defaults to `False`    | Whether or not it is okay for the input file to not explicitly list all compartments.                     |

## Plugins

In addition to the builtin methods users can also code their own initial conditions plugins. Under the hood the builtin initial condition methods are actually themselves plugins. To implement a plugin place a python file in the working directory, say `model_input/my_initial_conditions.py` for example, which contains a class that subclasses `gempyor.initial_conditions.InitialConditionsABC`. For example:

```python
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
from pydantic import Field

class TwoCompartmentInitialConditions(InitialConditionsABC):
    method: Literal["TwoCompartment"] = "TwoCompartment"
    weight: float = Field(gt=0.0, lt=1.0)
    
    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
    ) -> npt.NDArray[np.float64]:
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        y0[0, :] = self.weight * subpopulation_structure.subpop_pop
        y0[1, :] = (1.0 - self.weight) * subpopulation_structure.subpop_pop
        return y0

register_initial_conditions_plugin(TwoCompartmentInitialConditions)
```

This file:

* Contains an initial conditions plugin class, `TwoCompartmentInitialConditions`, that subclasses `InitialConditionsABC`.
  * Has a required `method` attribute that determines this plugin's type. Note that plugins can have multiple values for `method` if they implement different behaviors.
  * Has an additional `weight` attribute which is required to be configured by the user since it does not have a default value.
  * Implements a `create_initial_conditions` method that takes at least `sim_id`, `compartments`, and `subpopulation_structure` as arguments and returns a numpy array of dimension `(# comparments, # subpopulations)`.
    * This plugin weights the subpopulation population between the first two compartments by the `weight` attribute.
* Registers the initial conditions plugin with `gempyor` by calling the `register_initial_conditions_plugin` function.

To use this plugin, your configuration file would look like:

```yaml
compartments:
  infection_stage: ["S", "I", "R"]
 
initial_conditions:
  method: TwoCompartment
  module: model_input/my_initial_conditions.py
  weight: 0.5
```

Where the `module` option tells `gempyor` where to search for plugins. Note that you could have more than one plugin in this file and switch between them using the `method` configuration. In this case this will evenly split the subpopulation population between the 'S' and 'I' compartments for the initial conditions.

### Requesting Parameters

To integrate initial conditions with parameters a plugin can "request" parameters be passed to the `create_initial_conditions` method by adding them as arguments. For example, suppose you have a parameters/modifiers section like:

```yaml
seir:
  parameters:
    alpha:
      value: 1.0
  ...

seir_modifiers:
  ...
  alpha_mod:
    method: SinglePeriodModifier
    parameter: alpha
    period_start_date: 2023-08-01
    period_end_date: 2024-07-31
    subpop: all
    subpop_groups: all
    value:
      distribution: truncnorm
      mean: 0.0
      sd: 2.0
      a: -4.0
      b: 4.0
    perturbation: # Not used by EMCEE, but required to mark this as inferable
      distribution: truncnorm
      mean: 0
      sd: 0.25
      a: -1
      b: 1
  ...
```

EMCEE only fits modifiers so what this config above does is:

* Declare the existence of a parameter called `alpha` which is set to 1 by default,
* Declare a modifier `alpha_mod` that modifies `alpha` by:
  * Since the reduction method isn't specified the modifier value is multiplied by the original value,
  * The `period_start_date` and `period_end_date` span the full range of the simulation so the modifier effectively becomes the parameter,
  * The `subpop: all` and `subpop_groups: all` forces this modifier to affect all subpopulations the same way,
  * The value specifies the initial distribution of the modifier.

And your custom plugin looked like:

```python
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
        weight = expit(alpha).item()
        y0 = np.zeros((len(compartments.compartments), subpopulation_structure.nsubpops))
        y0[0, :] = [weight * pop for pop in subpopulation_structure.subpop_pop]
        y0[1, :] = [(1.0 - weight) * pop for pop in subpopulation_structure.subpop_pop]
        return y0

register_initial_conditions_plugin(TwoCompartmentInitialConditions)
```

Note that in the above plugin an additional argument has been added to `create_initial_conditions` whose name, `alpha`, corresponds to the SEIR parameter name. You can finally use this plugin in your configuration file:

```yaml
compartments:
  infection_stage: ["S", "I", "R"]

initial_conditions:
  method: TwoCompartment
  module: model_input/my_initial_conditions.py
```

Furthermore, users can also request timeseries parameters as well and these will be provided as a numpy array with shape `(# days, # subpopulations)`. If you're plugin requests a parameter that is not found in the configuration file you can an expect an error like:

```
Traceback (most recent call last):
  ...
ValueError: The requested parameter, 'alph', not found in the arguments of create_initial_conditions. The available parameters are: dict_keys(['alpha']).
```

The above error was raised because 'alph', a clear typo of 'alpha', was not found in the available parameters in the configuration.

#### Further Examples

For more examples of plugins, especially ones that request parameters available in the configuration so they are fittable, see:

* The `config_sample_2pop_inference_with_initial_conditions.yml` configuration file in the `examples/tutorial/` project directory that shows how a custom plugin can be done with an R inference config, and
* The `simple_usa_statelevel.yml` configuration file in the `examples/simple_usa_statelevel/` project directory that shows how a custom plugin can be done with EMCEE inference config.

Both of these configurations and their initial condition plugins are very similar but provide a great starting point for implementing your own fittable initial conditions plugin with your preferred inference method.
