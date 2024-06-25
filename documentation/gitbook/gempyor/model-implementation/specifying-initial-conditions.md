---
description: >-
  This section describes how to specify the values of each model state at the
  time the simulation starts, and how to make instantaneous changes to state
  values at other times (e.g., due to importations)
---

# Specifying initial conditions

## Overview

In order for the models specified previously to be dynamically simulated, the user must provide _initial conditions,_ in addition to the model structure and parameter values. Initial conditions describe the value of each variable in the model at the time point that the simulation is to start. For example, on day zero of an outbreak, we may assume that the entire population is susceptible except for one single infected individual. Alternatively, we could assume that some portion of the population already has prior immunity due to vaccination or previous infection. Different initial conditions lead to different model trajectories.

The `initial_conditions` section of the configuration file is detailed below. Note that in some cases, [the `seeding` section](specifying-seeding.md) can replace or complement the initial condition, the table below provides a quick comparison of these sections.

| Feature                               | initial\_conditions                                                                                                                                                                                                   | seeding                                                                                                                                                                                                         |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Config section optional or required?  | Optional                                                                                                                                                                                                              | Optional                                                                                                                                                                                                        |
| Function of section                   | Specify number of individuals in each compartment at time zero                                                                                                                                                        | Allow for instantaneous changes in individuals' states                                                                                                                                                          |
| Default                               | Entire population in first compartment, zero in all other compartments                                                                                                                                                | No seeding events                                                                                                                                                                                               |
| Requires input file?                  | Yes, .csv                                                                                                                                                                                                             | Yes, .csv                                                                                                                                                                                                       |
| Input description                     | Input is a list of compartment names, location names, and amounts of individuals in that compartment location. All compartments must be listed unless a setting to default missing compartments to zero is turned on. | Input is list of seeding events defined by source compartment, destination compartment, number of individuals transitioning, and date of movement. Compartments without seeding events don't need to be listed. |
| Specifies an incidence or prevalence? | Amounts specified are prevalence values                                                                                                                                                                               | Amounts specified are instantaneous incidence values                                                                                                                                                            |
| Useful for?                           | Specifying initial conditions, especially if simulation does not start with a single infection introduced into a naive population.                                                                                    | Modeling importations, evolution of new strains, and specifying initial conditions                                                                                                                              |

## Specifying model initial conditions

The configuration items in the `initial_conditions` section of the config file are

`initial_conditions:method` Must be either `"Default"`, `"SetInitialConditions"`, or `"FromFile".`

`initial_conditions:initial_conditions_file`Required for methods “`SetInitialConditions`” and “`FromFile`” . Path to a .csv or .parquet file containing the list of initial conditions for each compartment.

`initial_conditions:initial_file_type` Only required for `method: “FolderDraw”`. Description TBA

`initial_conditions::allow_missing_subpops` Optional for all methods, determines what will happen if `initial_conditions_file` is missing values for some subpopulations. If FALSE, the default behavior, or unspecified, an error will occur if subpopulations are missing. If TRUE, then for subpopulations missing from the `initial_conditions` file, it will be assumed that all individuals begin in the first compartment (the “first” compartment depends on how the model was specified, and will be the compartment that contains the first named category in each compartment group), unless another compartment is designated to hold the rest of the individuals.&#x20;

`initial_conditions::allow_missing_compartments` Optional for all methods. If FALSE, the default behavior, or unspecified, an error will occur if any compartments are missing for any subpopulation. If TRUE, then it will be assumed there are zero individuals in compartments missing from the `initial_conditions file`.

`initial_conditions::proportional` If TRUE, assume that the user has specified all input initial conditions as fractions of the population, instead of numbers of individuals (the default behavior, or if set to FALSE). Code will check that initial values in all compartments sum to 1.0 and throw an error if not, and then will multiply all values by the total population size for that subpopulation.&#x20;

Details on implementing each initial conditions method and the options that go along with it are below.

### `initial_conditions::method`

#### Default

The default initial conditions are that the initial value of all compartments for each subpopulation will be zero, except for the first compartment, whose value will be the population size. The “first” compartment depends on how the model was specified, and will be the compartment that contains the first named category in each compartment group.

For example, a model with the following compartments

```
 compartments:
   infection_stage: ["S", "I", "R"]
   age_group: ["child", "adult"]
   vaccination_status: ["unvaxxed", "vaxxed"]
 
 initial_conditions:
   method: default
```

with the accompanying geodata file

```
subpop,          population
large_province, 10000
small_province, 1000
```

will be started with 1000 individuals in the S\_child\_unvaxxed in the "small province" and 10,000 in that compartment in the "large province".

#### SetInitialConditions

With this method users can specify arbitrary initial conditions in a convenient formatted input .csv or .parquet file.

For example, for a model with the following `compartments` and `initial_conditions` sections

```
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

with the accompanying geodata file

```
subpop,          population
large_province, 10000
small_province, 1000
```

where `initial_conditions.csv` contains

```
subpop, mc_name, amount
small_province, S_child_unvaxxed, 500
small_province, S_adult_unvaxxed, 500
large_province, S_child_unvaxxed, 5000
large_province, E_adult_unvaxxed, 5
large_province, S_adult_unvaxxed, "rest"
```

the model will be started with half of the population of both subpopulations, consisting of children and the other half of adults, everyone unvaccinated, and 5 infections (in exposed-but-not-yet-infectious class) among the unvaccinated adults in the large province, with the remaining individuals susceptible (4995).  All other compartments will contain zero individuals initially.&#x20;

`initial_conditions::initial_conditions_file` must contain the following columns:

* `subpop` – the name of the subpopulation for which the initial condition is being specified. By default, all subpopulations must be listed in this file, unless the `allow_missing_subpops` option is set to TRUE.
* `mc_name` – the concatenated name of the compartment for which an initial condition is being specified. The order of the compartment groups in the name must be the same as the order in which these groups are defined in the config for the model, e.g., you cannot say `unvaccinated_S`.
* `amount` – the value of the initial condition; either a numeric value or the string "rest".

For each subpopulation, if there are compartments that are not listed in `SetInitialConditions`, an error will be thrown unless `allow_missing_compartments` is set to TRUE, in which case it will be assumed there are zero individuals in them. If the sum of the values of the initial conditions in all compartments in a location does not add up to the total population of that location (specified in the geodata file), an error will be thrown. To allocate all remaining individuals in a subpopulation (the difference between the total population size and those allocated by defined initial conditions) to a single pre-specified compartment, include this compartment in the `initial_conditions_file` but instead of a number in the `amount` column, put the word "rest".&#x20;

If `allow_missing_subpops` is FALSE or unspecified, an error will occur if initial conditions for some subpopulations are missing. If TRUE, then for subpopulations missing from the `initial_conditions` file, it will be assumed that all individuals begin in the first compartment. (The “first” compartment depends on how the model was specified, and will be the compartment that contains the first named category in each compartment group.)

#### FromFile

Similar to `"SetInitialConditions"`, with this method users can specify arbitrary initial conditions in a formatted .csv or .parquet input file. However, the format of the input file is different. The required file format is consistent with the [output "seir" file ](../output-files.md#seir-infection-model-output)from the compartmental model, so the user could take output from one simulation and use it as input into another simulation with the same model structure.&#x20;

For example, for an input configuration file containing

```
name: test_simulation
start_date: 2021-06-01

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

with the accompanying geodata file

```
subpop,          population
large_province, 10000
small_province, 1000
```

where `initial_conditions_from_previous.csv` contains

```
mc_value_type, mc_infection_stage, mc_age, mc_vaccination_status, mc_name, small_province, large_province, date
....
prevalence, S, child, unvaxxed, 400, 900, 2021-06-01
prevalence, S, child, vaxxed, 0, 0, 2021-06-01
prevalence, I, child, unvaxxed, 5, 100, 2021-06-01
prevalence, I, child, vaxxed, 0, 0, 2021-06-01
prevalence, R, child, unvaxxed, 95, 4000, 2021-06-01
prevalence, R, child, vaxxed, 0, 0, 2021-06-01
prevalence, S, adult, unvaxxed, 50, 900, 2021-06-01
prevalence, S, adult, vaxxed, 400, 0, 2021-06-01
prevalence, I, adult, unvaxxed, 4, 100, 2021-06-01
prevalence, I, adult, vaxxed, 1, 0, 2021-06-01
prevalence, R, adult, unvaxxed, 75, 4000, 2021-06-01
prevalence, R, adult, vaxxed, 20, 0, 2021-06-01
...
```

The simulation would be initiated on 2021-06-01 with these values in each compartment (no children vaccinated, only adults in the small province vaccinated, some past and current infection in both compartments but ).

`initial_conditions::initial_conditions_file` must contain the following columns:

* `mc_value_type` – in model output files, this is either `prevalence` or `incidence`. Prevalence values only are selected to be used as initial conditions, since compartmental models described the prevalence (number of individuals at any given time) in each compartment. Prevalence is taken to be the value measured instantaneously at the start of the day
* `mc_name` – The name of the compartment for which the value is reported, which is a concatenation of the compartment status in each state type, e.g. "S\_adult\_unvaxxed" and must be in the same order as these groups are defined in the config for the model, e.g., you cannot say `unvaxxed_S_adult`.
* `subpop_1`, `subpop_2`, etc. – one column for each different subpopulation, containing the value of the number of individuals in the described compartment in that subpopulation at the given date. Note that these are named after the nodenames defined by the user in the `geodata` file.
* `date` – The calendar date in the simulation, in YYYY-MM-DD format. Only values with a date that matches to the simulation `start_date` will be used.&#x20;

#### SetInitialConditionsFolderDraw, FromFileFolderDraw

The way that initial conditions is specified with `SetInitialConditions` and `FromFile` results in a single value for each compartment and does not easily allow the user to instead specify a distribution (like is possible for compartmental or outcome model parameters). If a user wants to use different possible initial condition values each time the model is run, the way to do this is to instead specify a folder containing a set of file with initial condition values for each simulation that will be run. The user can do this using files with the format described in i`nitial_conditions::method::SetInitialConditions` using instead `method::SetInitialConditionsFolder` draw. Similarly, to provide a folder of initial condition files with the format described in `initial_conditions::method:FromFile` using instead `method::FromFileFolderDraw`.&#x20;

Each file in the folder needs to be named according to the same naming conventions as the model output files: run\_number.runID.file\_type.\[csv or parquet] where ....\[DESCRIBE] as it is now taking the place of the seeding files the model would normally output&#x20;

Only one additional config argument is needed to use a FolderDraw method for initial conditions:

`initial_file_type`:  either `seir` or `seed`

When using FolderDraw methods, `initial_conditions_file` should now be the path to the directory that contains the folder with all the initial conditions files. For example, if you are using output from another model run and so the files are in an seir folder within a model\_output folder which is in within your project directory, you would use initial\_conditions\_file: model\_output&#x20;

