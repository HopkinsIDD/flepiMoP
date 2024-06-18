---
description: >-
  This section describes how to specify the values of each model state at the
  time the simulation starts, and how to make instantaneous changes to state
  values at other times (e.g., due to importations)
---

# Specifying initial conditions and seeding

## Overview

In order for the models specified previously to be dynamically simulated, the user must provide _initial conditions,_ in addition to the model structure and parameter values. Initial conditions describe the value of each variable in the model at the time point that the simulation is to start. For example, on day zero of an outbreak, we may assume that the entire population is susceptible except for one single infected individual. Alternatively, we could assume that some portion of the population already has prior immunity due to vaccination or previous infection. Different initial conditions lead to different model trajectories.

_flepiMoP_ also allows users to specify instantaneous changes in values of model variables, at any time during the simulation. We call this "_seeding_". For example, some individuals in the population may travel or otherwise acquire infection from outside the population throughout the epidemic, and this importation of infection could be specified with the seeding option. As another example, new genetic variants of the pathogen may arise due to mutation and selection that occurs within infected individuals, and this generation of new strains can also be modeled with seeding. Seeding allows individuals to change state at specified times in ways that do not depend on the model equations. In the first example, the individuals would be "seeded" into the infected compartment from the susceptible compartment, and in the second example, individuals would be seeded into the "infected with new variant" compartment from the "infected with wild type" compartment.

The seeding option can also be used as a convenient alternative way to specify initial conditions. By default, _flepiMoP_ initiates models by putting the entire population size (specified in the `geodata` file) in the first model compartment. If the desired initial condition is only slightly different than the default state, it may be more convenient to specify it with a few "seedings" that occur on the first day of the simulation. For example, for a simple SIR model where the desired initial condition is just a small number of infected individuals, this could be specified by a single seeding into the infected compartment from the susceptible compartment at time zero, instead of specifying the initial values of three separate compartments. For larger models, the difference becomes more relevant.

The `seeding` and `initial_conditions` section of the configuration file are detailed below.

This table is useful for a quick comparison of these sections

| Feature                               | initial\_conditions                                                                                                                                                                                                   | seeding                                                                                                                                                                                                         |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Config section optional or required?  | Optional                                                                                                                                                                                                              | Optional                                                                                                                                                                                                        |
| Function of section                   | Specify number of individuals in each compartment at time zero                                                                                                                                                        | Allow for instantaneous changes in individuals' states                                                                                                                                                          |
| Default                               | Entire population in first compartment, zero in all other compartments                                                                                                                                                | No seeding events                                                                                                                                                                                               |
| Requires input file?                  | Yes, .csv                                                                                                                                                                                                             | Yes, .csv                                                                                                                                                                                                       |
| Input description                     | Input is a list of compartment names, location names, and amounts of individuals in that compartment location. All compartments must be listed unless a setting to default missing compartments to zero is turned on. | Input is list of seeding events defined by source compartment, destination compartment, number of individuals transitioning, and date of movement. Compartments without seeding events don't need to be listed. |
| Specifies an incidence or prevalence? | Amounts specified are prevalence values                                                                                                                                                                               | Amounts specified are instantaneous incidence values                                                                                                                                                            |
| Useful for?                           | Specifying initial conditions, especially if simulation does not start with a single infection introduced into a naive population.                                                                                    | Modeling importations, evolution of new strains, and specifying initial conditions                                                                                                                              |

An example configuration file containing both initial condition and seeding section is given below.

## Specifying model seeding

The configuration items in the `seeding` section of the config file are

`seeding:method` Must be either `"NoSeeding"`, `"FromFile"`, `"PoissonDistributed"`, `"NegativeBinomialDistributed"`, or `"FolderDraw".`

`seeding::seeding_file` Only required for `method: “FromFile”.` Path to a .csv file containing the list of seeding events

`seeding::lambda_file` Only required for methods `"PoissonDistributed"` or `"NegativeBinomialDistributed".` Path to a .csv file containing the list of the events from which the actual seeding will be randomly drawn.

`seeding::seeding_file_type` Only required for method `"FolderDraw".` Either `seir` or `seed`

Details on implementing each seeding method and the options that go along with it are below.

### seeding::method

#### NoSeeding

If there is no seeding, then the amount of individuals in each compartment will be initiated using the values specified in the`initial_conditions` section and will only be changed at later times based on the equations defined in the `seir` section. No other arguments are needed in the seeding section in this case

Example

```
seeding:
    method: “NoSeeding”
```

#### FromFile

This seeding method reads in a user-defined file with a list of seeding events (instantaneous transitions of individuals between compartments) including the time of the event and subpopulation where it occurs, and the source and destination compartment of the individuals. For example, for the simple two-subpopulation SIR model where the outbreak starts with 5 individuals in the small province being infected from a source outside the population, the seeding section of the config could be specified as

```
seeding:
  method: "FromFile"
  seeding_file: seeding_2pop.csv
```

Where seeding.csv contains

```
subpop, date, amount, source_infection_stage, destination_infection_stage
small_province, 2020-02-01, 5, S, E
```

`seeding::seeding_file` must contain the following columns:

* `subpop` – the name of the subpopulation in which the seeding event takes place. Seeding cannot move individuals between different subpopulations.
* `date` – the date the seeding event occurs, in YYYY-MM-DD format
* `amount` – an integer value for the amount of individuals who transition between states in the seeding event
* `source_*` and `destination_*` – For each compartment group (i.e., infection stage, vaccination stage, age group), a different column describes the status of individuals before and after the transition described by the seeding event. For example, for a model where individuals are stratified by age and vaccination status, and a 1-day vaccination campaign for young children and the elderly moves a large number of individuals into a vaccinated state, this file could be something like

```
subpop, date, amount, source_infection_stage, source_vaccine_doses, source_age_group, destination_infection_stage, destination_vaccine_doses, destination_age_group
anytown, 1950-03-15, 452, S, 0dose, under5years, S, 1dose, under5years
anytown, 1950-03-16, 527, S, 0dose, 5_10years, S, 1dose, 5_10years
anytown, 1950-03-17, 1153, S, 0dose, over65years, S, 1dose, over65years
```

#### PoissonDistributed or NegativeBinomialDistributed

These methods are very similar to FromFile, except the seeding value used in the simulation is randomly drawn from the seeding value specified in the file, with an average value equal to the file value. These methods can be useful when the true seeded value is unknown, and only an observed value is available which is assumed to be observed with some uncertainty. The input requirements are the same for both distributions

```
seeding:
  method: "PoissonDistributed"
  lambda_file: seeding.csv
```

or

```
seeding:
  method: "NegativeBinomialDistributed"
  lambda_file: seeding.csv
```

and the `lambda_file` has the same format requirements as the `seeding_file` for the FromFile method described above.

For `method::PoissonDistributed`, the seeding value for each seeding event is drawn from a Poisson distribution with mean and variance equal to the value in the `amount` column. For`method::NegativeBinomialDistributed`, seeding is drawn from a negative binomial distribution with mean `amount` and variance `amount+5` (so identical to `"PoissonDistributed"` for large values of `amount` but has higher variance for small values).

#### FolderDraw

TBA

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

