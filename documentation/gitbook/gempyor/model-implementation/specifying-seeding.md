---
description: >-
  This section describes how to specify the values of each model state at the
  time the simulation starts, and how to make instantaneous changes to state
  values at other times (e.g., due to importations)
---

# Specifying seeding

## Overview

_flepiMoP_ allows users to specify instantaneous changes in values of model variables, at any time during the simulation. We call this "_seeding_". For example, some individuals in the population may travel or otherwise acquire infection from outside the population throughout the epidemic, and this importation of infection could be specified with the seeding option. As another example, new genetic variants of the pathogen may arise due to mutation and selection that occurs within infected individuals, and this generation of new strains can also be modeled with seeding. Seeding allows individuals to change state at specified times in ways that do not depend on the model equations. In the first example, the individuals would be "seeded" into the infected compartment from the susceptible compartment, and in the second example, individuals would be seeded into the "infected with new variant" compartment from the "infected with wild type" compartment.

The seeding option can also be used as a convenient alternative way to specify [initial conditions](specifying-seeding.md#specifying-model-initial-conditions). By default, _flepiMoP_ initiates models by putting the entire population size (specified in the `geodata` file) in the first model compartment. If the desired initial condition is only slightly different than the default state, it may be more convenient to specify it with a few "seedings" that occur on the first day of the simulation. For example, for a simple SIR model where the desired initial condition is just a small number of infected individuals, this could be specified by a single seeding into the infected compartment from the susceptible compartment at time zero, instead of specifying the initial values of three separate compartments. For larger models, the difference becomes more relevant.

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

TBA&#x20;

