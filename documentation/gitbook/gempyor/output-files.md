---
description: >-
  (This section describes the location and contents of each of the output files
  produced during a non-inference model run)
---

# Model Output

The model will output 2–6 different types of files depending on whether the configuration file contains optional sections (such [interventions](model-implementation/intervention-templates.md), [outcomes](model-implementation/outcomes-for-compartments.md), and outcomes interventions) and whether [model inference](https://github.com/HopkinsIDD/flepimop-documentation/blob/main/gitbook/gempyor/broken-reference/README.md) is conducted.&#x20;

These files contain the values of the variables for both the infection and (if included) observational model at each point in time and for each subpopulation. A new file of the same type is produced for each independent simulation and each intervention scenario. Other files report the values of the initial conditions, seeding, and model parameters for each subpopulation and independent simulation (since parameters may be chosen to vary randomly between simulations). When [model inference](https://github.com/HopkinsIDD/flepimop-documentation/blob/main/gitbook/gempyor/broken-reference/README.md) is run, there are also file types reporting the model likelihood (relative to the provided data) and files for each iteration of the inference algorithm.

Within the `model_output` directory in the project's directory, the files will be organized into folders named for the file types: `seir`, `spar`, `snpi`, `hpar`, `hnpi`, `seed`, `init`, or `llik` (see descriptions below). Within each file type folder, files will further be organized by the simulation name (`setup_name` in config), the modifier scenario names - if scenarios exist for either `seir` or `outcome` parameters (specified with `seir_modifiers::scenarios` and `outcome_modifiers::scenarios` in config), and the `run_id` (the date and time of the simulation, by default). For example:

<pre><code><strong>flepimop_sample
</strong>├── model_output
│   ├── seir
│   │   └── sample_2pop
│   │       └── None
│   │           └── 2023.05.24.02/12/48.
│   │               └── 000000001.2023.05.24.02/12/48..seir.csv
│   ├── spar
│   ├── snpi
</code></pre>

The name of each individual file contains (in order) the .... Describe filing naming conventions

Each file is a data table that is by default saved as a [parquet file](https://parquet.apache.org/) (a compressed representation that can be opened and manipulated with minimal memory) but can alternatively be saved as a csv file. See options for specifying output type in [Other Configuration Options.](model-implementation/other-configuration-options.md)

The example files outputs we show were generated with the following configuration file

```
TBA
```

The types and contents of the model output files changes slightly depending on whether the model is run as a forward simulation only, or is run in inference mode, in which parameter values are estimated by comparing the model to data. Output specific to model inference is described in a [separate section](../model-inference/inference-model-output.md).&#x20;

## SEIR (infection model output)

Files in the `seir` folder contain the output of the infection model over time. They contain the value of every variable for each day of the simulation for every subpopulation.

For the example configuration file shown above, the `seir` file is

```
// Some code
```

The meanings of the columns are:

`mc_value_type` – either `prevalence` or `incidence`. Variable values are reported both as a **prevalence** (number of individuals in that state measured instantaneously at the start of the day, equivalent to the meaning of the S, I, or R variable in the differential equations or their stochastic representation) and as **incidence** (total number of individuals who newly entered this state, from all other states, over the course of the 24-hour period comprising that calendar day).

`mc_infection_stage`, `mc_vaccination_status`, etc. – The name of the compartment for which the value is reported, broken down into the infection stage for each state type (eg. vaccination, age).

`mc_name` – The name of the compartment for which the value is reported, which is a concatenation of the compartment status in each state type.

`subpop_1`, `subpop_2`, etc. – one column for each different subpopulation, containing the value of the number of individuals in the described compartment in that subpopulation at the given date. Note that these are named after the nodenames defined by the user in the geodata file.

`date` – The calendar date in the simulation, in YYYY-MM-DD format.

There will be a separate `seir` file output for each slot (independent simulation) and for each iteration of the simulation if [Model Inference](https://github.com/HopkinsIDD/flepimop-documentation/blob/main/gitbook/gempyor/broken-reference/README.md) is conducted.

## SPAR (infection model parameter values)

The files in the `spar` folder contain the parameters that define the transitions in the compartmental model of disease transmission, defined in the `seir::parameters` section of the config.&#x20;

The `value` column gives the numerical values of the parameters defined in the corresponding column `parameter`.

## SNPI (infection model parameter intervention values)

Files in the `snpi` folder contain the time-dependent modifications to the transmission model parameter values (defined in `seir_modifiers` section of the config) for each subpopulation. They contain the interventions that apply to a given subpopulation and the dates within which they apply, and the value of the reduction to the given parameter.

The meanings of the columns are:

`subpop` – The subpopulation to which this intervention parameter applies.

`npi_name` – The name of the intervention parameter.

`start_date` – The start date of this intervention, as defined in the configuration file.

`end_date` – The end date of this intervention, as defined in the configuration file.

`parameter` – The parameter to which the intervention applies, as defined in the configuration file.

`reduction` – The size of the reduction to the parameter either from the config, or fit by inference if that is run.

## HPAR (observation model parameter values)

Files in the `hpar` folder contain the output parameters of the observational model. They contain the values of the probabilities, delays or durations for each outcome in a given subpopulation.

The meanings of the columns are:

`subpop` – Values in this column are the names of the nodes as defined in the `geodata` file given by the user.

`quantity` – The values in this column are the types of parameter values described in the config. The options are `probability`, `delay`, and `duration`. These are the quantities to which there is some parameter defined in the config.

`outcome` – The values here are the outcomes to which this parameter applies. These are names of the outcome compartments defined in the model.

`value` – The values in this column are the parameter values of the quantity that apply to the given subpopulation and outcome.

## HNPI (observation model parameter intervention values)

Files in the `hnpi` folder contain any parameter modifier values that apply to the outcomes model, defined in the `outcome_modifiers` section of the config. They contain the values of the outcome parameter modifiers, and the dates to which they apply in a given subpopulation.

The meanings of the columns are:

`subpop` – The values of this column are the names of the nodes from the `geodata` file.

`npi_name` – The names/labels of the modifier parameters, defined by the user in the config file, which applies to the given node and time period.

`start_date` – The start date of this intervention, as defined in the configuration file.

`end_date` – The end date of this intervention, as defined in the configuration file.

`parameter` – The outcome parameter to which the intervention applies.

`reduction` – The values in this column are the reduction values of the intervention parameters, which apply to the given parameter in a given subpopulation. Note that these are strictly reductions; thus a negative value corresponds to an increase in the parameter, while a positive value corresponds to a decrease in the parameter.

## SEED (model seeding values)

Files in the `seed` folder contain the seeded values of the infection model. They contain the amounts seeded into each variable, the variable they are seeded from, and the time at which the seeding occurs. The user can provide a single seeding file (which will be used across all simulations), or, if multiple simulations are being run the user can provide a separate file for each simulation.&#x20;

The meanings of the columns are:

`subpop` - The values of this column are the names of the nodes from the `geodata` file.

`date` - The values in this column are the dates of seeding.

`amount` - The amount seeded in the given subpopulation from source variables to destination variables, at the given date.&#x20;

`source_infection_stage`, `source_vaccination_status`, etc. -  The name of the compartment **from** which the amount is seeded, broken down into the infection stage for each state type (eg. vaccination, age).

`destination_infection_stage`, `destination_vaccination_status`, etc. - The name of the compartment **into** which the amount is seeded, broken down into the infection stage for each state type (eg. vaccination, age).

`no_perturb` - The values in this column can be either `true` or `false`. If true, then the amount and/or date can be perturbed if running an inference run. Whether the amount or date is perturbed is defined in the config using `perturb_amount` and `perturb_date`.&#x20;

## INIT (model initial conditions)

Files in the `init` folder contain the initial values of the infection model. Either seed or init files will be present, depending on the configuration of the model . These files contain the initial conditions of the infection model at the start date defined in the configuration file. As with seeding, the user can provide a single initial conditions file (which will be used across all simulations), or, if multiple simulations are being run the user can provide a separate file for each simulation.&#x20;

The meanings of the columns are:

`subpop` - The values of this column are the names of the nodes from the `geodata` file.

`mc_infection_stage`, `mc_vaccination_status`, etc. - The name of the compartment for which the value is reported, broken down into the infection stage for each state type (eg. vaccination, age).

`amount` -  The amount initialized seeded in the given subpopulation at the start date defined in the configuration file.&#x20;



###
