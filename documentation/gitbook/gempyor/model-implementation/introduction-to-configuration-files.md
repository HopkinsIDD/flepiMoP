# flepiMoP's configuration file

## About configuration files

_flepiMop_ is set up so that all parameters and other options for running the pipeline can be specified in a single "configuration" file (aka "config"). Users do not need to edit any other code files, or even be aware of their contents, to create and run complex model scenarios. Configuration files also provide a convenient record of model options and promote reproducibility of model results.

We use the `YAML` language syntax to write config files, which are typically named something like `config.yml`. The file has simple plain text contents and follows a tabbed outline structure. When config files are read by the model code, a data structure encoding the model options is created.

Comments can be added to the config file by starting with the hash key (`#`) then a space. Comments can start anywhere on a line and continue until the end, but if they run over to a new line, a new # must be used at the start of the new line.

## Example

_(give a simple configuration for a toy model with two subpopulations, SEIR, single "cases" outcome, single seeded infection, single NPI that starts after some time? this page is currently under development, please see our_ [_example repo_](https://github.com/HopkinsIDD/flepimop\_sample) _for some simple configurations) ;

When referring to config items (individual parameters), we use their full position in the outline. For example, in the sample config file above, we denote

```
subpop_setup:
  ...
  geodata: minimal
```

as `subpop_setup::geodata` having a value of `minimal`

## Notation

Parameters and other options specified in the configuration files can take on a variety of types of values, using the following notations:

* **dates** are specified as \[year]-\[month]-\[day]. (e.g., 2020-01-31)
* **boolean** values are either "TRUE" or "FALSE"
* **files** names are strings
* **probability** is a float between 0 and 1
* **distribution** is a probability distribution from which a random value for the parameter is drawn each time a new simulation is run (or chain, if doing inference). See [here](distributions.md) for the require schema.

## Configuration files sections

### Global header

{% hint style="success" %}
Required section
{% endhint %}

These global configuration options typically sit at the top of the configuration file.

<table><thead><tr><th width="137">Item</th><th width="177">Required?</th><th width="137">Type/Format</th><th>Description</th></tr></thead><tbody><tr><td>name</td><td><strong>required</strong></td><td>string</td><td>Name of this configuration. Will be used in file names created to store model output. </td></tr><tr><td>start_date</td><td><strong>required</strong></td><td>date</td><td>model simulation start date</td></tr><tr><td>end_date</td><td><strong>required</strong></td><td>date</td><td>model simulation end date</td></tr><tr><td>start_date_groundtruth</td><td><strong>optional for non-inference runs, required for inference runs</strong></td><td>date</td><td>start date for comparing model to data</td></tr><tr><td>end_date_groundtruth</td><td><strong>optional for non-inference runs, required for inference runs</strong></td><td>date</td><td>end date for comparing model to data</td></tr><tr><td>nslots</td><td><strong>optional (can also be defined by an environmental variable)</strong></td><td>int</td><td>number of independent simulations to run </td></tr><tr><td>setup_name</td><td>optional</td><td>string</td><td>setup name used to describe the run, used in setting up file names</td></tr><tr><td>model_output_dirname</td><td>optional</td><td>folder path </td><td>path to folder where all the outputs created by the model are stored, if not specified, default is <code>model_output</code></td></tr></tbody></table>

For example, for a configuration file to simulate the spread of COVID-19 in the US during 2020 and compare to data from March 1 onwards, with 1000 independent simulations, the header of the config might read:

```
name: USA_covid19_2020
model_output_dirname: model_output
start_date: 2020-01-01
end_date: 2020-12-31
start_date_groundtruth: 2020-03-01
end_date_groundtruth: 2020-12-31
nslots: 1000
```

### `subpop_setup` section

{% hint style="success" %}
Required section
{% endhint %}

This section specifies the population structure on which the model will be simulated, including the names and sizes of each subpopulation and the connectivity between them. More details [here](specifying-population-structure.md).

### `compartments` section

{% hint style="success" %}
Required section
{% endhint %}

This section is where users can specify the variables (infection states) that will be tracked in the infectious disease transmission model. More details can be found [here](compartmental-model-structure.md). The other details of the model are specified in the `seir` section, including transitions between these compartments (`seir::transitions`), the names of the parameters governing the transitions (`seir::parameters`), and the numerical method used to simulate the equations over time (`seir::integration`). The initial conditions of the model can be specified in the `initial_conditions` section, and any other inputs into the model from external populations or instantaneous transitions between states that occur at later times can be specified in the `seeding` section. ;

### `seir` section

{% hint style="success" %}
Required section
{% endhint %}

This section is where users can specify the details of the infectious disease transmission model they wish to simulate (e.g., SEIR). This model describes the allowed transitions (`seir::transitions`) between the compartments that were specified in the `compartments` section, the values of the parameters involved in these transitions (`seir::parameters`), and the numerical method used to simulate the equations over time (`seir::integration`).  More details [here](compartmental-model-structure.md).  The initial conditions of the model can be specified in the separate `initial_conditions` section, and any other inputs into the model from external populations or instantaneous transitions between states that occur at later times can be specified in the `seeding` section. ;

### `initial_conditions` section

{% hint style="info" %}
Optional section
{% endhint %}

This section is used to specify the initial conditions of the model, which define how individuals are distributed between the model compartments at the time the model simulation begins. Importantly, the initial conditions specify the time and location where infection is first introduced. If this section is omitted, default values are used. If users want to add infections to the population at later times, or add or remove individuals from compartments separately from the model rules, they can do so via the related `seeding` section. More details [here](specifying-initial-conditions.md) ;

### `seeding` section

{% hint style="info" %}
Optional section
{% endhint %}

This section is used to specify how individuals are instantaneously "seeded" from one compartment to another, where they then continue to be governed by the model equations. For example, this seeding could be used to represent importations of infected individuals from an outside population, mutation events that create new strains, or vaccinations that alter disease susceptibility. Seeding events can occur at any time in the simulation. The seeding section specifies the numeric values added to or removed from any compartment of the model. More details [here](specifying-initial-conditions.md) ;

### `outcomes` section

{% hint style="info" %}
Optional section
{% endhint %}

This section is where users can define new variables representing the observed quantities and how they are related to the underlying state variables in the model (e.g., the fraction of infections that are detected as cases). More details [here](outcomes-for-compartments.md) ;

### `interventions` section

{% hint style="success" %}
Required section
{% endhint %}

This section is where users can specify time-varying changes to parameters governing either the infectious disease model or the observational model. More details [here](intervention-templates.md) ;

### `inference` section

{% hint style="info" %}
Optional section
{% endhint %}

This section is where users can specify the details of how the model is fit to data, including what data streams they will be included and which outcome variables they represent and the likelihood functions describing the probability of the data given the model. More details [here](../../model-inference/inference-implementation/). ;
