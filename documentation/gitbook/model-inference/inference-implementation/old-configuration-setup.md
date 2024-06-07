# (OLD) Configuration setup

**Need to add MultiPeriodModifier and hospitalization interventions**

## Overview

This documentation describes the new YAML configuration file options that may be used when performing inference on model runs. As compared to previous model releases, there are additions to the `seeding` and `interventions` sections, the `outcomes` section replaces the `hospitalization` section, and the `filtering` section added to the file.

Importantly, we now name our pipeline modules: `seeding`, `seir`, `hospitalization` and this becomes relevant to some of the new `filtering` specifications.

Models may be calibrated to any available time series data that is also an outcome of the model (COVID-19 confirmed cases, deaths, hospitalization or ICU admissions, hospital or ICU occupancy, and ventilator use). Our typical usage has calibrated the model to deaths, confirmed cases, or both. We can also perform inference on intervention effectiveness, county-specific baseline R0, and the risk of specific health outcomes.

We describe these options below and present default values in the example configuration sections.

## Modifications to `seeding`

The model can perform inference on the seeding date and initial number of seeding infections in each subpop. An example of this new config section is:

```
seeding:
  method: FolderDraw
  seeding_file_type: seed
  folder_path: importation/minimal/
  lambda_file: data/minimal/seeding.csv
  perturbation_sd: 3
```

<table><thead><tr><th>Config Item</th><th width="214">Required?</th><th>Type/Format</th><th>Description</th></tr></thead><tbody><tr><td>method</td><td>required</td><td>"FolderDraw"</td><td></td></tr><tr><td>seeding_file_type</td><td>required for FolderDraw</td><td>"seed" or "impa"</td><td>indicates which seeding file type the SEIR model will look for, "seed", which is generated from create_seeding.R, or "impa", which refers to importation</td></tr><tr><td>folder_path</td><td>required</td><td>path to folder where importation inference files will be saved</td><td></td></tr><tr><td>lambda_file</td><td>required</td><td>path to seeding file</td><td></td></tr><tr><td>perturbation_sd</td><td>required</td><td>standard deviation for the proposal value of the seeding date, in number of days</td><td></td></tr></tbody></table>

The method for determining the proposal distribution for the seeding amount is hard-coded in the inference package (`R/pkgs/inference/R/functions/perturb_seeding.R`). It is pertubed with a normal distribution where the mean of the distribution 10 times the number of confirmed cases on a given date and the standard deviation is 1.

## Modifications to `interventions`

The model can perform inference on the effectiveness of interventions as long as there is at least some calibration health outcome data that overlaps with the intervention period. For example, if calibrating to deaths, there should be data from time points where it would be possible to observe deaths from infections that occurred during the intervention period (e.g., assuming 10-18 day delay between infection and death, on average).

An example configuration file where inference is performed on scenario planning interventions is as follows:

```
interventions:
  scenarios:
    - Scenario1
  settings:
    local_variance:
      template: SinglePeriodModifierR0
      value:
        distribution: truncnorm
        mean: 0
        sd: .1
        a: -1
        b: 1
      perturbation:
        distribution: truncnorm
        mean: 0
        sd: .1
        a: -1
        b: 1
    stayhome:
      template: SinglePeriodModifierR0
      period_start_date: 2020-04-04
      period_end_date: 2020-04-30
      value:
        distribution: truncnorm
        mean: 0.6
        sd: 0.3
        a: 0
        b: 0.9
      perturbation:
        distribution: truncnorm
        mean: 0
        sd: .1
        a: -1
        b: 1
    Scenario1:
      template: StackedModifier
      scenarios: 
        - local_variance
        - stayhome
```

### `interventions::settings::[setting_name]`

This configuration allows us to infer subpop-level baseline R0 estimates by adding a `local_variance` intervention. The baseline subpop-specific R0 estimate may be calculated as $$R0*(1-local_variance),$$ where R0 is the baseline simulation R0 value, and local\_variance is an estimated subpop-specific value.

Interventions may be specified in the same way as before, or with an added `perturbation` section that indicates that inference should be performed on a given intervention's effectiveness. As previously, interventions with perturbations may be specified for all modeled locations or for explicit `subpop.` In this setup, both the prior distribution and the range of the support of the final inferred value are specified by the `value` section. In the configuration above, the inference algorithm will search 0 to 0.9 for all subpops to estimate the effectiveness of the `stayhome` intervention period. The prior distribution on intervention effectiveness follows a truncated normal distribution with a mean of 0.6 and a standard deviation of 0.3. The `perturbation` section specifies the perturbation/step size between the previously-accepted values and the next proposal value.

<table><thead><tr><th width="216">Item</th><th>Required?</th><th>Type/Format</th></tr></thead><tbody><tr><td>template</td><td>Required</td><td>"SinglePeriodModifierR0" or "StackedModifier"</td></tr><tr><td>period_start_date</td><td>optional for SinglePeriodModifierR0</td><td>date between global <code>start_date</code> and <code>end_date</code>; default is global <code>start_date</code></td></tr><tr><td>period_end_date</td><td>optional for SinglePeriodModifierR0</td><td>date between global <code>start_date</code> and <code>end_date</code>; default is global <code>end_date</code></td></tr><tr><td>value</td><td>required for SinglePeriodModifierR0</td><td>specifies both the prior distribution and range of support for the final inferred values</td></tr><tr><td>perturbation</td><td>optional for SinglePeriodModifierR0</td><td>this option indicates whether inference will be performed on this setting and how the proposal value will be identified from the last accepted value</td></tr><tr><td>subpop</td><td>optional for SinglePeriodModifierR0</td><td>list of subpops, which must be in geodata</td></tr></tbody></table>

## New `outcomes` section

This section is now structured more like the `interventions` section of the config, in that it has scenarios and settings. We envision that separate scenarios will be specified for each IFR assumption.

```
outcomes:
  method: delayframe
  param_from_file: TRUE
  param_subpop_file: "usa-subpop-params-output.parquet" ## ../../Outcomes/data/usa-subpop-params-output.parquet
  scenarios:
    - med
  settings:
    med:
      incidH:
        source: incidI
        probability:
          value:
            distribution: fixed
            value: .035
        delay:
          value:
            distribution: fixed
            value: 7
        duration:
          value:
            distribution: fixed
            value: 7
          name: hosp_curr
      incidD:
        source: incidI
        probability:
          value:
            distribution: fixed
            value: .01
        delay:
          value:
            distribution: fixed
            value: 20
      incidICU:
        source: incidH
        probability: 
          value:
            distribution: fixed
            value: 0.167
        delay:
          value:
            distribution: fixed
            value: 3
        duration:
          value:
            distribution: fixed
            value: 8
      incidVent:
        source: incidICU
        probability: 
          value:
            distribution: fixed
            value: 0.463
        delay:
          value:
            distribution: fixed
            value: 1
        duration:
          value:
            distribution: fixed
            value: 7
      incidC:
        source: incidI
        probability:
          value:
            distribution: truncnorm
            mean: .1
            sd: .1
            a: 0
            b: 10
          perturbation:
            distribution: truncnorm
            mean: 0
            sd: .1
            a: -1
            b: 1
        delay:
          value:
            distribution: fixed
            value: 7
```

| Item                | Required?    | Type/Format                                                                                                                                                    |
| ------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| method              | **required** | "delayframe"                                                                                                                                                   |
| param\_from\_file   | required     | if TRUE, will look for param\_subpop\_file                                                                                                                     |
| param\_subpop\_file | optional     | path to subpop-params parquet file, which indicates location specific risk values. Values in this file will override values in the config if there is overlap. |
| scenarios           | required     | user-defined scenario name                                                                                                                                     |
| settings            | required     | See details below                                                                                                                                              |

### `outcomes::settings::[setting_name]`

The settings for each scenario correspond to a set of different health outcome risks, most often just differences in the probability of death given infection (Pr(incidD|incidI)) and the probability of hospitalization given infection (Pr(incidH|incidI)). Each health outcome risk is referenced in relation to the outcome indicated in `source.` For example, the probability and delay in becoming a confirmed case (incidC) is most likely to be indexed off of the number and timing of infection (incidI).

Importantly, we note that incidI is automatically defined from the SEIR transmission model outputs, while the other compartment sources must be defined in the config before they are used.

Users must specific two metrics for each health outcome, probability and delay, while a duration is optional (e.g., duration of time spent in the hospital). It is also optional to specify a perturbation section (similar to perturbations specified in the NPI section) for a given health outcome and metric. If you want to perform inference (i.e., if `perturbation` is specified) on a given metric, that metric must be specified as a distribution (i.e., not `fixed`) and the range of support for the distribution represents the range of parameter space explored in the inference.

| Item                      | Required?    | Type/Format                                                                                                        |
| ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------ |
| (health outcome metric)   | **required** | "incidH", "incidD", "incidICU", "incidVent", "incidC", corresponding to variable names                             |
| source                    | required     | name of health outcome metric that is used as the reference point                                                  |
| probability               | required     | health outcome risk                                                                                                |
| probability::value        | required     | specifies whether the value is fixed or distributional and the parameters specific to that metric and distribution |
| probability::perturbation | optional     | inference settings for the probability metric                                                                      |
| delay                     | required     | time delay between `source` and the specified health outcome                                                       |
| delay::value              | required     | specifies whether the value is fixed or distributional and the parameters specific to that metric and distribution |
| delay::perturbation       | optional     | inference settings for the time delay metric (coming soon)                                                         |
| duration                  | optional     | duration that health outcome status endures                                                                        |
| duration::value           | required     | specifies whether the value is fixed or distributional and the parameters specific to that metric and distribution |
| duration::perturbation    | optional     | inference settings for the duration metric (coming soon)                                                           |

## New `filtering` section

This section configures the settings for the inference algorithm. The below example shows the settings for some typical default settings, where the model is calibrated to the weekly incident deaths and weekly incident confirmed cases for each subpop. Statistics, hierarchical\_stats\_geo, and priors each have scenario names (e.g., `sum_deaths,` `local_var_hierarchy,` and `local_var_prior,` respectively).

```
filtering:
  simulations_per_slot: 350
  do_filtering: TRUE
  data_path: data/observed_data.csv
  likelihood_directory: importation/likelihood/
  statistics:
    sum_deaths:
      name: sum_deaths
      aggregator: sum ## function applied over the period
      period: "1 weeks"
      sim_var: incidD
      data_var: death_incid
      remove_na: TRUE
      add_one: FALSE
      likelihood:
        dist: sqrtnorm
        param: [.1]
    sum_confirmed:
      name: sum_confirmed
      aggregator: sum
      period: "1 weeks"
      sim_var: incidC
      data_var: confirmed_incid
      remove_na: TRUE
      add_one: FALSE
      likelihood:
        dist: sqrtnorm
        param: [.2]
  hierarchical_stats_geo:
    local_var_hierarchy:
      name: local_variance
      module: seir
      geo_group_col: USPS
      transform: none
    local_conf:
      name: probability_incidI_incidC
      module: hospitalization
      geo_group_col: USPS
      transform: logit
  priors:
    local_var_prior:
      name: local_variance
      module: seir
      likelihood:
        dist: normal
        param:
        - 0
        - 1
```

### `filtering` settings

With inference model runs, the number of simulations `nsimulations` refers to the number of final model simulations that will be produced. The `filtering$simulations_per_slot` setting refers to the number of iterative simulations that will be run in order to produce a single final simulation (i.e., number of simulations in a single MCMC chain).

| Item                     | Required?    | Type/Format                                                                                                                                   |
| ------------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| simulations\_per\_slot   | **required** | number of iterations in a single MCMC inference chain                                                                                         |
| do\_filtering            | required     | TRUE if inference should be performed                                                                                                         |
| data\_path               | required     | file path where observed data are saved                                                                                                       |
| likelihood\_directory    | required     | folder path where likelihood evaluations will be stored as the inference algorithm runs                                                       |
| statistics               | required     | specifies which data will be used to calibrate the model. see `filtering::statistics` for details                                             |
| hierarchical\_stats\_geo | optional     | specifies whether a hierarchical structure should be applied to any inferred parameters. See `filtering::hierarchical_stats_geo` for details. |
| priors                   | optional     | specifies prior distributions on inferred parameters. See `filtering::priors` for details                                                     |

### `filtering::statistics`

The statistics specified here are used to calibrate the model to empirical data. If multiple statistics are specified, this inference is performed jointly and they are weighted in the likelihood according to the number of data points and the variance of the proposal distribution.

| Item              | Required? | Type/Format                                                                                                                                          |
| ----------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| name              | required  | name of statistic, user defined                                                                                                                      |
| aggregator        | required  | function used to aggregate data over the `period`, usually sum or mean                                                                               |
| period            | required  | duration over which data should be aggregated prior to use in the likelihood, may be specified in any number of `days`, `weeks`, `months`            |
| sim\_var          | required  | column name where model data can be found, from the hospitalization outcomes files                                                                   |
| data\_var         | required  | column where data can be found in data\_path file                                                                                                    |
| remove\_na        | required  | logical                                                                                                                                              |
| add\_one          | required  | logical, TRUE if evaluating the log likelihood                                                                                                       |
| likelihood::dist  | required  | distribution of the likelihood                                                                                                                       |
| likelihood::param | required  | parameter value(s) for the likelihood distribution. These differ by distribution so check the code in `inference/R/functions.R/logLikStat` function. |

### `filtering::hierarchical_stats_geo`

The hierarchical settings specified here are used to group the inference of certain parameters together (similar to inference in "hierarchical" or "fixed/group effects" models). For example, users may desire to group all counties in a given state because they are geograhically proximate and impacted by the same statewide policies. The effect should be to make these inferred parameters follow a normal distribution and to observe shrinkage among the variance in these grouped estimates.

| Item            | Required? | Type/Format                                                                                                                                                         |
| --------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| scenario name   | required  | name of hierarchical scenario, user defined                                                                                                                         |
| name            | required  | name of the estimated parameter that will be grouped (e.g., the NPI scenario name or a standardized, combined health outcome name like `probability_incidI_incidC`) |
| module          | required  | name of the module where this parameter is estimated (important for finding the appropriate files)                                                                  |
| geo\_group\_col | required  | geodata column name that should be used to group parameter estimation                                                                                               |
| transform       | required  | type of transform that should be applied to the likelihood: "none" or "logit"                                                                                       |

### `filtering::priors`

It is now possible to specify prior values for inferred parameters. This will have the effect of speeding up model convergence.

<table><thead><tr><th>Item</th><th width="40">Required?</th><th>Type/Format</th></tr></thead><tbody><tr><td>scenario name</td><td>required</td><td>name of prior scenario, user defined</td></tr><tr><td>name</td><td>required</td><td>name of NPI scenario or parameter that will have the prior</td></tr><tr><td>module</td><td>required</td><td>name of the module where this parameter is estimated</td></tr><tr><td>likelihood</td><td>required</td><td>specifies the distribution of the prior</td></tr></tbody></table>
