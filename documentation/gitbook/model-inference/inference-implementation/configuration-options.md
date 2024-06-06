# (OLD) Configuration options

## `filtering` section

The `filtering` section configures the settings for the inference algorithm. The below example shows the settings for some typical default settings, where the model is calibrated to the weekly incident deaths and weekly incident confirmed cases for each subpop. Statistics, hierarchical\_stats\_geo, and priors each have scenario names (e.g., `sum_deaths,` `local_var_hierarchy,` and `local_var_prior,` respectively).

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

<table><thead><tr><th>Item</th><th width="104.33333333333331">Required?</th><th>Type/Format</th></tr></thead><tbody><tr><td>simulations_per_slot</td><td><strong>required</strong></td><td>number of iterations in a single MCMC inference chain</td></tr><tr><td>do_filtering</td><td>required</td><td>TRUE if inference should be performed</td></tr><tr><td>data_path</td><td>required</td><td>file path where observed data are saved</td></tr><tr><td>likelihood_directory</td><td>required</td><td>folder path where likelihood evaluations will be stored as the inference algorithm runs</td></tr><tr><td>statistics</td><td>required</td><td>specifies which data will be used to calibrate the model. see <code>filtering::statistics</code> for details</td></tr><tr><td>hierarchical_stats_geo</td><td>optional</td><td>specifies whether a hierarchical structure should be applied to any inferred parameters. See <code>filtering::hierarchical_stats_geo</code> for details.</td></tr><tr><td>priors</td><td>optional</td><td>specifies prior distributions on inferred parameters. See <code>filtering::priors</code> for details</td></tr></tbody></table>

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

<table><thead><tr><th width="164">Item</th><th width="40">Required?</th><th>Type/Format</th></tr></thead><tbody><tr><td>scenario name</td><td>required</td><td>name of prior scenario, user defined</td></tr><tr><td>name</td><td>required</td><td>name of NPI scenario or parameter that will have the prior</td></tr><tr><td>module</td><td>required</td><td>name of the module where this parameter is estimated</td></tr><tr><td>likelihood</td><td>required</td><td>specifies the distribution of the prior</td></tr></tbody></table>

## Ground truth data

## Likelihood function

## Fitting parameters

## Ground truth data
