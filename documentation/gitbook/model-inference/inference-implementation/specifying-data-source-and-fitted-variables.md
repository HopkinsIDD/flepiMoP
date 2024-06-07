# Specifying data source and fitted variables

## inference settings

iterations\_per\_slot

do\_inference

gt\_data\_path

With inference model runs, the number of simulations `nsimulations` refers to the number of final model simulations that will be produced. The `filtering$simulations_per_slot` setting refers to the number of iterative simulations that will be run in order to produce a single final simulation (i.e., number of simulations in a single MCMC chain).

<table><thead><tr><th width="169">Item</th><th width="98.33333333333331">Required?</th><th width="165">Type/Format</th><th>Description</th></tr></thead><tbody><tr><td>iterations_per_slot</td><td>required</td><td>Integer <span class="math">\geq</span> 1</td><td>Number of iterations in a single MCMC inference chain</td></tr><tr><td>do_inference</td><td>required</td><td>TRUE/FALSE</td><td>TRUE if inference should be performed. If FALSE,  just runs a single run per slot, without perturbing parameters</td></tr><tr><td>gt_data_path</td><td>required</td><td>file path</td><td>Path to files containing "ground truth" data to which model output will be compared</td></tr><tr><td>statistics</td><td>required</td><td>config subsection</td><td>Specifies details of how each model output variable will be compared to data during fitting. See inference::statistics section. </td></tr><tr><td>hierarchical_stats_geo</td><td>optional</td><td>config subsection</td><td>Specifies whether a hierarchical structure should be applied the likelihood function for any of the fitted parameters. See <code>inference::hierarchical_stats_geo</code> for details.</td></tr><tr><td>priors</td><td>optional</td><td>config subsection</td><td>Specifies prior distributions on fitted parameters. See <code>inference::priors</code> for details</td></tr></tbody></table>

### `f`

## inference::statistics options

### required options

name

aggregator

period

sim\_var

data\_var

likelihood

The statistics specified here are used to calibrate the model to empirical data. If multiple statistics are specified, this inference is performed jointly and they are weighted in the likelihood according to the number of data points and the variance of the proposal distribution.

<table><thead><tr><th width="160.33333333333331">Item</th><th width="108">Required?</th><th width="154">Type/Format</th><th>Description</th></tr></thead><tbody><tr><td>name</td><td>required</td><td>string</td><td>name of statistic, user defined</td></tr><tr><td>period</td><td>required</td><td><code>days</code>, <code>weeks</code>, or <code>months</code></td><td>Duration of time over which data and model output should be aggregated before being used in the likelihood. If <code>weeks</code>, <code>epiweeks</code> are used</td></tr><tr><td>aggregator</td><td>required</td><td>string, name of any R function</td><td>Function used to aggregate data over the<code>period</code>, usually <code>sum</code> or <code>mean</code></td></tr><tr><td>sim_var</td><td>required</td><td>string</td><td>Name of the outcome variable -  as defined in<code>outcomes</code> section of the config - that will be compared to data when calculating the likelihood. This will also be the column name of this variable in the <code>hosp</code> files in the <code>model_output</code> directory</td></tr><tr><td>data_var</td><td>required</td><td>string</td><td><p>Name of the data variable that will be compared to the model output variable when calculating the likelihood. This should be the name of a column in the </p><p>file specified in <code>inference::gt_data_path</code> config option </p></td></tr><tr><td>remove_na</td><td>required</td><td>logical</td><td>if TRUE<br>if FALSE</td></tr><tr><td>add_one</td><td>required</td><td>logical</td><td>if TRUE<br>if FALSE<br>Will be overwritten to TRUE if the likelihood distribution is chosen to be log</td></tr><tr><td>likelihood::dist</td><td>required</td><td></td><td>Distribution of the likelihood</td></tr><tr><td>likelihood::param</td><td>required</td><td></td><td>parameter value(s) for the likelihood distribution. These differ by distribution so check the code in <code>inference/R/functions.R/logLikStat</code> function.</td></tr></tbody></table>

### `f`

### optional options ?

remove\_na

add\_one

gt\_start\_date

gt\_end\_date

Optional sections

### `inference::hierarchical_stats_geo`

The hierarchical settings specified here are used to group the inference of certain parameters together (similar to inference in "hierarchical" or "fixed/group effects" models). For example, users may desire to group all counties in a given state because they are geograhically proximate and impacted by the same statewide policies. The effect should be to make these inferred parameters follow a normal distribution and to observe shrinkage among the variance in these grouped estimates.

| Item            | Required? | Type/Format                                                                                                                                                         |
| --------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| scenario name   | required  | name of hierarchical scenario, user defined                                                                                                                         |
| name            | required  | name of the estimated parameter that will be grouped (e.g., the NPI scenario name or a standardized, combined health outcome name like `probability_incidI_incidC`) |
| module          | required  | name of the module where this parameter is estimated (important for finding the appropriate files)                                                                  |
| geo\_group\_col | required  | geodata column name that should be used to group parameter estimation                                                                                               |
| transform       | required  | type of transform that should be applied to the likelihood: "none" or "logit"                                                                                       |

### `inference::priors`

It is now possible to specify prior values for inferred parameters. This will have the effect of speeding up model convergence.

<table><thead><tr><th width="164">Item</th><th width="40">Required?</th><th>Type/Format</th></tr></thead><tbody><tr><td>scenario name</td><td>required</td><td>name of prior scenario, user defined</td></tr><tr><td>name</td><td>required</td><td>name of NPI scenario or parameter that will have the prior</td></tr><tr><td>module</td><td>required</td><td>name of the module where this parameter is estimated</td></tr><tr><td>likelihood</td><td>required</td><td>specifies the distribution of the prior</td></tr></tbody></table>

## Ground truth data

name

module

geo\_group\_col

transform

inference:::priors



inference::



