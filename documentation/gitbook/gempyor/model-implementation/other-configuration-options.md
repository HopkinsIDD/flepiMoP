# Other configuration options

## Command line inputs

_flepiMoP_ allows some input parameters/options to be specified in the command line at the time of model submission, in addition to or instead of in the configuration file. This can be helpful for users who want to quickly run different versions of the model – typically a different number of simulations or a different intervention scenario from among all those specified in the config – without having to edit or create a new configuration file every time. In addition, some arguments can only be specified via the command line.

In addition to the configuration file and the command line, the inputs described below can also be specified as environmental variables.

In all cases, command line arguments override configuration file entries which override environmental variables. The order of command line arguments does not matter.

Details on how to run the model, including how to add command line arguments or environmental variables, are in the section [How to Run](broken-reference).

### Command-line only inputs

<table><thead><tr><th width="131">Argument</th><th width="139">Env. Variable</th><th width="113">Value type</th><th width="238">Description</th><th width="102">Required?</th><th>Default</th></tr></thead><tbody><tr><td><code>-c</code> or <code>--config</code></td><td><code>CONFIG_PATH</code></td><td>file path</td><td>Name of configuration file. Must be located in the current working directory, or else relative or absolute file path must be provided.</td><td>Yes</td><td>NA</td></tr><tr><td><code>-i</code> or <code>--first_sim_index</code></td><td><code>FIRST_SIM_INDEX</code></td><td>integar <span class="math">\geq</span>1</td><td>The index of the first simulation</td><td>No</td><td>1</td></tr><tr><td><code>-j</code> or <code>--jobs</code></td><td><code>FLEPI_NJOBS</code></td><td>integar <span class="math">\geq</span>1</td><td>Number of parallel processors used to run the simulation. If there are more slots that jobs, slots will be divided up between processors and run in series on each.</td><td>No</td><td>Number of processors on the computer used to run the simulation</td></tr><tr><td><code>--interactive</code>or <code>--batch</code></td><td>NA</td><td>Choose either option</td><td>Run simulation in interactive or batch mode</td><td>No</td><td><code>batch</code></td></tr><tr><td><code>--write-csv</code> or <code>--no-write-csv</code></td><td>NA</td><td>Choose either option</td><td>Whether model output will be saved as .csv files</td><td>No</td><td><code>no_write_csv</code></td></tr><tr><td><code>--write-parquet</code> or <code>--no-write-parquet</code></td><td>NA</td><td>Choose either option</td><td>Whether model output will be saved as .parquet files (a compressed representation that can be opened and manipulated with minimal memory. May be required for large simulations). Read more about <a href="https://parquet.apache.org/">parquet files</a>.</td><td>No</td><td><code>write_parquet</code></td></tr></tbody></table>

### Command-line versions of configuration file inputs

<table><thead><tr><th width="131">Argument</th><th width="126">Config item</th><th width="130">Env. Variable</th><th width="117">Value type</th><th width="331">Description</th><th width="108">Required?</th><th>Default</th></tr></thead><tbody><tr><td><code>-s</code> or <code>--npi_scenario</code></td><td><code>interventions: scenarios</code></td><td><code>FLEPI_NPI_SCENARIOS</code></td><td>list of strings</td><td>Names of the intervention scenarios described in the config file that will be run. Must be a subset of scenarios defined.</td><td>No</td><td>All scenarios described in config</td></tr><tr><td><code>-n</code> or <code>--nslots</code></td><td><code>nslots</code></td><td><code>FLEPI_NUM_SLOTS</code></td><td>integar <span class="math">\geq</span>1</td><td>Number of independent simulations of the model to be run</td><td>No</td><td>Config value</td></tr><tr><td><code>--stochastic</code> or <code>--non-stochastic</code></td><td><code>seir: integration: method</code></td><td><code>FLEPI_STOCHASTIC_RUN</code></td><td>choose either option</td><td>Whether the model will be run stochastically or non-stochastically (deterministic numerical integration of equations using the RK4 algorithm)</td><td>No</td><td>Config value</td></tr><tr><td><code>--in-id</code></td><td></td><td><code>FLEPI_RUN_INDEX</code></td><td>string</td><td>Unique ID given to the model runs. If the same config is run multiple times, you can avoid the output being overwritten by using unique model run IDs.</td><td>No</td><td>Constructed from current date and time as YYYY.MM.DD.HH/MM/SS</td></tr><tr><td><code>--out-id</code></td><td></td><td><code>FLEPI_RUN_INDEX</code></td><td>string</td><td>Unique ID given to the model runs. If the same config is run multiple times, you can avoid the output being overwritten by using unique model run IDs.</td><td>No</td><td>Constructed from current date and time as YYYY.MM.DD.HH/MM/SS</td></tr></tbody></table>

#### Example

As an example, consider running the following configuration file

```
name: sir
setup_name: minimal
start_date: 2020-01-31
end_date: 2020-05-31
data_path: data
nslots: 1

subpop_setup:
  geodata: geodata_sample_1pop.csv
  mobility: mobility_sample_1pop.csv
  popnodes: population
  nodenames: name

seeding:
  method: FromFile
  seeding_file: data/seeding_1pop.csv

compartments:
  infection_stage: ["S", "I", "R"]

seir:
  integration:
    method: stochastic
    dt: 1 / 10
  parameters:
    gamma:
      value:
        distribution: fixed
        value: 1 / 5
    Ro:
      value:
        distribution: uniform
        low: 2
        high: 3
  transitions:
    - source: ["S"]
      destination: ["I"]
      rate: ["Ro * gamma"]
      proportional_to: [["S"],["I"]]
      proportion_exponent: ["1","1"]
    - source: ["I"]
      destination: ["R"]
      rate: ["gamma"]
      proportional_to: ["I"]
      proportion_exponent: ["1"]

interventions:
  scenarios:
    - None
    - Lockdown
  modifiers:
    None:
      method: SinglePeriodModifier
      parameter: r0
      period_start_date: 2020-04-01
      period_end_date: 2020-05-15
      value:
        distribution: fixed
        value: 0
        settings:
    Lockdown:
      method: SinglePeriodModifier
      parameter: r0
      period_start_date: 2020-04-01
      period_end_date: 2020-05-15
      value:
        distribution: fixed
        value: 0.7
```

To run this model directly in Python (it can alternatively be run from R, for all details see section [How to Run](broken-reference)), we could use the command line entry

```
> gempyor-seir -c sir_control.yml
```

Alternatively, to run 100 simulations using only 4 of the available processors on our computer, but only running the "" scenario with a deterministic model, and to save the files as .csv (since the model is relatively simple), we could call the model using the command line entry

```
/> gempyor-seir -c sir_control.yml -n 100 -j 4 -npi_scenario None --non_stochastic --write_csv
```

## Environmental variables

TBA

## US-specific configuration file options

{% hint style="warning" %}
Things below here are very out of date. Put here as place holder but not updated recently.
{% endhint %}

global: smh\_round, setup\_name, disease

spatial\_setup: census\_year, modeled\_states, state\_level

#### For US-specific population structures

For creating US-based population structures using the helper script `build_US_setup.R` which is run before the main model simulation script, the following extra parameters can be specified

<table><thead><tr><th>Config Item</th><th>Required?</th><th width="187">Type/Format</th><th>Description</th></tr></thead><tbody><tr><td>census_year</td><td>optional</td><td>integer (year)</td><td>Determines the year for which census population size data is pulled.</td></tr><tr><td>state_level</td><td>optional</td><td>boolean</td><td>Determines whether county-level population-size data is instead grouped into state-level data (TRUE). Default FALSE</td></tr><tr><td>modeled_states</td><td>optional</td><td>list of location codes</td><td>A vector of locations that will be modeled; others will be ignored</td></tr></tbody></table>

#### Example 2

To simulate an epidemic across all 50 states of the US or a subset of them, users can take advantage of built in machinery to create geodata and mobility files for the US based on the population size and number of daily commuting trips reported in the US Census.

Before running the simulation, the script `build_US_setup.R` can be run to get the required population data files from online census data and filter out only states/territories of interest for the model. More details are provided in the How to Run section.

This example simulates COVID-19 in the New England states, assuming no transmission from other states, using 2019 census data for the population sizes and a pre-created file for estimated interstate commutes during the 2011-2015 period.

```
subpop_setup:
  census_year: 2010
  state_level: TRUE
  geodata: geodata_2019_statelevel.csv
  mobility: mobility_2011-2015_statelevel.csv
  modeled_states:
    - CT
    - MA
    - ME
    - NH
    - RI
    - VT
  
```

`geodata.csv` contains

```
USPS	subpop	population
AL	01000	4876250
AK	02000	737068
AZ	04000	7050299
AR	05000	2999370
CA	06000	39283497
.....
```

`mobility_2011-2015_statelevel.csv` contains

```
ori	dest	amount
01000	02000	198
01000	04000	292
01000	05000	570
01000	06000	1030
01000	08000	328
.....
```

### `importation` section (optional)

This section is optional. It is used by the [covidImportation package](https://github.com/HopkinsIDD/covidImportation) to import global air importation data for seeding infections into the United States.

If you wish to include it, here are the options.

| Config Item                      | Required?    | Type/Format       | Description                                                                             |
| -------------------------------- | ------------ | ----------------- | --------------------------------------------------------------------------------------- |
| census\_api\_key                 | **required** | string            | [get an API key](https://api.census.gov/data/key\_signup.html)                          |
| travel\_dispersion               | **required** | number            | ow dispersed daily travel data is; default = 3.                                         |
| maximum\_destinations            | **required** | integer           | number of airports to limit importation to                                              |
| dest\_type                       | **required** | categorical       | location type                                                                           |
| dest\_country                    | **required** | string (Country)  | ISO3 code for country of importation. Currently only USA is supported                   |
| aggregate\_to                    | **required** | categorical       | location type to aggregate to                                                           |
| cache\_work                      | **required** | boolean           | whether to save case data                                                               |
| update\_case\_data               | **required** | boolean           | deprecated; whether to update the case data or used saved                               |
| draw\_travel\_from\_distribution | **required** | boolean           | whether to add additional stochasticity to travel data; default is FALSE                |
| print\_progress                  | **required** | boolean           | whether to print progress of importation model simulations                              |
| travelers\_threshold             | **required** | integer           | include airports with at least the `travelers_threshold` mean daily number of travelers |
| airport\_cluster\_distance       | **required** | numeric           | cluster airports within `airport_cluster_distance` km                                   |
| param\_list                      | **required** | See section below | see below                                                                               |

### `importation::param_list`

| Config Item                  | Required?    | Type/Format | Description                                                    |
| ---------------------------- | ------------ | ----------- | -------------------------------------------------------------- |
| incub\_mean\_log             | **required** | numeric     | incubation period, log mean                                    |
| incub\_sd\_log               | **required** | numeric     | incubation period, log standard deviation                      |
| inf\_period\_nohosp\_mean    | **required** | numeric     | infectious period, non-hospitalized, mean                      |
| inf\_period\_nohosp\_sd      | **required** | numeric     | infectious period, non-hospitalized, sd                        |
| inf\_period\_hosp\_mean\_log | **required** | numeric     | infectious period, hospitalized, log-normal mean               |
| inf\_period\_hosp\_sd\_log   | **required** | numeric     | infectious period, hospitalized, log-normal sd                 |
| p\_report\_source            | **required** | numeric     | reporting probability, Hubei and elsewhere                     |
| shift\_incid\_days           | **required** | numeric     | mean delay from infection to reporting of cases; default = -10 |
| delta                        | **required** | numeric     | days per estimations period                                    |

```
importation:
  census_api_key: "fakeapikey00000"
  travel_dispersion: 3
  maximum_destinations: Inf
  dest_type: state
  dest_county: USA
  aggregate_to: airport
  cache_work: TRUE
  update_case_data: TRUE
  draw_travel_from_distribution: FALSE
  print_progress: FALSE
  travelers_threshold: 10000
  airport_cluster_distance: 80
  param_list:
    incub_mean_log: log(5.89)
    incub_sd_log: log(1.74)
    inf_period_nohosp_mean: 15
    inf_period_nohosp_sd: 5
    inf_period_hosp_mean_log: 1.23
    inf_period_hosp_sd_log: 0.79
    p_report_source: [0.05, 0.25]
    shift_incid_days: -10
    delta: 1
```

### `report` section

The `report` section is completely optional and provides settings for making an R Markdown report. For an example of a report, see the Supplementary Material of our [preprint](https://www.medrxiv.org/content/10.1101/2020.06.11.20127894v1)

If you wish to include it, here are the options.

| Config Item                         | Required? | Type/Format                                                          | Description                                                                                |
| ----------------------------------- | --------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| data\_settings::pop\_year           |           | integer                                                              |                                                                                            |
| plot\_settings::plot\_intervention  |           | boolean                                                              |                                                                                            |
| formatting::scenario\_labels\_short |           | list of strings; one for each scenario in `interventions::scenarios` |                                                                                            |
| formatting::scenario\_labels        |           | list of strings; one for each scenario in `interventions::scenarios` |                                                                                            |
| formatting::scenario\_colors        |           | list of strings; one for each scenario in `interventions::scenarios` |                                                                                            |
| formatting::pdeath\_labels          |           | list of strings                                                      |                                                                                            |
| formatting::display\_dates          |           | list of dates                                                        |                                                                                            |
| formatting::display\_dates2         | optional  | list of dates                                                        | a 2nd string of display dates that can optionally be supplied to specific report functions |

```
report:
  data_settings:
    pop_year: 2018
  plot_settings:
    plot_intervention: TRUE
  formatting:
    scenario_labels_short: ["UC", "S1"]
    scenario_labels:
      - Uncontrolled
      - Scenario 1
    scenario_colors: ["#D95F02", "#1B9E77"]
    pdeath_labels: ["0.25% IFR", "0.5% IFR", "1% IFR"]
    display_dates: ["2020-04-15", "2020-05-01", "2020-05-15", "2020-06-01", "2020-06-15"]
    display_dates2: ["2020-04-15", "2020-05-15", "2020-06-15"]
```
