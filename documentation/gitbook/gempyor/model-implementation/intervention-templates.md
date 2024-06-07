---
description: >-
  This section describes how to specify modifications to any of the parameters
  of the transmission model or observational model during certain time periods.
---

# Specifying time-varying parameter modifications

**Modifiers** are a powerful feature in _flepiMoP_ to enable users to modify any of the parameters being specified in the model during particular time periods. They can be used, for example, to mirror public health control interventions, like non-pharmaceutical interventions (NPIs) or increased access to diagnosis or care, or annual seasonal variations in disease parameters. Modifiers can act on any of the transmission model parameters or observation model parameters.&#x20;

In the `seir_modifiers` and `outcome_modifiers` sections of the configuration file the user can specify several possible types of modifiers which will then be implemented in the model. Each modifier changes a parameter during one or multiple time periods and for one or multiple specified subpopulations.

We currently support the following intervention types. Each of these is described in detail below:

* `"SinglePeriodModifier"` – Modifies a parameter during a single time period
* `"MultiPeriodModifier"` – Modifies a parameter by the same amount during a multiple time periods
* `"ModifierModifier"` – Modifies another intervention during a single time period
* `"StackedModifier"` – Combines two or more interventions additively or multiplicatively, and is used to be able to turn on and off groups of interventions easily for different runs.&#x20;

{% hint style="info" %}
Note that if you want a parameter to vary continuously over time (for example, a daily transmission rate that is influenced by temperature and humidity), then it is easier to do this by using a "timeseries" parameter value than by combining many separate modifiers. Timeseries parameter values are described in the [seir::parameters](compartmental-model-structure.md#specifying-parameters-values-from-distributions) section. Timeseries parameters for[`outcomes`](outcomes-for-compartments.md) parameters (e.g., a testing rate that fluctuates rapidly due to test availability) are in development but not currently available.&#x20;
{% endhint %}

Within _flepiMoP_, modifiers can be run as "scenarios". With scenarios, we can use the same configuration file to run multiple versions of the model where only the modifiers applied differ.

The `modifiers` section contains two sub-sections: `modifiers::scenarios`, which lists the name of the modifiers that will run in each separate scenario, and `modifiers::modifiers`, where the details of each modifier are specified (e.g.,  the parameter it acts on, the time it is active, and the subpopulation it is applied to). An example is outlined below

```
seir_modifiers:
  scenarios:
    -NameOfIntervention1
    -NameofIntervention2
  modifiers:
    NameOfIntervention1:
      ...
    NameOfIntervention2:
      ...
```

In this example, each scenario runs a single intervention, but more complicated examples are possible. &#x20;

The major benefit of specifying both "scenarios" and "modifiers" is that the user can use `"StackedModifier"` option to combine other modifiers in different ways, and then run either the individual or combined modifiers as scenarios. This way, each scenario may consist of one or more individual parameter modifications, and each modification may be part of multiple scenarios. This provides a shorthand to quickly consider multiple different versions of a model that have different combinations of parameter modifications occurring. For example, during an outbreak we could evaluate the impact of school closures, case isolation, and masking, or any one or two of these three measures. An example of a configuration file combining modifiers to create new scenarios is given below

```
seir_modifiers:
  scenarios:
    -SchoolClosures
    -AllNPIs
  modifiers:
    SchoolClosures:
      method:SinglePeriodModifier
      ...
    CaseIsolation:
      method:SinglePeriodModifier
      ...
    Masking:
      method:SinglePeriodModifier
      ....
    AllNPIs
      method: StackedModifier
      modifiers: ["SchoolClosures","CaseIsolation","Masking"]
```

{% hint style="info" %}
The `seir_modifiers::scenarios` and`outcome_modifiers::scenarios` sections are optional. If the `scenarios`section is **not** included, the model will run with all of the modifiers turned "on".&#x20;
{% endhint %}

{% hint style="info" %}
If the`scenarios`section is included for either `seir` or `outcomes`, then each time a configuration file is run, the user much specify which modifier scenarios will be run. If not specified, the model will be run one time for each combination of `seir` and `outcome` scenario.&#x20;
{% endhint %}

#### Example

\[Give a configuration file that tries to use all the possible option available. Based on simple SIR model with parameters `beta` and `gamma` in 2 subpopulations. Maybe a SinglePeriodModifier on `beta` for a lockdown and `gamma` for isolation, one having a fixed value and one from a distribution, MultiPeriodModifier for school year in different places, ModifierModifer for ..., StackedModifier for .... ]

## **`modifiers::scenarios`**

A optional list consisting of a subset of the modifiers that are described in `modifiers::settings`, each of which will be run as a separate scenario. For example

```
seir_modifiers:
  scenarios:
    -SchoolClosures
    -AllNPIs
```

or

```
outcome_modifiers
  scenarios:
    -BaselineTesting
    -TestShortage
```

## **`modifiers::settings`**

A formatted list consisting of the description of each modifier, including its name, the parameter it acts on, the duration and amount of the change to that parameter, and the subset of subpopulations in which the parameter modification takes place. The list items are summarized in the table below and detailed in the sections below.

<table><thead><tr><th>Config item</th><th width="164">Required</th><th>Type/format</th><th>Description</th></tr></thead><tbody><tr><td><code>method</code></td><td>required</td><td>string</td><td>one of <code>SinglePeriodModifier</code>, <code>MultiPeriodModifier</code>, <code>ModifierModifier</code>, or <code>StackedModifier</code></td></tr><tr><td><code>parameter</code></td><td>required</td><td>string</td><td>The parameter on which the modification is acting. Must be a parameter defined in <code>seir::parameters</code> or <code>outcomes</code></td></tr><tr><td><code>period_start_date</code> or <code>periods::start_date</code></td><td>required</td><td>numeric, YYYY-MM-DD</td><td>The date when the modification starts. Notation depends on value of <code>method.</code></td></tr><tr><td><code>period_end_date</code> or <code>periods::end_date</code></td><td>required</td><td>numeric, YYYY-MM-DD</td><td>The date when the modification ends. Notation depends on value of <code>method.</code></td></tr><tr><td><code>subpop</code></td><td>required</td><td>String, or list of strings</td><td>The subpopulations to which the modifications will be applied, or <code>"all"</code> . Subpopulations must appear in the <code>geodata</code> file. </td></tr><tr><td><code>value</code></td><td>required</td><td>Distribution, or single value</td><td>The relative amount by which a modification <em>reduces</em> the value of a parameter. </td></tr><tr><td><code>subpop_groups</code></td><td>optional</td><td>string or a list of lists of strings</td><td>A list of lists defining groupings of subpopulations, which defines how modification values should be shared between them, or <code>'all'</code> in which case all subpopulations are put into one group with identical modification values. By default, if parameters are chosen randomly from a distribution or fit based on data, they can have unique values in each subpopulation. </td></tr><tr><td><code>baseline_scenario</code></td><td>Used only for <code>ModifierModifier</code></td><td>String</td><td>Name of the original modification which will be further modified</td></tr><tr><td><code>modifiers</code></td><td>Used only for <code>StackedModifier</code></td><td>List of strings</td><td>List of modifier names to be grouped  into the new combined modifier/scenario name</td></tr></tbody></table>

### SinglePeriodModifier

`SinglePeriodModifier` interventions enable the user to specify a multiplicative reduction to a `parameter` of interest. It take a `parameter`, and reduces it's value by `value` (new = (1-`value`) \* old) for the subpopulations listed in`subpop` during the time interval \[`period_start_date`, `period_end_date`]

For example, if you would like to create an SEIR modifier called `lockdown` that reduces transmission by 70% in the state of California and the District of Columbia between two dates, you could specify this with a SinglePeriodModifier, as in the example below

#### Example

```
seir_modifiers:
  modifiers:
    lockdown: 
      method: SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-03-15
      period_end_date: 2020-05-01
      subpop: ['06000', '11000']
      value: 0.7
```

Or to create an outcome variable modifier called enhanced\_testing during which the case detection rate doubles&#x20;

```
outcome_modifiers:
  modifiers:
    enhanced_testing: 
      method: SinglePeriodModifier
      parameter: incidC::probability
      period_start_date: 2020-03-15
      period_end_date: 2020-05-01
      subpop: ['06000', '11000']
      value: -1.0
```

#### Configuration options

`method`: `SinglePeriodModifier`

`parameter`: The name of the parameter that will be modified. This could be a parameter defined for the transmission model in [`seir::parameters`](compartmental-model-structure.md) or for the observational model in [`outcomes`](outcomes-for-compartments.md). If the parameter is used in multiple transitions in the model then all those transitions will be modified by this amount.&#x20;

`period_start_date`: The date when the modification starts, in YYYY-MM-DD format. The modification will only reduce the value of the parameter after (inclusive of) this date.

`period_end_date`: The date when the modification ends, in YYYY-MM-DD format. The modification will only reduce the value of the parameter before (inclusive of) this date.

`subpop:`A list of subpopulation names/ids in which the specified modification will be applied. This can be a single `subpop`, a list, or the word `"all"` (specifying the modification applies to all existing subpopulations in the model). The modification will do nothing for any subpopulations not listed here.

`value:`The fractional reduction of the parameter during the time period the modification is active.  This can be a scalar number, or a distribution using the notation described in the [Distributions](introduction-to-configuration-files.md#distributions) section. The new parameter value will be

```
new_parameter_value = old_parameter_value * (1 - value)
```

`subpop_groups:` An optional list of lists specifying which subsets of subpopulations in subpop should share parameter values; when parameters are drawn from a distribution or fit to data. See [`subpop_groups`](intervention-templates.md#interventions-settings-groups) section below for more details.&#x20;

### MultiPeriodModifier

`MultiPeriodModifier` interventions enable the user to specify a multiplicative reduction to the `parameter` of interest by `value` (new = (1-`value`) \* old) for the subpopulations listed in `subpop` during multiple different time intervals each defined by a `start_date` and `end_date.`

For example, if you would like to describe the impact that transmission in schools has on overall disease spread, you could create a modifier that increases transmission by 30% during the dates that K-12 schools are in session in different regions (e.g., Massachusetts and Florida):

#### Example

```
school_year:
  method: MultiPeriodModifier
  parameter: beta
  groups:
    - subpop: ["25000"] 
      periods:
        - start_date: 2021-09-09
          end_date: 2021-12-23
        - start_date: 2022-01-04
          end_date: 2022-06-22
    - subpop: ["12000"]
      periods:
        - start_date: 2021-08-10
          end_date: 2021-12-17
        - start_date: 2022-01-04
          end_date: 2022-05-27
  value: -0.3
```

#### Configuration options

`method: MultiPeriodModifier`

`parameter`: The name of the parameter that will be modified. This could be a parameter defined for the transmission model in [`seir::parameters`](compartmental-model-structure.md) or for the observational model in [`outcomes`](outcomes-for-compartments.md). If the parameter is used in multiple transitions in the model then all those transitions will be modified by this amount.&#x20;

`groups:` A list of subpopulations (`subpops`) or groups of them, and time periods the modification will be active in each of them

* `groups:subpop` A list of subpopulation names/ids in which the specified modification will be applied. This can be a single `subpop`, a list, or the word `"all" (`specifying the modification applies to all existing subpopulations in the model). The modification will do nothing for any subpopulations not listed here.
* `groups: periods` A list of time periods, each defined by a start and end date, when the modification will be applied
  * `groups:periods:start_date` The date when the modification starts, in YYYY-MM-DD format. The modification will only reduce the value of the parameter after (inclusive of) this date.
  * `groups:periods:end_date` The date when the modification ends, in YYYY-MM-DD format. The modification will only reduce the value of the parameter before (inclusive of) this date.

`value:`The fractional reduction of the parameter during the time period the modification is active.  This can be a scalar number, or a distribution using the notation described in the [Distributions](introduction-to-configuration-files.md#distributions) section. The new parameter value will be

```
new_parameter_value = old_parameter_value * (1 - value)
```

`subpop_groups:` An optional list of lists specifying which subsets of subpopulations in subpop should share parameter values; when parameters are drawn from a distribution or fit to data. See [`subpop_groups`](intervention-templates.md#interventions-settings-groups) section below for more details.&#x20;

### ModifierModifier

`ModifierModifier` interventions allow the user to specify an intervention that acts to modify the value of _another intervention,_ as opposed to modifying a baseline parameter value.  The intervention multiplicatively reduces the `modifier` of interest by `value` (new = (1-`value`) \* old) for the subpopulations listed in `subpop` during  the time interval \[`period_start_date`, `period_end_date`].

#### Example

For example, `ModifierModifier` could be used to describe a social distancing policy that is in effect between two dates and reduces transmission by 60% if followed by the whole population, but part way through this period, adherence to the policy drops to only 50% of in one of the subpopulations population:

```
seir_modifiers:
  modifiers:
    social_distancing: 
      method: SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-03-15
      period_end_date: 2020-06-30
      subpop: ['all']
      value: 0.6
    fatigue: 
      method: ModifierModifier
      baseline_scenario: social_distancing
      parameter: beta
      period_start_date: 2020-05-01
      period_end_date: 2020-06-30
      subpop: ['large_province']
      value: 0.5
```

Note that this configuration is identical to the following alternative specification

```
seir_modifiers:
  modifiers:
    social_distancing_initial: 
      method: SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-03-15
      period_end_date: 2020-04-31
      subpop: ['all']
      value: 0.6
    social_distancing_fatigue_sp: 
      method: SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-05-01
      period_end_date: 2020-06-30
      subpop: ['small_province']
      value: 0.6
    social_distancing_fatigue_lp: 
      method: SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-05-01
      period_end_date: 2020-06-30
      subpop: ['large_province']
      value: 0.3
```

However, there are situations when the `ModiferModifier` notation is more convenient, especially when doing parameter fitting. &#x20;

#### Configuration options

`method: ModifierModifier`

`baseline_modifier:` The name of the original parameter modification which will be further modified.

`parameter`: The name of the parameter in the `baseline_scenario` that will be modified.&#x20;

`period_start_date`: The date when the intervention modifier starts, in YYYY-MM-DD format. The intervention modifier will only reduce the value of the other intervention after (inclusive of) this date.

`period_end_date`: The date when the intervention modifier ends, in YYYY-MM-DD format. The intervention modifier will only reduce the value of the other intervention before (inclusive of) this date.

`subpop:`A list of subpopulation names/ids in which the specified intervention modifier will be applied. This can be a single `subpop`, a list, or the word `"all"` (specifying the interventions applies to all existing subpopulations in the model). The intervention will do nothing for any subpopulations not listed here.

`value:`The fractional reduction of the baseline intervention during the time period the modifier intervention is active.  This can be a scalar number, or a distribution using the notation described in the [Distributions](introduction-to-configuration-files.md#distributions) section. The new parameter value will be

```
new_intervention_value = old_intervention_value * (1 - value)
```

and so the value of the underlying parameter that was modified by the baseline intervention will be

```
new_parameter_value = original_parameter_value * (1 - baseline_intervention_value * (1 - value) )
```

`subpop_groups:` An optional list of lists specifying which subsets of subpopulations in subpop should share parameter values; when parameters are drawn from a distribution or fit to data. See [`subpop_groups`](intervention-templates.md#interventions-settings-groups) section below for more details.&#x20;

### StackedModifier

Combine two or more modifiers into a scenario, so that they can easily be singled out to be run together without the other modifiers. If multiply modifiers act during the same time period in the same subpopulation, their effects are combined multiplicatively. Modifiers of different types (i.e. SinglePeriodModifier, MultiPeriodModifier, ModifierModifier, other StackedModifiers) can be combined.&#x20;

#### Examples

```
seir_modifiers:
  scenarios:
    -SchoolClosures
    -AllNPIs
  modifiers:
    SchoolClosures:
      method:SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-03-15
      period_end_date: 2020-05-01
      subpop: 'all'
      value: 0.7
    CaseIsolation:
      method:SinglePeriodModifier
      parameter: gamma
      period_start_date: 2020-04-01
      period_end_date: 2020-05-01
      subpop: 'all'
      value: -1.0
    Masking:
      method:SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-04-15
      period_end_date: 2020-05-01
      subpop: 'all'
      value: 0.5
    AllNPIs
      method: StackedModifier
      modifiers: ["SchoolClosures","CaseIsolation","Masking"]
```

or

```
outcome_modifiers:
  scenarios:
    - ReducedTesting
    - AllDelays
  modifiers:
    DelayedTesting
      method:SinglePeriodModifier
      parameter: incidC::probability
      period_start_date: 2020-03-15
      period_end_date: 2020-05-01
      subpop: 'all'
      value: 0.5
    DelayedHosp
      method:SinglePeriodModifier
      parameter: incidD::delay
      period_start_date: 2020-04-01
      period_end_date: 2020-05-01
      subpop: 'all'
      value: -1.0
    LongerHospStay
      method:SinglePeriodModifier
      parameter: incidH::duration
      period_start_date: 2020-04-15
      period_end_date: 2020-05-01
      subpop: 'all'
      value: -0.5
```

#### Configuration options

`method`: `StackedModifier`

`modifiers`: A list of names of the other modifiers (specified above) that will be combined to create the new modifier (which we typically refer to as a "scenario")

## modifiers::modifiers::groups

`subpop_groups:` For any of the modifier types, `subpop_groups` is an optional list of lists specifying which subsets of subpopulations in `subpop` should share parameter values; when parameters are drawn from a distribution or fit to data. All other subpopulations not listed will have unique intervention values unlinked to other areas. If the value is `'all',` then all subpopulations will be assumed to have the same modifier value. When the `subpop_groups` option is not specified, all subpopulations will be assumed to have unique values of the modifier.&#x20;

For example, for a model of disease spread in Canada where we want to specify that the (to be varied) value of a modification to the transmission rate should be the same in all the Atlantic provinces (Nova Scotia, Newfoundland, Prince Edward Island, and New Brunswick), the same in all the prairie provinces (Manitoba, Saskatchewan, Alberta), the same in the three territories (Nunavut, Northwest Territories, and Yukon), and yet take unique values in Ontario, Quebec, and British Columbia, we could write

```
seir_modifiers:
  modifiers:
    lockdown: 
      method: SinglePeriodModifier
      parameter: beta
      period_start_date: 2020-03-15
      period_end_date: 2020-05-01
      subpop: 'all'
      subpop_groups: [['NS','NB','PE','NF'],['MB','SK','AB'],['NV','NW','YK']]
      value: 
        distribution: uniform
        low: 0.3
        high: 0.7
        
```
