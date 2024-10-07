---
description: >-
  This page describes how to specify the outcomes section of the configuration
  file
---

# Specifying observational model

## Thinking about `outcomes` variables

Our pipeline allows users to encode state variables describing the infection status of individuals in the population in two different ways. The first way is via the state variables and transitions of the compartmental model of disease transmission, which are specified in the [`compartments`](compartmental-model-structure.md) and [`seir`](compartmental-model-structure.md) sections of the config. This model should include all variables that influence the natural course of the epidemic (i.e., all variables that feed back into the model by influencing the rate of change of other variables). For example, the number of infected individuals influences the rate at which new infections occur, and the number of immune individuals influences the number of individuals at risk of acquiring infection.

However, these intrinsic model variables may be difficult to observe in the real world and so directly comparing model predictions about the values of these variables to data might not make sense. Instead, the observable outcomes of infection may include only a subset of individuals in any state, and may only be observed with a time delay. Thus, we allow users to define new `outcome` variables that are functions of the underlying model variables. Commonly used examples include detected cases or hospitalizations.&#x20;

Variables should not be included as outcomes if they influence the infection trajectory. The choice of what variables to include in the compartmental disease model vs. the outcomes section may be very model specific. For example, hospitalizations due to infection could be encoded as an outcome variable that is some fraction of infections, but if we believe hospitalized individuals are isolated from the population and don't contribute to onward infection, or that the number of hospitalizations feeds back into the population's perception of risk of infection and influences everyone's contact behavior, this would not be the best choice. Similarly, we could include deaths due to infection as an outcome variable that is also some fraction of infections, but unless death is a very rare outcome of infection and we aren't worried about actually removing deceased individuals from the modeled populations, deaths should be in the compartmental model instead.

The `outcomes` section is not required in the config. However, there are benefits to including it, even if the only outcome variable is set to be equivalent to one of the infection model variables. If the compartmental model is complicated but you only want to visualize a few output variables, the [outcomes output file](../output-files.md) will be much easier to work with.  Outcome variables always occur with some fixed delay from their source infection model variable, which can be more convenient than the exponential distribution underlying the infection model.  Outcome variables can be created to automatically sum over multiple compartments of the infection model, removing the need for post-processing code to do this. If the model is being fit to data, then the `outcomes` section is required, as only outcome variables can be compared to data.

As an example, imagine we are simulating an SIR-style model and want to compare it to real epidemic data in which cases of infection and death from infection are reported. Our model doesn't explicitly include death, but suppose we know that 1% of all infections eventually lead to hospitalization, and that hospitalization occurs on average 1 week after infection. We know that not all infections are reported as cases, and assume that only 50% are detected and are reported 2 days after infection begins. The model and `outcomes` section of the config for these outcomes, which we call `incidC` (daily incidence of cases) and `incidH` (daily incidence of hospital admission) would be

<pre><code>compartments:
  infection_stage: ["S", "I", "R"]
  
seir:
  transitions:
    # infection
    - source: [S]
      destination: [I]
      proportional_to: [[S], [I]]
      rate: [beta]
      proportion_exponent: 1
    # recovery
    - source: [I]
      destination: [R]
      proportional_to: [[I]]
      rate: [gamma]
      proportion_exponent: 1
  parameters:
    beta: 
      value: 0.2
    gamma: 
      value: 0.1

<strong>outcomes:
</strong>  settings:
    method: delayframe
  outcomes:
    incidC:
      source:
        incidence:
          infection_stage: "I"
      probability: 
        value: 0.5
      delay: 
        value: 2
    incidH:
      source:
        incidence:
          infection_stage: "I"
      probability: 
        value: 0.01
      delay: 
        value: 21 
</code></pre>

in the following sections we describe in more detail how this specification works

## Specifying `outcomes` in the configuration file

The `outcomes` config section consists of a list of defined outcome variables (observables), which are defined by a user-created name (e.g., "`incidH`"). For each of these outcome variables, the user defines the `source` compartment(s) in the infectious disease model that they draw from and  whether they draw from the `incidence` (new individuals entering into that compartment) or `prevalence` (total current individuals in that compartment). Each new outcome variable is always associated with two mandatory parameters:&#x20;

* `probability` of being counted in this outcome variable if in the source compartment
* &#x20;`delay` between when an individual enters the `source` compartment and when they are counted in the outcome variable

and one optional parameter

* `duration` after entering that an individual is counted as part of the outcome variable

The `value` of the `probability`, `delay`, and `duration` parameters can be a single value or come from [distribution](distributions.md).&#x20;

Outcome model parameters `probability`, `delay`, and `distribution` can have an additional attribute beyond `value` called `modifier_key`. This value is explained in the section on coding [`time-dependent parameter modifications`](intervention-templates.md) (also known as "modifiers") as it provides a way to have the same modifier act on multiple different outcomes.&#x20;

{% hint style="info" %}
Just like the case for [compartment model parameters](compartmental-model-structure.md#specifying-compartmental-model-parameters-seir-parameters), when outcome parameters are drawn from a distribution, each time the model is run, a different value for this parameter will be drawn from this distribution, but that value will be used for all calculations within this model run. Note that understanding when a new parameter values from this distribution is drawn becomes more complicated when the model is run in [Inference](../../model-inference/inference-description.md) mode. In Inference mode, we distinguish model runs as occurring in different "slots" – i.e., completely independent model instances that could be run on different processing cores in a parallel computing environment – and different "iterations" of the model that occur sequentially when the model is being fit to data and update fitted parameters each time based on the fit quality found in the previous iteration. A new parameter values is only drawn from the above distribution **once per slot**. Within a slot, at each iteration during an inference run, the parameter is only changed if it is being fit and the inference algorithm decides to perturb it to test a possible improved fit. Otherwise, it would maintain the same value no matter how many times the model was run within a slot.
{% endhint %}

**Example**

```
// Some code
```

####

<table><thead><tr><th width="165">Config item</th><th>Required?</th><th>Type/format</th><th>Description</th></tr></thead><tbody><tr><td><code>source</code></td><td>Yes</td><td>Varies</td><td>The infection model variable or outcome variable from which the named outcome variable is created</td></tr><tr><td><code>probability</code></td><td>Yes, unless sum option is used instead</td><td>value or distribution</td><td>The probability that an individual in the <code>source</code> variable appears in the named outcome variable</td></tr><tr><td><code>delay</code></td><td>Yes, unless sum option is used instead</td><td>value or distribution</td><td>The time delay between individual's appearance in <code>source</code> variable and appearance in named outcome variable</td></tr><tr><td><code>duration</code></td><td>No</td><td>value or distribution</td><td>The duration of time an individual remains counted within the  named outcome variablet</td></tr><tr><td><code>sum</code></td><td>No</td><td>List</td><td>A list of other outcome variables to sum into the current outcome variable</td></tr></tbody></table>

#### `source`&#x20;

Required, unless [`sum`](outcomes-for-compartments.md#sum) option is used instead. This sub-section describes the compartment(s) in the infectious disease model from which this outcome variable is drawn. Outcome variables can be drawn from the `incidence` of a variable - meaning that some fraction of new individuals entering the infection model state each day are chosen to contribute to the outcome variable - or from the `prevalence`, meaning that each day some fraction of individuals currently in the infection state are chosen to contribute to the outcome variable. Note that whatever the source type, **the named outcome variable itself is always a measure of incidence**.&#x20;

To specify which compartment(s) contribute the user must specify the state(s) within each model stratification. For stratifications not mentioned, the outcome will sum over that states in all strata.&#x20;

For example, consider a configuration in which the compartmental model was constructed to track infection status stratified by vaccination status and age group.  The following code would be used to create an outcome called `incidH_child` (incidence of hospitalization for children) and `incidH_adult` (incidence of hospitalization for adults) where some fraction of infected individuals would become hospitalized and we wanted to track separately track pediatric vs adult hospitalizations, but did not care about tracking the vaccination status of hospitalized individuals as in reality it was not tracked by the hospitals.&#x20;

```
 compartments:
   infection_state: ["S", "I", "R"]
   age_group: ["child", "adult"]
   vaccination_status: ["unvaxxed", "vaxxed"]
   
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    ...
  incidH_adult:
    source:
      incidence:
        infection_state: "I"
        age_group: "adult"
    ...
  incidH_all:
    source:
      incidence:
        infection_state: "I"
    ...
```

to instead create an outcome variable for cases where on each day of infection there is some probability of testing positive (for example, for the situation of an asymptomatic infection where testing is administered totally randomly), the following code would be used

```
 compartments:
   infection_state: ["S", "I", "R"]
   age_group: ["child", "adult"]
   vaccination_status: ["unvaxxed", "vaxxed"]
   
outcomes:
  incidC:
    source:
      prevalence:
        infection_state: "I"
    ...
```

The source of an outcome variable can also be a previous defined outcome variable. For example, t to create a new variable for the number of individuals recruited to be part of a contact tracing program (incidT), which is just some fraction of diagnosed cases,&#x20;

```
outcomes:
  incidC:
    source:
      prevalence:
        infection_state: "I"
    ...
  incidT:
    source: incidC
    ...
```

#### `probability`&#x20;

Required, unless [`sum`](outcomes-for-compartments.md#sum) option is used instead. `Probability` is the fraction of individuals in the source compartment who are counted as part of this outcome variable (if the source is incidence; if the source is prevalence it is the fraction of individuals per day). It must be between 0 and 1.&#x20;

Specifying the probability creates a parameter called `outcome_name::probability` that can be referred to in the [`outcome_modifiers`](intervention-templates.md) section of the config. The value of this parameter can be changed using the `probability::intervention_param_name` option.&#x20;

For example, to track the incidence of hospitalization when 5% of children but only 1% of adults infected require hospitalization, and to create a `modifier_key` such that both of these rates could be modified by the same amount during some time period using the [`outcomes_modifier`](intervention-templates.md) section:

```
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    probability: 
      value: 0.05
      modifier_key: hosp_rate
  incidH_adult:
    source:
      incidence:
        infection_state: "I"
        age_group: "adult"
    probability: 
      value: 0.01
      modifier_key: hosp_rate
```

To track the incidence of diagnosed cases iterating over uncertainty in the case detection rate (ranging 20% to 30%), and naming this parameter "case\_detect\_rate"

```
outcomes:
  incidC:
    source:
      prevalence:
        infection_state: "I"
    probability:
      value:
        distribution: uniform
        low: 
          value: 0.2
        high: 
          value: 0.3
      intervention_param_name: "case_detect_rate"
```

Each time the model is run a new random value for the probability of case detection will be chosen.&#x20;

#### Delay

Required, unless [`sum`](outcomes-for-compartments.md#sum) option is used instead. `delay` is the time delay between when individuals are chosen from the source compartment and when they are counted as part of this outcome variable.&#x20;

For example, to track the incidence of hospitalization when 5% of children are hospitalized and hospitalization occurs 7 days after infection:

```
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    probability: 
      value: 0.05
    delay: 
      value: 7
```

To iterate over uncertainty in the exact delay time, we could include some variation between simulations in the delay time using a normal distribution with standard deviation of 2 (truncating to make sure the delay does not become negative). Note that a delay distribution here **does not mean** that the delay time varies between individuals - it is identical).&#x20;

```
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    probability: 
      value: 0.05
    delay: 
      value: 
        distribution: truncnorm
        mean: 7
        sd: 2
        a: 0
        b: Inf
```

#### Duration

By default, all outcome variables describe incidence (new individuals entering each day). However, they can also track an associated "prevalence" if the user specifies how long individuals will stay classified as the outcome state the outcome variable describes. This is the `duration` parameter.&#x20;

When the duration parameter is set, a new outcome variable is automatically created and named with the name of the original outcome variable + "\_curr". This name can be changed using the `duration::name` option.&#x20;

For example, to track the incidence and prevalence of hospitalization when 5% of children are hospitalized, hospitalization occurs 7 days after infection, and the duration of hospitalization is 3 days:

```
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    probability: 
      value: 0.05
    delay: 
      value: 7
    duration: 
      value: 3
```

which creates the variable "incidH\_child\_curr" to track all currently hospitalized children. Since it doesn't make sense to call this new outcome variable an incidence, as it is a prevalence, we could instead rename it:

```
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    probability: 
      value: 0.05
    delay: 
      value: 7
    duration: 
      value: 3
      name: "hosp_child_curr"
```

#### Sum

Optional. `sum` is used to create new outcome variables that are sums over other previously defined outcome variables.&#x20;

If `sum` is included, `source`, `probability`, `delay`, and `duration` will be ignored.&#x20;

For example, to track new hospital admissions and current hospitalizations separately for children and adults, as well as for all ages combined

```
outcomes:
  incidH_child:
    source:
      incidence:
        infection_state: "I"
        age_group: "child"
    probability: 0.05
    delay: 6
    duration: 
      value: 14
      name: "hosp_child_curr"
  incidH_adult:
    source:
      incidence:
        infection_state: "I"
        age_group: "adult"
    probability: 0.01
    delay: 8
    duration:
      value: 7
      name: "hosp_adult_curr"
  incidH_total: 
    sum: ["incidH_child","incidH_adult"]
  hosp_curr_total:   
    sum: ["hosp_child_curr","hosp_adult_curr"]
```

### outcomes::settings

There are other required and optional configuration items for the `outcomes` section which can be specified under `outcomes::settings`:

`method`:  `delayframe.`This is the mathematical method used to create the outcomes variable values from the transmission model variables. Currently, the only model supported is `delayframe`, which ...&#x20;

`param_from_file:`  Optional, `TRUE` or `FALSE`.  It is possible to allow any of the outcomes variables to have values that vary across the subpopulations. For example, disease severity rates or diagnosis rates may differ by demographic group. In this case, all the outcome parameter values defined in [outcomes::outcomes](outcomes-for-compartments.md#outcomes-outcomes) will represent baseline values, and then you can define a relative change from this baseline for any particular subpopulation using the paths section. If `params_from_file: TRUE` is specified, then these relative values will be read from the `params_subpop_file`. Otherwise, if `params_from_file: FALSE` or is not listed at all, all subpopulations will have the same values for the outcome parameters, defined below.&#x20;

`param_subpop_file`: Required if `params_from_file: TRUE`. The path to a .csv or .parquet file that contains the relative amount by which a given outcome variable is shifted relative to baseline in each subpopulation. File must contain the following columns:

* `subpop`: The subpopulation for which the parameter change applies. Must be a subpopulation defined in the [`geodata`](specifying-population-structure.md#geodata-file) file. For example, `small_province`
* parameter:  The outcomes parameter which will be altered for this subpopulation. For example, `incidH_child: probability`
* value: The amount by which the baseline value will be multiplied, for example, 0.75 or 1.1

## Examples

Consider a disease described by an SIR model in a population that is divided into two age groups, adults and children, which experience the disease separately. We are interested in comparing the predictions of the model to real world data, but we know we cannot observe every infected individual. Instead, we have two types of outcomes that are observed.

First, via syndromic surveillance, we have a database that records how many individuals in the population are experiencing symptoms from the disease at any given time. Suppose careful cohort studies have shown that 50% of infected adults and 80% of infected children will develop symptoms, and that symptoms occur in both age groups around 3 days after infection (following a log-normal distribution with log mean X and log standard deviation of Y). The duration that symptoms persist is also a variable, following a ...

Secondly, via laboratory surveillance we have a database of every positive test result for the infection. We assume the test is 100% sensitive and specific. Only individuals with symptoms are tested, and they are always tested exactly 1 day after their symptom onset. We are unsure what portion of symptomatic individuals are seeking out testing, but are interested in considering two extreme scenarios: 95% of symptomatic individuals are tested, or only 75% of individuals are tested.

The configuration file we could use to model this situation includes

```
// Some code

```

