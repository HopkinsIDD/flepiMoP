---
description: >-
  This section describes how to specify the compartmental model of infectious
  disease transmission.
---

# Specifying compartmental model

We want to allow users to work with a wide variety of infectious diseases or, one infectious disease under a wide variety of modeling assumptions. To facilitate this, we allow the user to specify their compartmental model of disease dynamics via the configuration file.

We originally considered asking users to specify each compartment and transition manually. However, we quickly found that this created long, confusing configuration files, and so we created a shorthand to more succinctly specify both compartments and transitions between them. This works especially well for models where individuals are stratified by other properties (like age, vaccination status, etc.) in addition to their infection status.

The model is specified in two separate sections of the configuration file. In the `compartments` section, users define the possible states individuals can be categorized into. Then in the `seir` section, users define the possible transitions between states, the values of parameters that govern the rates of these transitions, and the numerical method used to simulate the model.

An example section of a configuration file defining a simple SIR model is below.

```
compartments:
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
    beta: 0.1
    gamma: 0.2
  integration:
     method: rk4
     dt: 1.00
```

## Specifying model compartments (`compartments`)

The first stage of specifying the model is to define the infection states (variables) that the model will track. These "compartments" are defined first in the `compartments` section of the config file, before describing the processes that lead to transitions between them. The compartments are defined separately from the rest of the model because they are also used by the `seeding` section that defines initial conditions and importations.

For simple disease models, the compartments can simply be listed with whatever notation the user chooses. For example, for a simple SIR model, the compartments could be `["S", "I", "R"]`. The config also requires that there be a variable name for the property of the individual that these compartments describe, which for example in this case could be `infection_stage`

```
compartments:
  infection_stage: ["S", "I", "R"]
```

Our syntax allows for more complex models to be specified without much additional notation. For example, consider a model of a disease that followed SIR dynamics but for which individuals could receive vaccination, which might change how they experience infection.

In this case we can specify compartments as the cross product of multiple states of interest. For example:

```
 compartments:
   infection_stage: ["S", "I", "R"]
   vaccination_status: ["unvaccinated", "vaccinated"]
```

Corresponds to 6 compartments, which the code internally converts to this data frame

```
infection_stage, vaccination_status, compartment_name
S,               unvaccinated,       S_unvaccinated
I,               unvaccinated,       I_unvaccinated
R,               unvaccinated,       R_unvaccinated
S,               vaccinated,         S_vaccinated
I,               vaccinated,         I_vaccinated
R,               vaccinated,         R_vaccinated
```

In order to more easily describe transitions, we want to be able to refer to a compartment by its components, but then use it by its compartment name.

If the user wants to specify a model in which some compartments are repeated across states but others are not, there will be pros and cons of how the model is specified. Specifying it using the cross product notation is simpler, less error prone, and makes config files easier to read, and there is no issue with having compartments that have zero individuals in them throughout the model. However, for very large models, extra compartments increase the memory required to conduct the simulation, and so having unnecessary compartments tracked may not be desired.

For example, consider a model of a disease that follows SI dynamics in two separate age groups (children and adults), but for which only adults receive vaccination, with one or two doses of vaccine. With the simplified notation, this model could be specified as:

```
 compartments:
   infection_stage: ["S", "I"]
   age_group: ["child", "adult"]
   vaccination_status: ["unvaccinated", "1dose", "2dose"]
```

corresponding to 12 compartments, 4 of which are unnecessary to the model

```
infection_stage, age_group, vaccination_status, compartment_name
S,		 child,	    unvaccinated,	S_child_unvaccinated	
I,		 child,	    unvaccinated,	I_child_unvaccinated
S,		 adult,	    unvaccinated,	S_adult_unvaccinated
I,		 adult,	    unvaccinated,	I_adult_unvaccinated
S,		 child,	    1dose,		S_child_1dose
I,		 child,	    1dose,		I_child_1dose
S,		 adult,     1dose,		S_adult_1dose
I,		 adult,     1dose,		I_adult_1dose
S,		 child,     2dose,		S_child_2dose	
I,		 child,     2dose,		I_child_2dose
S,		 adult,	    2dose,		S_adult_2dose
I,		 adult,	    2dose,		I_adult_2dose
```

Or, it could be specified with the less concise notation

```
compartments:
   overall_state: ["S_child", "I_child", "S_adult_unvaccinated", "I_adult_unvaccinated", "S_adult_1dose", "I_adult_1dose", "S_adult_2dose", "I_adult_2dose"]
```

which does not result in any unnecessary compartments being included.

These compartments are referenced in multiple different subsequent sections of the config. In the `seeding (LINK TBA)` section the user can specify how the initial (or later imported) infections are distributed across compartments; in the [`seir`](compartmental-model-structure.md#transitions-seir-transitions) section the user can specify the form and rate of the transitions between these compartments encoded by the model; in the [`outcomes`](outcomes-for-compartments.md) section the user can specify how the observed variables are generated from the underlying model states.

Notation must be consistent between these sections.

## Specifying compartmental model transitions (`seir::transitions`)

The way we specify transitions between compartments in the model is a bit more complicated than how the compartments themselves are specified, but allows users to specify complex stratified infectious disease models with minimal code. This makes checking, sharing, and updating models more efficient and less error-prone.

We specify one or more _transition globs_, each of which corresponds to one or more transitions. Since transition globs are shorthand for collections of transitions, we will first explain how to **specify a single transition** before discussing transition globs.

A transition has 5 pieces of associated information that a user can specify:

* `source`
* `destination`
* `rate`
* `proportional_to`
* `proportion_exponent`

For more details on the mathematical forms possible for transitions in our models, read the [Model Description section](../model-description.md#generalized-compartmental-infection-model).

We first consider a simple example of an SI model where individuals may either be vaccinated (_v_) or unvaccinated (_u_), but the vaccine does not change the susceptibility to infection nor the infectiousness of infected individuals.

<figure><img src="../../.gitbook/assets/simple_model_for_transitions (1).png" alt=""><figcaption></figcaption></figure>

We will focus on describing the first transition of this model, the rate at which unvaccinated individuals move from the susceptible to infected state.

### Specifying a single transition

#### Source

The compartment the transition moves individuals _out of_ (e.g., the _source_ compartment) is an array. For example, to describe a transition that moves unvaccinated susceptible individuals to another state, we would write

```
[S,unvaccinated]
```

which corresponds to the compartment `S_unvaccinated`

#### Destination

The compartment the transition moves individuals _into_ (e.g. the _destination_ compartment) is an array. For example, to describe a transition that moves individuals into the unvaccinated but infected state, we would write

```
[I,unvaccinated]
```

which corresponds to the compartment `I_unvaccinated`

#### Rate

The rate constant specifies the probability per time that an individual in the source compartment changes state and moves to the destination compartment. For example, to describe a transition that occurs with rate 5/time, we would write:

```
5
```

instead, we could describe the rate using a parameter `beta`, which can be given a numeric value later:

```
beta
```

The interpretation and unit of the rate constant depend on the model details, as the rate may potentially also be per number (or proportion) of individuals in other compartments (see below).

#### Proportional to

A vector of groups of compartments (each of which is an array) that modify the overall rate of transition between the source and destination compartment. Each separate group of compartments in the vector are first summed, and then all entries of the vector are multiplied to get the rate modifier. For example, to specify that the transition rate depends on the product of the number of unvaccinated susceptible individuals and the total infected individuals (vaccinated and unvaccinated), we would write:

```
[[[S,unvaccinated]], [[I,unvaccinated], [I, vaccinated]]]
```

To understand this term, consider the compartments written out as strings

```
[[S_unvaccinated], [I_unvaccinated, I_vaccinated]]
```

and then sum the terms in each group

```
[S_unvaccinated, I_unvaccinated + I_vaccinated]
```

From here, we can say that the transition we are describing is proportional to `S_unvaccinated` and `I_unvaccinated + I_vaccinated,` i.e., the rate depends on the product `S_unvaccinated * (I_unvaccinated + I_vaccinated)`.

For transitions that occur at a constant per-capita rate (ie, E -> I at rate $$\gamma$$ in an SEIR model), it is possible to simply write `proportional_to: ["source"]`.

#### Proportion exponent

This is an exponent modifying each group of compartments that contribute to the rate. It is equivalent to the "order" term in chemical kinetics. For example, if the reaction rate for the model above depends linearly on the number of unvaccinated susceptible individuals but on the total infected individuals sub-linearly, for example to a power 0.9, we would write:

```
[1, 0.9]
```

or a power parameter `alpha`, which can be given a numeric value later:

```
[1, alpha]
```

The (top level) length of the `proportion_exponent` vector must be the same as the (top level) length of the `proportional_to` vector, even if the desire of the user is to have the same exponent for all terms being multiplied together to get the rate.

#### Summary

Putting it all together, the model transition is specified as

```
source: [S, unvaccinated]
destination: [I, unvaccinated]
proportional_to: [[[S,unvaccinated]], [[I,unvaccinated], [I,vaccinated]]]
rate: [5]
proportion_exponent: [1, 0.9]
```

would correspond to the following model if expressed as an ordinary differential equation

$$
\frac{\delta \text{S}_\text{unvaccinated}}{\delta t} = - \beta \text{S}_\text{unvaccinated}^1 (\text{I}_\text{unvaccinated}+\text{I}_\text{vaccinated})^{\alpha}
$$

$$
\frac{\delta \text{I}_\text{unvaccinated}}{\delta t} = \beta \text{S}_\text{unvaccinated}^1 (\text{I}_\text{unvaccinated}+\text{I}_\text{vaccinated})^{\alpha}
$$

with parameter and parameter (we will describe how to use parameter symbols in the transitions and specify their numeric values separately in the section [Specifying compartmental model parameters](compartmental-model-structure.md#specifying-compartmental-model-parameters)).

### Transition globs

We now explain a shorthand we have developed for specifying multiple transitions that have similar forms all at once, via _transition globs_. The basic idea is that for each component of the single transitions described above where a term corresponded to a single model compartment, we can instead specify _one or more_ compartment. Similarly, multiple rate values can be specified at once, for each involved compartment. From one transition glob, multiple individual transitions are created, by _broadcasting_ across the specified compartments.

For transition globs, any time you could specify multiple arguments as a list, you may instead specify one argument as a non-list, which will be used for every broadcast. So \[1,1,1] is equivalent to 1 if the dimension of that broadcast is 3.

We continue with the same SI model example, where individuals are stratified by vaccination status, but expand it to allow infection to occur at different rates in vaccinated and unvaccinated individuals:

<figure><img src="../../.gitbook/assets/simple_model_for_transitions_v2 (2).png" alt=""><figcaption><p>A stratified SI model including vaccination</p></figcaption></figure>

#### Source

We allow one or more arguments to be specified for each compartment. So to specify the transitions out of both susceptible compartments (`S_unvaccinated` and `S_unvaccinated`), we would use

```
[[S], [unvaccinated,vaccinated]]
```

#### Destination

The destination variable should be the same shape as the `source`, and in the same relative order. So to specify a transition from `S_unvaccinated` to `I_unvaccinated` and `S_vaccinated` to `I_vaccinated`, we would write the `destination` as:

```
[[I], [unvaccinated,vaccinated]]
```

If instead we wrote:

```
[[I], [vaccinated,unvaccinated]]
```

we would have a transition from `S_unvaccinated` to `I_vaccinated` and `S_vaccinated` to `I_unvaccinated`.

#### Rate

The rate vector allows users to specify the rate constant for all the source -> destination transitions that are defined in a shorthand way, by instead specifying how the rate is altered depending on the compartment type. For example, the rate of transmission between a susceptible (S) and an infected (I) individual may vary depending on whether the susceptible individual is vaccinated or not **and** whether the infected individual is vaccinated or not. The overall rate constant is constructed by multiplying together or "broadcasting" all the compartment type-specific terms that are relevant to a given compartment.

For example,

```
rate: [[3], [0.6,0.5]]
```

This would mean our transition from `S_unvaccinated` to `I_unvaccinated` would have a rate of `3 * 0.6` while our transition from `S_vaccinated` to I`_vaccinated` would have a rate of `3 * 0.5`.

The rate vector should be the same shape as `source` and `destination` and in the same relative order.

Note that if the desire is to make a model where the difference in the rate constants varies in a more complicated than multiplicative way between different compartment types, it would be better to specify separate transitions for each compartment type instead of using this shorthand.

#### Proportional to

The broadcasting here is a bit more complicated. In other cases, each broadcast is over a single component. However, in this case, we have a broadcast over a group of components. We allow a different group to be chosen for each broadcast.

```
[
  [[S,unvaccinated], [S,vaccinated]],
  [[I,unvaccinated],[I, vaccinated]], [[I,unvaccinated],[I, vaccinated]]
]
```

Again, let's unpack what it says. Since the broadcast is over groups, let's split the config back up

into those groups

```
[
  [S,unvaccinated],
  [[I,unvaccinated],[I, vaccinated]]
]
[
  [S,vaccinated],
  [[I,unvaccinated],[I, vaccinated]]
]
```

From here, we can say that we are describing two transitions. Both occur proportionally to the same compartments: `S_unvaccinated` and the total number of infections (`I_unvaccinated+I_vaccinated`).

If, for example, we want to model a situation where vaccinated susceptibles cannot be infected by unvaccinated individuals, we would instead write:

```
[
  [[S,unvaccinated], [S,vaccinated]],
  [[I,unvaccinated],[I, vaccinated]], [[I, vaccinated]]
]
```

#### Proportion exponent

Similarly to `rate` and `proportional_to`, we provide an exponent for each component and every group across the broadcast. So we could for example use:

```
[[1,1], [0.9,0.8]]
```

The (top level) length of the `proportion_exponent` vector must be the same as the (top level) length of the `proportional_to` vector, even if the desire of the user is to have the same exponent for all terms being multiplied together to get the rate. Within each vector entry, the arrays must have the same length as the `source` and `destination` vectors.

#### Summary

Putting it all together, the transition glob

```
seir:
  transitions:
    source: [[S],[unvaccinated,vaccinated]]
    destination: [[I],[unvaccinated,vaccinated]]
    proportional_to: [
                       [[S,unvaccinated], [S,vaccinated]],
                       [[I,unvaccinated],[I, vaccinated]], [[I, vaccinated]]
                     ]
    rate: [[3], [0.6,0.5]]
    proportion_exponent: [[1,1], [0.9,0.8]]
```

is equivalent to the following transitions

```
seir:
  transitions:
    - source: [S,unvaccinated]
      destination: [I,unvaccinated]
      proportional_to: [[[S,unvaccinated]], [[I,unvaccinated],[I, vaccinated]]]
      proportion_exponent: [1 * 0.9]
      rate: [3*0.6]
    - source: [S,vaccinated]
      destination: [I,vaccinated]
      proportional_to: [[[S,vaccinated]], [[I, vaccinated]]]
      proportion_exponent: [1 * 0.8]
      rate: [3*0.5]
```

#### Warning

We warn the user that with this shorthand, it is possible to specify large models with few lines of code in the configuration file. The more compartments and transitions you specify, the longer the model will take to run, and the more memory it will require.

## Specifying compartmental model parameters (`seir::parameters`)

When the transitions of the compartmental model are specified as described above, they can either be entered as numeric values (e.g., `0.1`) or as strings which can be assigned numeric values later (e.g., `beta`). We recommend the latter method for all but the simplest models, since parameters may recur in multiple transitions and so that parameter values may be edited without risk of editing the model structure itself. It also improves readability of the configuration files.

Parameters can take on three types of values:

* Fixed values
* Value drawn from distributions
* Values read from timeseries specified in a data file

### Specifying fixed parameter values

Parameters can be assigned values by using the `value` argument after their name and then simply stating their numeric argument. For example, in a config describing a simple SIR model with transmission rate $$\beta$$ (`beta`) = 0.1/day and recovery rate $$\gamma$$ (`gamma`) = 0.2/day. This could be specified as

```
seir:
  parameters:
    beta: 
      value: 0.1
    gamma: 
      value: 0.2
```

The full model section of the config could then read

```
compartments:
  infection_state: ["S", "I", "R"]
  
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
      proportion_exponent: [1,1]
  parameters:
    beta: 
      value: 0.1
    gamma: 
      value: 0.2
```

For the stratified SI model described [above](compartmental-model-structure.md#transition-globs), this portion of the config would read

```
compartments:
  infection_stage: ["S", "I", "R"]
  vaccination_status: ["unvaccinated", "vaccinated"]
  
seir:
  transitions:
    source: [[S],[unvaccinated,vaccinated]]
    destination: [[I],[unvaccinated,vaccinated]]
    proportional_to: [
                       [[S,unvaccinated], [S,vaccinated]],
                       [[I,unvaccinated],[I, vaccinated]], [[I, vaccinated]]
                     ]
    rate: [[beta], [theta_u,theta_v]]
    proportion_exponent: [[1,1], [alpha_u,alpha_v]]
  parameters:
    beta: 
      value: 0.1
    theta_u: 
      value: 0.6
    theta_v: 
      value: 0.5
    alpha_u: 
      value: 0.9
    alpha_v: 
      value: 0.8
```

If there are no parameter values that need to be specified (all rates given numeric values when defining model transitions), the `seir::parameters` section of the config can be left blank or omitted.

### Specifying parameters values from distributions

Parameter values can also be specified as random values drawn from a distribution, as a way of including uncertainty in parameters in the model output. In this case, every time the model is run independently, a new random value of the parameter is drawn. For example, to choose the same value of `beta` = 0.1 each time the model is run but to choose a random values of `gamma` with mean on a log scale of $$e^{-1.6} = 0.2$$ and standard deviation on a log scale of $$e^{0.2} = 1.2$$ (e.g., 1.2-fold variation):

```
seir:
  parameters:
    beta: 
      value:
        distribution: fixed
        value: 0.1
    gamma: 
      value:
        distribution: lognorm
        logmean: -1.6
        logsd: 0.2
```

Details on the possible distributions that are currently available, and how to specify their parameters, is provided in the [Distributions section](distributions.md).

Note that understanding when a new parameter values from this distribution is drawn becomes more complicated when the model is run in [Inference](../../model-inference/inference-description.md) mode. In Inference mode, we distinguish model runs as occurring in different "slots" – i.e., completely independent model instances that could be run on different processing cores in a parallel computing environment – and different "iterations" of the model that occur sequentially when the model is being fit to data and update fitted parameters each time based on the fit quality found in the previous iteration. A new parameter values is only drawn from the above distribution **once per slot**. Within a slot, at each iteration during an inference run, the parameter is only changed if it is being fit and the inference algorithm decides to perturb it to test a possible improved fit. Otherwise, it would maintain the same value no matter how many times the model was run within a slot.

### Specifying parameter values as timeseries from data files

Sometimes, we want to be able to specify model parameters that have different values at different timepoints. For example, the relative transmissibility may vary throughout the year based on the weather conditions, or the rate at which individuals are vaccinated may vary as vaccine programs are rolled out. One way to do this is to instead specify the parameter values as a timeseries.

This can be done by providing a data file in .csv or .parquet format that has a list of values of the parameter for a corresponding timepoint and subpopulation name. One column should be `date`, which should have an entry for every calendar day of the simulation, with the first and last date corresponding to the `start_date` and `end_date` for the simulation specified in the header of the config. There should be another column for each subpopulation, where the column name is the subpop name used in other files and the values are the desired parameter values for that subpopulation for the corresponding day. If any day or subpopulation is missing, an error will occur. However, if you want all subpopulations to have the same parameter value for every day, then only a single column in addition to date is needed, which can have any name, and will be applied to every subpop ;

For example, for an SIR model with a simple [two-province population structure](specifying-population-structure.md#example-1) where the relative transmissibility peaks on January 1 then decreases linearly to a minimal value on June 1 then increases linearly again, but varies more in the small province than the large province, the `theta` parameter could be constructed from the file **seasonal\_transmission\_2pop.csv** with contents including

```
date,        small_province,    large_province
2022-01-01,  1.5,               1.3
.....
2022-05-01,  0.5,               0.7 
....
2022-12-31,  1.5,               1.3
```

as a part of a configuration file with the model sections:

```
compartments:
  infection_stage: ["S", "I", "R"]

seir:
  transitions:
    # infection
    - source: [S]
      destination: [I]
      proportional_to: [[S], [I]]
      rate: [beta*theta]
      proportion_exponent: 1
    # recovery
    - source: [I]
      destination: [R]
      proportional_to: [[I]]
      rate: [gamma]
      proportion_exponent: 1
  parameters:
    beta: 
      value: 0.1
    gamma: 
      value: 0.2
    theta:
       timeseries: data/seasonal_transmission.csv
```

Note that there is an alternative way to specify time dependence in parameter values that is described in the [Specifying time-varying parameter modifications](intervention-templates.md) section. That method allows the user to define intervention parameters that apply specific additive or multiplicative shifts to other parameter values for a defined time interval. Interventions are useful if the parameter doesn't vary frequently and if the values of the shift is unknown and it is desired to either sample over uncertainty in it or try to estimate its value by fitting the model to data. If the parameter varies frequently and its value or relative value over time is known, specifying it as a timeseries is more efficient.

Compartmental model parameters can have an additional attribute beyond `value` or `timeseries`, which is called `stacked_modifier_method`. This value is explained in the section on coding [`time-dependent parameter modifications`](intervention-templates.md) (also known as "modifiers") as it determines what happens when two different modifiers act on the same parameter at the same time (are they combined additively or multiplicatively?) ;

| Config item               | Required?                              | Type/Format                                   | Description                                                                               |
| ------------------------- | -------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `value`                   | either value or timeseries is required | numerical, or distribution                    | This defines the value of the parameter, as described above.                              |
| `timeseries`              | either value or timeseries is required | path to a csv file                            | This defines a timeseries for each day, as above.                                         |
| `stacked_modifier_method` | optional                               | string: `sum`, `product`, `reduction_product` | This option defines the method used when modifiers are applied. The default is `product`. |
| `rolling_mean_windows`    | optional                               | integer                                       | The size of the rolling mean window if a rolling mean is applied.                         |

## Specifying model simulation method `(seir::integration)`

A compartmental model defined using the notation in the previous sections describes rules for classifying individuals in the population based on infection state dynamically, but does not uniquely specify the mathematical framework that should be used to simulate the model.

Our framework allows for two major methods for implementing compartmental models of disease transmission:

* ordinary differential equations, which are completely deterministic, operate in continuous time (consider infinitesimally small timesteps), and allow for arbitrary fractions of the population (i.e., not just discrete individuals) to move between model compartments
* discrete-time stochastic process, which tracks discrete individuals and produces random variation in the number of individuals transitioning between states for any given rate, and which allows transitions between states only to occur at discrete time intervals

The mathematics behind each implementation is described in the [Model Description](../model-description.md) section

<table><thead><tr><th>Config item</th><th width="122">Required?</th><th width="171">Type/Format</th><th>Description</th></tr></thead><tbody><tr><td><code>method</code></td><td>optional</td><td>string: <code>stochastic</code>,<code>rk4</code>, or <code>legacy</code></td><td>The algorithm used to simulate the mode equations. If <code>stochastic</code>, uses a discrete-time stochastic process with a rate-saturation correction. If <code>rk4</code>, model is simulated deterministically by numerical integration using a 4th order Runge-Kutta algorithm. If <code>legacy</code>(Default), uses the transition rates for the stochastic model but always chooses the average rate (an Euler style update)</td></tr><tr><td><code>dt</code></td><td>optional</td><td>Any positive real number</td><td>The timestep used for the numerical integration or discrete time stochastic update. Default is <code>dt = 2</code></td></tr></tbody></table>

For example, to simulate a model deterministically using the 4th order Runge-Kutta algorithm for numerical integration with a timestep of 1 day:

```
seir:
  integration:
     method: rk4
     dt: 1.00
```

Alternatively, to simulate a model stochastically with a timestep of 0.1 days

```
seir:
  integration:
     method: stochastic
     dt: 0.1
```

For any method, the results of the model will be more accurate when the timestep is smaller (i.e., output will more precisely match the mathematics of the model description and be invariant to the choice of timestep). However, the computing time required to simulate the model for a certain time range of interest increases with the number of timesteps required (i.e., with smaller timesteps). In our experience, the 4th order Runge-Kutta algorithm (for details see [Advanced](../../more/advanced/) section) is a very accurate method of numerically integrating such models and can handle timesteps as large as roughly a day for models with the maximum per capita transition rates in this same order of magnitude. However, the discrete time stochastic model or the legacy method for integrating the model in deterministic mode require smaller timesteps to be accurate (around 0.1 for COVID-19-like dynamics in our experience.
