# Modeling infectious disease dynamics

Within _flepiMoP, gempyor_ is an open-source Python package that constructs and simulates compartmental infectious disease dynamics models. _gempyor_ is meant to be used within _flepiMoP_, where it integrates with parameter inference and data processing scripts, but can also be **run standalone with a command-line interface**, generating simulations of disease incidence under different scenario assumptions.

To simulate an infectious disease dynamics problem, the following building blocks needs to be defined:

* The [population structure](model-description.md#population-structure) over which the disease is transmitted
* The [transmission model](model-implementation/compartmental-model-structure.md), defining the compartments and the transitions between compartments
* An [observation model](model-implementation/outcomes-for-compartments.md), defining different observable outcomes (serology, hospitalization, deaths, cases) from the transmission model
* The parameters and modifiers that apply to them

### Generalized compartmental infection model

At the core of our pipeline is a dynamic mathematical model that categorizes individuals in the population into a discrete set of states ('compartments') and describes the rates at which transitions between states can occur. Our modeling approach was developed to describe classic infectious disease transmission models, like the SIR model, but is much more general. It can encode _any_ compartmental model in which transitions between states are of the form

$$
X \xrightarrow{b X Z^{a}} Y,
$$

where $$X$$, $$Y$$, and $$Z$$ are time-dependent variables describing the number of individuals in each state, $$b$$ is a rate parameter (units of time$$^{-1}$$) and $$a$$ is a scaling parameter (unitless). $$Z$$ may be $$X$$, $$Y$$, a different variable, or 1, and the rate may also be the sum of terms of this form. Rates that involve non-linear functions or more than two variables are currently not possible. For simplicity, we omitted the time dependencies on parameters (e.g $$X$$ is in fact $$X(t)$$ and $$a$$,$$b$$ are $$a(t)$$,$$b(t)$$).

The model can be simulated as a continuous-time, deterministic process (i.e., a set of **ordinary differential equations**), which in this example would be in the form

$$
\frac{dX}{dt} = b X Z^a.
$$

Details on the numerical integration procedure for simulating such an equation is given in the [Advanced](../more/advanced/) section.

Alternatively, the model can be simulated as a **discrete-time stochastic process**, where the number of individuals transitioning between states $$X$$ and $$Y$$ at time $$t$$ is a binomial random variable

$$
N_{X\rightarrow Y}(t) = \textrm{Binom}(X,1-e^{-\Delta{t} \cdot bZ(t)^a}),
$$

where the second term is the expected fraction of individuals in the $$X$$ state at time $$t$$ who would transition to $$Y$$ by time $$t+\Delta t$$ if there were no other changes to $$X$$ in this time, and time step $$\Delta{t}$$ is a chosen parameter that must be small for equivalence between continuous- and discrete-time versions of the model.

#### SEIR model

For example, an **SEIR model** – which describes an infection for which susceptible individuals ($$S$$) who are infected first pass through a latent or exposed ($$E$$) phase before becoming infectious ($$I$$) and that confers perfect lifelong immunity after recovery ($$R$$) – could be encoded as

$$
S \xrightarrow{\beta S I/N} E \xrightarrow{\sigma E} I \xrightarrow{\gamma I} R,
$$

where $$\beta$$ is the transmission rate (rate of infectious contact per infectious individual), $$\sigma$$ is the rate of progression ($$1/\sigma$$ is the average latent/incubation period), and $$\gamma$$ is the recovery rate ($$1/\gamma$$ is the average duration of the infectious period), and $$N$$ is the total population size ($$N=S+E+I+R$$). In differential equation form, this model is

$$
\frac{dS}{dt} = - \beta S \frac{I}{N} ,
$$

$$
\frac{dE}{dt} = \beta S \frac{I}{N} - \sigma E,
$$

$$
\frac{dI}{dt} = \sigma E - \gamma I,
$$

$$
\frac{dR}{dt} = \gamma I,
$$

and as a stochastic process, it is

$$
N_{S\rightarrow E}(t) = \textrm{Binom}(S(t),1-e^{-\Delta{t} \cdot \beta I(t)/N}),
$$

$$
N_{E\rightarrow I}(t) = \textrm{Binom}(E(t),1-e^{-\Delta{t} \cdot \sigma}),
$$

$$
N_{I\rightarrow R}(t) = \textrm{Binom}(I(t),1-e^{-\Delta{t} \cdot \gamma }).
$$

A common COVID-19 model is a variation of this SEIR model that incorporates:

1. multiple identical stages of the infectious period, which allows us to model gamma-distributed durations of infectiousness, and
2. an infection rate modified by a 'mixing coefficient', $$\alpha \in [0,1]$$, which is a rough heuristic for the slowdown in disease spread that occurs in realistically heterogeneous populations where more well-connected individuals are infected first.

A three-stage infectious period model is given by

$$
S \xrightarrow{\beta S (I_1+I_2+I_3)^\alpha/N} E \xrightarrow{\sigma E} I_1 \xrightarrow{3\gamma I_1} I_2 \xrightarrow{3\gamma I_2} I_3 \xrightarrow{3\gamma I_3} R.
$$

The _flepiMoP_ model structure is specifically designed to make it simple to encode the type of more complex "stratified'' models that often arise in infectious disease dynamics. The following are some examples of possible stratifications.

#### Age groups

To describe an SEIR-type disease that spreads and progresses differently among children versus adults, one may want to repeat each compartment of the model for each of the two age groups (C – Children, A – Adults), creating an age-stratified model

$$
S_C \xrightarrow{S_C (\beta_{CC} I_C/N_C + \beta_{AC} I_A/N_A)} E_C \xrightarrow{\sigma_C E_C} I_C \xrightarrow{\gamma_C I_C} R_C,
$$

$$
S_A \xrightarrow{S_A (\beta_{AA} I_A/N_A + \beta_{CA} I_C/N_C)} E_A \xrightarrow{\sigma_A E_A} I_ A \xrightarrow{\gamma_A I_A} R_A,
$$

where $$\beta_{XY}$$ is the transmission rate between age X and Y, and we have assumed individuals do not age on the timescale relevant to the model.

#### Vaccination status

Vaccination status could influence disease progression and infectiousness, and could also change over time as individuals choose to get the vaccine (V – vaccinated, U – unvaccinated)

$$
S_U \xrightarrow{\beta S_U (I_U + I_V)/N} E_U \xrightarrow{\sigma_U E_U} I_U \xrightarrow{\gamma_U I_U} R_U,
$$

$$
S_V \xrightarrow{\beta (1-\theta) S_V (I_U + I_V)/N} E_V \xrightarrow{\sigma_V E_V} I_V \xrightarrow{\gamma_V I_V} R_V,
$$

$$
S_U \xrightarrow{\nu S_U} S_V,
$$

$$
R_U \xrightarrow{\nu R_U} R_V,
$$

where $$u$$ is the vaccination rate (we assume that individuals do not receive the vaccine while they are exposed or infectious) and $$\theta$$ is the vaccine efficacy against infection. Similar structures could be used for other sources of prior immunity or other dynamic risk groups.

#### Pathogen strain

Another common stratification would be pathogen strain, such as COVID-19 variants. Individuals may be infected with one of several variants, strains, or serotypes. Our framework can easily create multistrain models, for example

$$
S_A \xrightarrow{\beta_A S_A I_A/N_A} E_A \xrightarrow{\sigma_A E_A} I_ A \xrightarrow{\gamma_A I_A} R_A,
$$

$$
S_A \xrightarrow{\beta_B S_B I_B/N_B} E_B \xrightarrow{\sigma_B E_B} I_B \xrightarrow{\gamma_B I_B} R_B,
$$

$$
R_{A} \xrightarrow{\beta_B(1-\phi_{AB}) R_A I_B/N_B} E_{AB} \xrightarrow{\sigma_{AB} E_{AB}} I_{AB} \xrightarrow{\gamma_{AB} I_{AB}} R_{AB},
$$

$$
R_{B} \xrightarrow{\beta_A (1-\phi_{BA}) R_B I_A/N_B} E_{AB} \xrightarrow{\sigma_{AB} E_{AB}} I_{AB} \xrightarrow{\gamma_{AB} I_{AB}} R_{AB},
$$

where $$\phi_{AB}$$ is the immune cross-protection conferred from infection with strain A to subsequent infection with strain B. Co-infection is ignored. All individuals are assumed to be initially equally susceptible to both infections and are just categorized as $$S_A$$ (vs $$S_B$$) for convenience.

All combinations of these situations can be quickly specified in _flepiMoP_. Details on how to encode these models is provided in the [Model Implementation](model-implementation/) section, with examples given in the [Tutorials](https://github.com/HopkinsIDD/flepimop-documentation/blob/main/gitbook/gempyor/broken-reference/README.md) section.

### Clinical outcomes and observations model

The pipeline allows for an additional type of dynamic state variable beyond those included in the mathematical model. We refer to these extra variables as "Outcomes" or "Observations". Outcome variables can be functions of model variables, but do not feed back into the model by influencing other state variables. Typically, we use outcome variables to describe the process through which some subset of individuals in a compartment are "observed'' and become part of the data to which models are compared and attempt to predict. For example, in the context of a model for an infectious disease like COVID-19, outcome variables include reported cases, hospitalizations, and deaths.

An outcome variable $$H(t)$$ can be generated from a state variable of the mathematical model $$X(t)$$ using the following properties:

* The proportion of all individuals in $$X$$ who will be observed as $$H$$, $$p$$
* The delay between when an individual enters state $$X$$ and when they are observed as $$H$$, which can follow a class of probability distributions $$f(\Delta t;\theta)$$ where $$\theta$$ is the parameters of the distribution (e.g., the mean and standard deviation of a normal distribution)
* (optional) the duration spent in observable $$H$$, in which case the output will also contain the prevalence (number of individuals currently in $$H$$ in addition to the incidence into $$H$$

In addition to single values (drawn from a distribution), the duration and delay can be inputted as distributions, producing a convolution of the output.

The number of individuals in $$X$$ at time $$t_1$$ who become part of the outcome variable $$H(t_2)$$ is a random variable, and individuals who are observed in $$H$$ at time $$t$$ could have entered $$X$$ at different times in the past.

Formally, for a deterministic, continuous-time model

$$
H(t) = \int_{\tau} p X(\tau) f(t-\tau, \theta) d\tau
$$

For a discrete-time, stochastic model

$$
H(t) = \sum_{\tau_i=0}^{t}\text{Multinomial} (\text{Binomial}(X(\tau_i), p), \{f(t-\tau_i, \theta)\}).
$$

Note that outcomes $$H(t)$$ constructed in this way always represent _incidence_ values; meaning they describe the number of individuals newly entering this state at time $$t$$. If the model state $$X(t)$$ is also an incidence, then $$p$$ is a unitless probability, whereas if $$X(t)$$ is a _prevalence_ (number of individuals currently in state at time $$t$$), then $$p$$ is instead a probability per time unit.

Outcomes can also be constructed as functions of other outcomes. For example, a fraction of hospitalized patients may end up in the intensive care unit (ICU).

There are several benefits to separating outcome variables from the mathematical model. Firstly, these variables can be calculated **after** the model is run, and only at the timepoints of interest, which can dramatically reduce the memory needed during model simulation. Secondly, outcome variables can be fully stochastic even when the mathematical model is simulated deterministically. This becomes useful when an infection might be at high enough prevalence that a deterministic simulation is appropriate, but when there is a rare and therefore quite stochastic outcome reported in the data (e.g., severe cases) that the model is tasked with predicting. Thirdly, outcome variables can have arbitrary delay distributions, to take into account the complexities of health reporting practices, whereas our mathematical modeling framework is designed mainly for exponentially distributed delays and only easily permits extensions to gamma-distributed delays. Finally, this separation keeps the pipeline modular and allow for easy editing of one component of the model without disrupting the other.

Details on how to specify these outcomes in the model configuration files is provided in the [Model Implementation](model-implementation/) section, with examples given in the [Tutorials](https://github.com/HopkinsIDD/flepimop-documentation/blob/main/gitbook/gempyor/broken-reference/README.md) section.

### Population structure

The pipeline was designed specifically to simulate infection dynamics in a set of **connected subpopulations**. These subpopulations could represent geographic divisions, like **countries**, **states**, **provinces**, or **neighborhoods**, or **demographic groups**, or potentially even **different host species**. The equations and parameters of the transmission and outcomes models are repeated for each subpopulation, but the values of the **parameters can differ by location**. Within each subpopulation, infection is equally likely to spread between any pair of susceptible/infected individuals after accounting for their infection class, whereas between subpopulations there may be **varying levels of mixing**.

Formally, this type of population structure is often referred to as a “_metapopulation_”, and each subpopulation may be called a “_deme_”.

The following properties may be different between subpopulations:

* the population size
* the parameters of the transmission model (see LINK)
* the parameters of the outcomes model (see LINK)
* the amount of transmission that occurs within this subpopulation versus from any other subpopulation (see LINK)
* the timing and extent of any interventions that modify these parameters (see LINK)
* the initial timing and number of external introductions of infections into the population (see LINK)
* the ground truth timeseries data used to compare to model output and infer model parameters (see LINK)

Currently, the following properties must be the same across all subpopulations:

* the compartmental model structure
* the form of the likelihood function used to estimate parameters by fitting the model to data (LINK)
* ...

#### Mixing between subpopulations

The generalized compartmental model allows for second order “interaction” terms that describe transitions between model states that depend on interactions between pairs of individuals. For example, in the context of a classical SIR model, the rate of new infections depends on interactions between susceptible and infectious individuals and the transmission rate

$$
\frac{dI}{dt} = \beta S I - \gamma I
$$

For a model with multiple subpopulations, each of these interactions can occur either between individuals in the same or different subpopulations, with specific rate parameters for each combination of individual locations

$$
\frac{dI_i}{dt} = \sum_j \beta_{ji} I_j S_i - \gamma I_i
$$

where $$\beta_{ji}$$ is the per-contact per-time rate of disease transmission between an infected individual residing in subpopulation $$j$$ and a susceptible individual from subpopulation $$i$$.

In general for infection models in connected subpopulations, the transmission rates $$\beta_{ji}$$ can take on arbitrary values. In this pipeline, however, we impose an additional structure on these terms. We assume that interactions between subpopulations occur when individuals temporarily relocate to another subpopulation, where they interact with locals. We call this movement “mobility”, and it could be due to regular commuting, special travel, etc. There is a transmission rate ($$\beta_j$$) associated with each subpopulation $$j$$, and individuals physically in that subpopulation – permanently or temporarily – are exposed and infected with this local rate whenever they encounter local susceptible individuals.

The transmission matrix is then

$$
\beta_{ji} = \begin{cases} p_a \frac{M_{ij}}{N_i} \beta_j &\text{if } j \neq i \\ \left( 1- \sum_{j \neq i} p_a \frac{M_{ij}}{N_i} \right) &\text{if } j = i \end{cases}
$$

where $$\beta_j$$ is the onward transmission rate from infected individuals in subpopulation $$j$$, $$M_{ij}$$ is the number of individuals in subpopulation i who are interacting with individuals in subpopulation $$j$$ at any given time (for example, fraction who commute each day), and $$p_a$$ is a fractional scaling factor for the strength of inter-population contacts (for example, representing the fraction of hours in a day commuting individuals spend outside vs. inside their subpopulation).

The list of all pairwise mobility values and the interaction scaling factor are model input parameters. Details on how to specify them are given in the [Model Implementation](model-implementation/) section.

If an alternative compartmental disease model is created that has other interactions (second order terms), then the same mobility values are used to determine the degree of interaction between each pair of subpopulations.

### Initial conditions

Initial conditions can be specified by setting the values of the compartments in the disease transmission model at time zero, or the start of the simulation. For example, we might assume that for day zero of an outbreak the whole population is susceptible except for one single infected individual, i.e. $$S(0) = N-1$$ and $$I(0) = 1$$. Alternatively, we might assume that a certain proportion of the population has prior immunity from previous infection or vaccination.

It might also be necessary to model instantaneous changes in values of model variables at any time during a simulation. We call this 'seeding'. For example, individuals may import infection from other external populations, or instantaneous mutations may occur, leading to new variants of the pathogen. These processes can be modeled with seeding, allowing individuals to change state at specified times independently of model equations.

We also note that seeding can also be used as a convenient way to specify initial conditions, particularly ealy in an outbreak where the outbreak is triggered by a few 'seedings'.

### Time-dependent interventions

Parameters in the disease transmission model or the observation model may change over time. These changes could be, for example: environmental drivers of disease seasonality; “non-pharmaceutical interventions” like social distancing, isolation policies, or wearing of personal protective equipment; “pharmaceutical interventions” like vaccination, prophylaxis, or therapeutics; changes in healthcare seeking behavior like testing and diagnosis; changes in case reporting, etc.

The model allows for any parameter of the disease transmission model or the observation model to change to a new value for a time interval specified by start and end times (or multiple start and end times, for interventions that are recurring). Each change may be subpopulation-specific or apply to the entire population. Changes may be overlapping in time.

The magnitude of these changes are themselves model parameters, and thus may [inferred](https://github.com/HopkinsIDD/flepimop-documentation/blob/main/gitbook/gempyor/broken-reference/README.md) along with other parameters when the model is fit to data. Currently, the start and end times of interventions must be fixed and cannot be varied or inferred.

For example, the rate of transmission in subpopulation $$i$$, $$\beta_i$$, may be reduced by an intervention $$r_k$$that acts between times $$t_{k,\text{start}}$$ and $$t_{k,\text{end}}$$, and another intervention $$r_l$$ that acts between times $$t_{l,\text{start}}$$and $$t_{l,\text{end}}$$

$$
\beta_j'(t) = (1-r_k(t;t_{k,\text{start}},t_{k,\text{end}}))(1-r_l(t;t_{l,\text{start}},t_{l,\text{end}})))\beta_j^0
$$

In this case, $$r_k(t)$$ and $$r_l(t)$$ are both considered simple **`SinglePeriodModifier`** interventions. There are four possible types of interventions that can be included in the model

* **`SinglePeriodModifier`** - an intervention $$r_j$$ that leads to a fractional reduction in a parameter value in subpopulation $$j$$ (i.e., $$\beta_j$$) between two timepoints

$$
\beta_j'(t) = (1-r_j(t;t_{j,\text{start}},t_{j,\text{end}}))\beta_j^0
$$

$$
r_j(t;t_{j,\text{start}},t_{j,\text{end}}) = \begin{cases} r_j &\text{if } t_{j,\text{start}} < t <t_{j,\text{end}} \\ 0 &\text{otherwise} \end{cases}
$$

* **`MultiPeriodModifier`** - an intervention $$r_j$$ that leads to a fractional reduction in a parameter value in subpopulation $$j$$ (i.e., $$\beta_j$$) value between multiple sets of timepoints

$$
\beta_j'(t) = (1-r_j(t; \{t_{j,k,\text{start}},t_{j,k,\text{end}}\}_k))\beta_j^0
$$

$$
r_j(t;\{t_{j,k,\text{start}},t_{j,k,\text{end}}\}_k) = \begin{dcases} r_ j&\text{if } t_{j,k1,\text{start}} < t <t_{j,k1,\text{end}} \\ r_j &\text{if } t_{j,k2,\text{start}} < t <t_{j,k2,\text{end}} \\ & ... \\ r_j &\text{if } t_{j,kn,\text{start}} < t <t_{j,kn,\text{end}} \\ 0 &\text{otherwise} \end{dcases}
$$

* **`ModifierModifier`**- an intervention $$\pi_j$$ that leads to a fractional reduction in the value of another intervention $$r_j$$ between two timepoints

$$
\beta_j'(t) = (1-r_j(t;t_{j,\text{start}},t_{j,\text{end}})(1-\pi_{r,j}(t;t_{r,j,\text{start}},t_{r,j,\text{end}})))\beta_j^0
$$

$$
\pi_{r,j}(t;t_{r,j,\text{start}},t_{r,j,\text{end}}) = \begin{cases} \pi_{r,j} &\text{if } t_{r,j,\text{start}} < t <t_{r,j,\text{end}} \\ 0 &\text{otherwise} \end{cases}
$$

* **`StackedModifier`** - TBA
