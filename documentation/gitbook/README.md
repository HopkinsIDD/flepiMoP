# Home

Welcome to _flepiMoP_ documentation!

The “**FLexible EPIdemic MOdeling Pipeline**” (_flepiMoP_; formerly known as the _COVID Scenario Modeling Pipeline_ or _CSP_) is an open-source software suite designed by researchers in the [Johns Hopkins Infectious Disease Dynamics Group](http://www.iddynamics.jhsph.edu/) and at [UNC Chapel Hill ](https://sph.unc.edu/epid/epidemiology-landing/)to simulate a wide range of compartmental models of infectious disease transmission. The disease transmission and observation models are defined by a no-code configuration file, which allows models of varying complexity to be specified quickly and consistently, from simple problems described by SIR-style models in a single population to more complicated models of multiple pathogen strains transmitting between thousands of connected spatial divisions and age groups.

It was initially designed in early 2020 and was routinely used to provide projections of the emerging COVID-19 epidemic to health authorities worldwide. Currently, _flepiMoP_ provides COVID-19 projections to the US CDC-funded model aggregation sites, the [COVID-19 Forecast Hub](https://covid19forecasthub.org/) and the [COVID-19 Scenario Modeling Hub](https://covid19scenariomodelinghub.org/), influenza projections to [FluSight ](https://www.cdc.gov/flu/weekly/flusight/index.html)and to the [Flu Scenario Modeling Hub](https://fluscenariomodelinghub.org), and RSV projections to the [RSV Scenario Modeling Hub](https://rsvscenariomodelinghub.org/).

However, the pipeline is much more general and can be used to simulate the dynamics of any infection that can be expressed as a [compartmental epidemic model](https://en.wikipedia.org/wiki/Compartmental\_models\_in\_epidemiology). These include applications in chemical reaction kinetics, pharmacokinetics, within-host disease dynamics, or applications in the social sciences.

In addition to producing forward simulations given a specified model and parameter values, the pipeline can also attempt to optimize unknown parameters (e.g., transmission rate, case detection rate, intervention efficacy) to fit the model to datasets the user provides (e.g., hospitalizations due to severe disease) using a Bayesian inference framework. This feature allows the pipeline to be utilized for short-term forecasting or longer-term scenario projections for ongoing epidemics, since it can simultaneously be fit to data for dates in the past and then use best-fit parameters to make projections into the future.

### General description of _flepiMoP_

The main features of _flepiMoP_ are:

* Open-source (GPL v3.0) infectious dynamics modeling software, written in R and Python
* Versatile, no-code design applicable for most compartmental models and outcome observation models, allowing for quick iteration in reaction to epidemic events (e.g., emergence of new variants, vaccines, non-pharmaceutical interventions (NPIs))
* Powerful, just-in-time compiled disease transmission model and distributed inference engine ready for large scale simulations on high-performance computing clusters or cloud workflows
* Adapted to small- and large-scale problems, from a simple SIR model to a complex model structure with hundreds of compartments on thousands of connected populations
* Strong emphasis on mechanistic processes, with a design aimed at leveraging domain knowledge in conjunction with statistical inference
* Portable for Windows WSL, MacOS, and Linux with the provided Docker image and an Anaconda environment

<figure><img src=".gitbook/assets/CSP Overview.png" alt=""><figcaption><p>Overview of the pipeline organization</p></figcaption></figure>

The mathematical model within the pipeline is a _compartmental epidemic model_ embedded within a _well-mixed metapopulation_. A compartmental epidemic model is a model that divides all individuals in a population into a discrete set of states (e.g. “infected”, “recovered”) and tracks – over time – the number of individuals in each state and the rates at which individuals transition between these states. The well-known SIR model is a classic example of such a model, and much more complex versions of this model type have been simulated with this framework (for example, an SEIR-style model in which individuals are further subdivided into multiple age groups and vaccination statuses).

The structure of the desired model, as well as the parameter values and initial conditions, can be specified flexibly by the user in a no-code fashion. The pipeline allows for parameter values to change over time at discrete intervals, which can be used to specify time-dependent aspects of disease transmission and control (such as seasonality or vaccination campaigns).

The model is embedded within a meta-population structure, which consists of a series of distinct subpopulations (e.g. states, provinces, or other communities) in which the model structure is repeated, albeit with potentially different parameter values. The subpopulations can interact, either through the movement of individuals or the influence of individuals in one subpopulation on the transition rate of individuals in another ;

Within each subpopulation, the population is assumed to be well-mixed, meaning that interactions are assumed to be equally likely between any pair of individuals (since unique identities of individuals are not explicitly tracked). The same model structure can be simulated in a continuous-time deterministic or discrete-time stochastic manner ;

In addition to the variables described by the compartmental model, the model can track other observable variables (“outcomes”) that are functions of the basic model variables but do not themselves influence the dynamics (i.e., some portion of infections are reported as cases, depending on a testing rate). The model can be run iteratively to tune the values of certain parameters so that these outcome variables best match timeseries data provided by the user for a certain time period ;

Fitting is done using a Bayesian-like framework, where the user can specify the likelihood of observed outcomes in data given modeled outcomes, and the priors on any parameters to be fit. Multiple data streams (e.g., cases and deaths) can be fit simultaneously. A custom Markov Chain Monte Carlo method is used to sequentially propose and accept or reject parameter values based on the model fit to data, in a way that balances fit quality within each individual subpopulation with that of the total aggregate population, and that takes advantage of parallel computing environments.

The code is written in a combination of [R](https://www.r-project.org/) and [Python](https://www.python.org/), and the vast majority of users only need to interact with the pipeline via the components written in R. It is structured in a modular fashion, such that individual components – such as the epidemic model, the observable variables, the population structure, or the parameters – can be edited or completely replaced without any handling of other parts of the code ;

When model simulation is combined with fitting to data, the code is designed to run most efficiently on a supercomputing cluster with many cores. We most commonly run the code on [Amazon Web Services](https://aws.amazon.com/) or on high-performance computers using SLURM. However, even relatively large models can be run efficiently on most personal computers. Typically, the memory of the machine will limit the number of compartments (i.e., variables) that can be included in the epidemic model, while the machine’s CPU will determine the speed at which each model run is completed and the number of iterations of the model that can be run during parameter searches when fitting the model to data. While the pipeline can be installed on any computer, it is sometime easier to use an Anaconda environment or the provided [Docker](https://www.docker.com/) container, where all the software dependencies (e.g., standardized R and Python versions along with required packages) are included, independent of the user’s local machine. All the code is maintained on [our GitHub](https://github.com/HopkinsIDD/flepiMoP) and shared with the GNU General Public License v3.0 license. It is build on top of a fully open-source software stack.

This documentation is organized as follows. The [Model Description](gempyor/model-description.md) section describes the mathematical framework for the compartmental epidemic models that can be simulated forward in time by the pipeline. The [Model Inference](model-inference/inference-description.md) section describes the statistical framework for fitting the model to data. The [Data and Parameter](broken-reference) section describes the inputs the user must provide to the pipeline, in terms of the model structure and parameters, the population characteristics, the initial conditions, time-varying interventions, data to be fit, and more. The [How to Run](broken-reference) section provides concrete guidance on setting up and running the model and analyzing the output. The [Quick Start Guide](how-to-run/quick-start-guide.md) provides a simple example model setup. The [Advanced](how-to-run/advanced-run-guides/) section goes into more detail on specific features of the model and the code that are likely to only be of interest to users who want to run more complex models or data fitting routines or substantially edit the code. It includes a subsection describing each file and package used in the pipeline and their interactions during a model run.

Users who wish to jump to running the model themselves can see [Quick Start Guide](how-to-run/quick-start-guide.md).

For questions about the pipeline or to report a bug, please use the “Issues” or "Discussions" feature on [our GitHub](https://github.com/HopkinsIDD/flepiMoP).

### Acknowledgments

_flepiMoP_ is actively developed by its current contributors, including Joseph C Lemaitre, Sara L Loo, Emily Przykucki, Clifton McKee, Claire Smith, Sung-mok Jung, Koji Sato, Pengcheng Fang, Erica Carcelen, Alison Hill, Justin Lessler, and Shaun Truelove, affiliated with the ;

* Department of Epidemiology, Gillings School of Global Public Health, University of North Carolina at Chapel Hill, Chapel Hill, NC, USA for (JCL, JL)
* Johns Hopkins University International Vaccine Access Center, Department of International Health, Baltimore, MD, USA for (SLL, KJ, EC, ST)
* Department of Epidemiology, Johns Hopkins Bloomberg School of Public Health, Baltimore, Maryland, USA for (CM, CS, JL, ST)
* Carolina Population Center, University of North Carolina at Chapel Hill, Chapel Hill, NC, USA for (S-m.J, JL)
* Institute for Computational Medicine, Johns Hopkins University, Baltimore, MD, USA for (AH).

The development of this model was supported by from funds the National Science Foundation (`2127976`; ST, CPS, JK, ECL, AH), Centers for Disease Control and Prevention (`200-2016-91781`; ST, CPS, JK, AH, JL, JCL, SL, CM, EC, KS, S-m.J), US Department of Health and Human Services / Department of Homeland Security (ST, CPS, JK, ECL, AH, JL), California Department of Public Health (ST, CPS, JK, ECL, JL), Johns Hopkins University (ST, CPS, JK, ECL, JL), Amazon Web Services (ST, CPS, JK, ECL, AH, JL, JCL), National Institutes of Health (`R01GM140564`; JL, `5R01AI102939`; JCL), and the Swiss National Science Foundation (`200021-172578`; JCL)

We need to also acknowledge past contributions to the development of the COVID Scenario Pipeline, which evolved into _flepiMoP_. These include contributions by Heramb Gupta, Kyra H. Grantz, Hannah R. Meredith, Stephen A. Lauer, Lindsay T. Keegan, Sam Shah, Josh Wills, Kathryn Kaminsky, Javier Perez-Saez, Joshua Kaminsky, and Elizabeth C. Lee.
