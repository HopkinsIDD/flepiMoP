# File descriptions

## _flepiMoP_

[https://github.com/HopkinsIDD/flepiMoP](https://github.com/HopkinsIDD/flepiMoP)

Current branch: `main`

This repository contains all the code underlying the mathematical model and the data fitting procedure, as well as ...

To actually run the model, this repository folder must be located inside a location folder (e.g. `COVID19_USA`) which contains additional files describing the specifics of the model to be run (i.e. the config file), all the necessary input data (i.e. the population structure), and any data to which the model will be fit (i.e. cases and death counts each day)

### **/gempyor\_pkg**

This directory contains the core Python code that creates and simulates generic compartmental models and additionally simulates observed variables. This code is called `gempyor` for **General Epidemics Modeling Pipeline with Yterventions and Outcome Reporting.** The code in gempyor is called from R scripts (see **/main\_scripts** and **/R** sections below) that read the config, run the model simulation via gempyor as required, read in data, and run the model inference algorithms.

* `pyproject.toml` - contains the build system requirements and dependencies for the gempyor package; used during package installation
* `setup.cfg` - contains information used by Python's `setuptools` to build the `gempyor` package. Contains the definitions of command line shortcuts for running simulations directly from `gempyor` (bypassing R interface) if desired

#### **/gempyor\_pkg/src/gempyor/**

* seir.py - Contains the core code for simulating the mathematical model. Takes in the model definition and parameters from the config, and outputs a file with a timeseries of the value of each state variable (# of individuals in each compartment)
* simulate\_seir.py -
* steps\_rk.py -
* steps\_source.py -
* outcomes.py - Contains the core code for generating the outcome variables. Takes in the output of the mathematical model and parameters from the config, and outputs a file with a timeseries of the value of each outcome (observed) variable
* simulate\_outcomes.py -
* setup.py
* file\_paths.py -
* compartments.py
* parameters.py
* results.py
* seeding\_ic.py
* /NPI/
  * base.py -
  * SinglePeriodModifier.py -
  * MultiPeriodModifier.py -
  * SinglePeriodModifierInterven.py -
* /dev - contains functions that are still in development
* /data - ?

#### **/gempyor\_pkg/docs**

Contains notebooks with some `gempyor`-specific documentation and examples

* Rinterface.Rmd - And R notebook that provides some background on `gempyor` and describes how to run it as a standalone package in python, without the R wrapper scripts or the Docker.
* Rinterface.html - HTML output of Rinterface.Rmd

### **/R**

### **/main\_scripts**

This directory contains the R scripts that takes the specifications in the configuration file and sets up the model simulation, reads the data, and performs inference.

* inference\_main.R - This is the master R script used to run the model. It distributes the model runs across computer cores, setting up runs for all the scenarios specified in the config, and for each model iteration used in the parameter inference. Note that despite the name "inference" in this file, this script must be used to run the model even if no parameter inference is conducted
* inference\_slot.R - This script contains the main code of the inference algorithm.
* create\_seeding.R -

### **/R\_packages**

This directory contains the core R code - organized into functions within packages - that handle the model setup, data pulling and processing, conducting parameter inference for the model, and manipulating model output.

* **flepicommon**
  * config.R
  * DataUtils.R
  * file\_paths.R
  * safe\_eval.R
  * compartments.R
* **inference** - contains code to
  * groundtruth.R - contains functions for pulling ground truth data from various sources. Calls functions in the `flepicommon` package
  * functions.R - contains many functions used in running the inference algorithm
  * inference\_slot\_runner\_funcs.R - contains many functions used in running the inference algorithm
  * inference\_to\_forecast.R -
  * documentation.Rmd - Summarizes the documentation relevant to the inference package, including the configuration file options relevant to model fitting
  * InferenceTest.R -
  * /tests/ -
* **config.writer**
  * create\_config\_data.R
  * process\_npi\_list.R
  * yaml\_utils.R
* **report.generation**
  * DataLoadFuncs.R
  * ReportBuildUtils.R
  * ReportLoadData.R
  * setup\_testing\_environment.R

####

####

### /test

### /data

Depreciated? Should be removed

### /vignettes

Depreciated? Should be removed

### /doc

Depreciated? Should be removed

### /batch

### /slurm\_batch

## COVID19\_USA Repository

[https://github.com/HopkinsIDD/COVID19\_USA](https://github.com/HopkinsIDD/COVID19/\_USA)

Current branch: `main`

### **/R**

Contains R scripts for generating model input parameters from data, writing config files, or processing model output. Most of the files in here are historic (specific to a particular model run) and not frequently used. Important scripts include:

* get\_vacc\_rate\_and\_outcomes\_R13.R - this pulls vaccination coverage and variant prevalence data specific to rounds (either empirical, or specified by the scenario), and adjusts these data to the formats required for the model. Several data files are created in this process: variant proportions for each scenario, vaccination rates by age and dose. A file is also generated that defines the outcome ratios (taking in to account immune escape, cross protection and VE).

#### **/R/scripts/config\_writers**

Scripts to generate config files for particular submissions to the Scenario Modeling Hub. Most of this functionality has now been replaced by the config writer package ()

#### **R/scripts/postprocess**

Scripts to process the output of model runs into data formats and plots used for Scenario Modeling Hub and Forecast Hub. These scripts pull runs from AWS S3 buckets and processes and formats them to specifications for submissions to Scenario Modeling Hubs, Forecast Hubs and FluSight. These formatted files are saved and the results visualized. This script uses functions defined in /COVIDScenarioPipeline/R/scripts/postprocess.

* run\_sum\_processing.R

### **/data**

Contains data files used in parameterizing the model for COVID-19 in the US (such as creating the population structure, describing vaccine efficacy, describing parameter alterations due to variants, etc). Some data files are re-downloaded frequently using scripts in the pipeline (us\_data.csv) while others are more static (geodata, mobility)

Important files and folders include

* geodata.csv
* geodata\_2019\_statelevel.csv
* mobility.csv
* mobility\_territories\_2011-2015\_statelevel.csv
* outcomes\_ratios.csv
* US\_CFR\_shift\_dates\_v3.csv
* US\_hosp\_ratio\_corrections.cs
* seeding\_agestrat\_RX.csv

#### **/data/shp**

"Shape-files" (.shp) that .....

#### **/data/outcomes**

* usa-subpop-params-output\_V2.parquet

**/data/intervention\_tracking**

Data files containing the dates that different non pharmaceutical interventions (like mask mandates, stay-at-home orders, school closures) were implemented by state

#### **/data/vaccination**

Files used to create the config elements related to vaccination, such as vaccination rates by state by age and vaccine efficacy by dose

**/data/variant**

Files created in the process of downloading and analyzing data on variant proportions

### **/manuscripts**

Contains files for scientific manuscripts using results from the pipeline. Not up to date

### **/config**

Contains an archive of configuration files used for previous model runs

### **/old\_configs**

Same as above. Contains an archive of configuration files used for previous model runs

### **/scripts**

Depreciated - to be removed? - contains rarely used scripts

### **/notebook**

Depreciated - to be removed? - contains rarely used notebooks to check model input. Might be used in some unit tests?

### **/NPI**

empty?

### **/ScenarioHub**

Depreciated - to be removed?
