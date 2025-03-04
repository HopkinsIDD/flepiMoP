---
description: >-
  More Indepth introduction to using flepiMoP.
  Executing a more realistic analysis with several stages.
---

# Indepth Tutorial

This tutorial covers using flepiMoP with a relatively simple model system, but with some realistic complexity to the analysis around that model.

## Installation

### Ensure access to `git`, `R`, `python`, and `conda`

 - conda set up on personal machine
 - WSL or linux setup

### Ensure access to an HPC

 - including setting up ssh hosts file

### Install `flepiMoP` on local machine

 - follow local install script

### Install `flepiMoP` on HPC

 - follow hpc install script

### Create a new project

 - create a new repo with proper file structure
 - initialize environmental variables?

<!--
### TODO, Idealized Version

```bash
curl some file from flepimop repo
./install-flepimop ## creates env variables, reference conda env, etc
# if necessary, ssh setup according to directions
ssh hpctarget
curl some file from flepimop repo ## again, creates env variables, conda env, etc; also sets up module loading config
./install-flepimop
exit
# if you've previous used flepimop, might instead
# flepimop update && ssh hpctarget flepimop update
flepimop init new_repo_directory && cd new_repo_directory
# later, will move items from new_repo_directory to HPC
```
-->

## HIV in Harare Analysis

In this tutorial, we're going to be reproducing and extending a model development and analysis original created for the [Meaningful Modelling of Epidemiological Data (MMED)](https://www.ici3d.org/MMED) workshop. This exercise concerns the emergence of HIV in several sub-Saharan African countries, and has you calculate the impact of intervention scenarios that vary in terms of effort and start time.

Through the exercise, you will:

 - configure an initial model, run it, and compare the results to data
 - consider alternative configurations, until settling on a model
 - troubleshoot formally fitting the model
 - ... across multiple locations
 - ... on the HPC
 - add interventions to the model
 - project intervention scenarios, using fitting results
 - generate a scenario analysis from results

### Looking at the Data

For any analysis, we should start by looking at our data. For this exercise, the data concerns the emergence of HIV in sub-Saharan Africa.

 - get data from R package
 - introduce flepimop project structural convention: model_input, model_output, and data directories
 - create a notebook to view the data
 - render the notebook with flepimop

### First Model: Deterministic XYZ

 - create a config
 - flepimop simulate
 - interacting with flepimop output structure
 - update notebook, re-render
  * add simulate time series to data series

### First Model: Sampled Parameters XYZ

 - update config
 - flepimop simulate with multiple runs
 - more interacting with flepimop output structure: samples
 - update notebook, re-render
  * add sample time series to get spaghetti plot
  * add parameter distribution plots

### Additional Models

 - update config to add extra model structures as scenarios
 - flepimop simulate in scenario mode
 - more interacting with flepimop output structure: scenarios
 - update notebook, re-render
  * expand plots to reflect different models

### Fitting

 - update config to add inference section
 - flepimop calibrate (for "best" model only? with model selection analysis?)
 - more interacting with flepimop output structure: calibration outputs
 - update notebook:
  * change from random sample of parameters to fitting results
  * update parameter distribution plots to reflect likelihood information
  * add fitting diagnostic plots

### Introduce an Intervention

 - update config to consider an intervention (rough idea: 90/90/90)
 - flepimop simulate (of the posterior, in the fitting interval)
 - flepimop simulate (in projection mode, continuations)
 - more interacting with flepimop output structure: continuations
 - update notebook:
  * add scenario projection plot

### Intervention Scenarios

 - update config to consider multiple interventions (different levels of 90/90/90; also a different kind of intervention, say vaccination?)

### Define environment variables (optional)

Since you'll be navigating frequently between the folder that contains your project code and the folder that contains the core flepiMoP model code, it's helpful to define shortcuts for these file paths. You can do this by creating environmental variables that you can then quickly call instead of writing out the whole file path.

For example, if you're on a **Mac** or Linux/Unix based operating system and storing the `flepiMoP` code in a directory called `Github`, you define the FLEPI\_PATH and PROJECT\_PATH environmental variables to be your directory locations as follows:

```bash
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepiMoP/examples/tutorials
```

or, if you have already navigated to your flepiMoP directory

```bash
export FLEPI_PATH=$(pwd)
export PROJECT_PATH=$(pwd)/examples/tutorials
```

You can check that the variables have been set by either typing `env` to see all defined environmental variables, or typing `echo $FLEPI_PATH` to see the value of `FLEPI_PATH`.

If you're on a **Windows** machine:

<pre class="language-bash"><code class="lang-bash"><strong>set FLEPI_PATH=C:\Users\YourName\Github\flepiMoP
</strong>set PROJECT_PATH=C:\Users\YourName\Github\flepiMoP\examples\tutorials
</code></pre>

or, if you have already navigated to your flepiMoP directory

<pre class="language-bash"><code class="lang-bash"><strong>set FLEPI_PATH=%CD%
</strong>set PROJECT_PATH=%CD%\examples\tutorials
</code></pre>

You can check that the variables have been set by either typing `set` to see all defined environmental variables, or typing `echo $FLEPI_PATH$` to see the value of `FLEPI_PATH`.

{% hint style="info" %}
If you choose not to define environment variables, remember to use the full or relative path names for navigating to the right files or folders in future steps.
{% endhint %}

## 🚀 Run the code

Everything is now ready 🎉 .

The next step depends on what sort of simulation you want to run: One that includes inference (fitting model to data) or only a forward simulation (non-inference). Inference is run from R, while forward-only simulations are run directly from the Python package `gempyor`.

First, navigate to the `PROJECT_PATH` folder and make sure to delete any old model output files that are there:

```bash
cd $PROJECT_PATH        # goes to your project repository
rm -r model_output/     # delete the outputs of past run if there are
```

For the following examples we use an example config from _flepimop\_sample_, but you can provide the name of any configuration file you want.

To get started, let's start with just running a forward simulation (non-inference).

### Non-inference run

Stay in the `PROJECT_PATH` folder, and run a simulation directly from forward-simulation Python package `gempyor`. Call `flepimop simulate` providing the name of the configuration file you want to run. For example here, we use `config_sample_2pop.yml`.

```
flepimop simulate config_sample_2pop.yml
```

This will produce a `model_output` folder, which you can look at using provided post-processing functions and scripts.

We recommend using `model_output_notebook.Rmd` as a starting point to interact with your model outputs. First, modify the YAML preamble in the notebook (make sure the configuration file listed matches the one used in simulation), then knit this markdown. This will produce plots of the prevalence of infection states over time. You can edit this markdown to produce any figures you'd like to explore your model output.

For your first `flepiMoP` run, it's better to run each command individually as described above to be sure each exits successfully. However, eventually you can **put all these steps together in a script**, seen below:

```
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepiMoP/examples/tutorials
cd $PROJECT_PATH
rm -rf model_output
flepimop simulate config.yml
```

Note that you only have to re-run the installation steps once each time you update any of the files in the flepimop repository (either by pulling changes made by the developers and stored on Github, or by changing them yourself). If you're just running the same or different configuration file, just repeat the final steps:

```
rm -rf model_output
flepimop simulate new_config.yml
```

### Inference run

An inference run requires a configuration file that has the `inference` section. Stay in the `$PROJECT_PATH` folder, and run the inference script, providing the name of the configuration file you want to run. For example here, we use `config_sample_2pop_inference.yml`.

{% code overflow="wrap" %}
```bash
flepimop-inference-main -c config_sample_2pop_inference.yml
```
{% endcode %}

This will run the model and create a lot of output files in `$PROJECT_PATH/model_output/`.

The last few lines visible on the command prompt should be:

> \[\[1]]
>
> \[\[1]]\[\[1]]
>
> \[\[1]]\[\[1]]\[\[1]]
>
> NULL

If you want to quickly do runs with options different from those encoded in the configuration file, you can do that from the command line, for example

```bash
flepimop-inference-main -j 1 -n 1 -k 1 -c config_inference.yml
```

where:

* `n` is the number of parallel inference slots,
* `j` is the number of CPU cores to use on your machine (if `j` > `n`, only `n` cores will actually be used. If `j` <`n`, some cores will run multiple slots in sequence)
* `k` is the number of iterations per slots.

Again, it is helpful to run the model output notebook (`model_output_notebook.Rmd` to explore your model outputs. Knitting this file for an inference run will also provide an analysis of your fits: the acceptance probabilities, likelihoods overtime, and the fits against the provided ground truth.

For your first `flepiMoP` inference run, it's better to run each command individually as described above to be sure each exits successfully. However, eventually you can **put all these steps together in a script**, seen below:

```bash
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepiMoP/examples/tutorials
cd $FLEPI_PATH
pip install --no-deps -e flepimop/gempyor_pkg/
Rscript build/local_install.R
cd $PROJECT_PATH
rm -rf model_output
flepimop-inference-main -c config_inference.yml
```

Note that you only have to re-run the installation steps once each time you update any of the files in the flepimop repository (either by pulling changes made by the developers and stored on Github, or by changing them yourself). If you're just running the same or different configuration file, just repeat the final steps

```
rm -rf model_output
flepimop-inference-main -c config_inference_new.yml
```

## 📈 Examining model output

If your run is successful, you should see your output files in the model\_output folder. The structure of the files in this folder is described in the [Model Output](../gempyor/output-files.md) section. By default, all the output files are .parquet format (a compressed format which can be imported as dataframes using R's arrow package `arrow::read_parquet` or using the free desktop application [Tad ](https://www.tadviewer.com/) for quick viewing). However, you can add the option `--write-csv` to the end of the commands to run the code (e.g.,  `flepimop simulate --write-csv config.yml`) to have everything saved as .csv files instead ;

## 🪜 Next steps

These configs and notebooks should be a good starting point for getting started with flepiMoP. To explore other running options, see [How to run: Advanced](advanced-run-guides/).
