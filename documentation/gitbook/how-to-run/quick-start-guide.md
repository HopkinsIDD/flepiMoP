---
description: >-
  Instructions to get started with using gempyor and flepiMoP, using some
  provided example configs to help you.
---

# Quick Start Guide

## 🧱 Set up&#x20;

### Access model files

Follow all the steps in the [Before any run](before-any-run.md) section to ensure you have access to the correct files needed to run your custom model or a sample model with flepiMoP.&#x20;

Take note of the location of the directory on your local computer where you cloned the flepiMoP model code (which we'll call `FLEPI_PATH`), as well as where you cloned your project files (which we'll call `PROJECT_PATH`). &#x20;

{% hint style="info" %}
For example, if you cloned your Github repositories into a local folder called `Github` and are using [_flepimop\_sample_](https://github.com/HopkinsIDD/flepimop\_sample) as a project repository, your directory names could be\
\
_**On Mac:**_&#x20;

/Users/YourName/Github/flepiMoP

/Users/YourName/Github/flepimop\_sample\
\
_**On Windows:**_ \
C:\Users\YourName\Github\flepiMoP

C:\Users\YourName\Github\flepimop\_sample
{% endhint %}

### Define environment variables (optional)

Since you'll be navigating frequently between the folder that contains your project code and the folder that contains the core flepiMoP model code, it's helpful to define shortcuts for these file paths. You can do this by creating environmental variables that you can then quickly call instead of writing out the whole file path.

If you're on a **Mac** or Linux/Unix based operating system, define the FLEPI\_PATH and PROJECT\_PATH environmental variables to be your directory locations, for example

```bash
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepimop_sample
```

or, if you have already navigated to your parent directory

```bash
export FLEPI_PATH=$(pwd)/flepiMoP
export PROJECT_PATH=$(pwd)/flepimop_sample
```

You can check that the variables have been set by either typing `env` to see all defined environmental variables, or typing `echo $FLEP_PATH` to see the value of `FLEPI_PATH`.&#x20;

If you're on a **Windows** machine

<pre class="language-bash"><code class="lang-bash"><strong>set FLEPI_PATH=C:\Users\YourName\Github\flepiMoP
</strong>set PROJECT_PATH=C:\Users\YourName\Github\flepimop_sample
</code></pre>

or, if you have already navigated to your parent directory

<pre class="language-bash"><code class="lang-bash"><strong>set FLEPI_PATH=%CD%\flepiMoP
</strong>set PROJECT_PATH=%CD%\flepimop_sample
</code></pre>

You can check that the variables have been set by either typing `set` to see all defined environmental variables, or typing `echo $FLEP_PATH$` to see the value of `FLEPI_PATH`.&#x20;

{% hint style="info" %}
If you don't want to bother defining the environmental variables, it's no problem, just remember to use the full or relative path names for navigating to the right files or folders in future steps
{% endhint %}

### Install packages

The code is written in a combination of [R](https://www.r-project.org/) and [Python](https://www.python.org/). The Python part of the model is a package called [_gempyor_](../gempyor/model-description.md), and includes all the code to simulate the epidemic model and the observational model and apply time-dependent interventions. The R component conducts the (optional) parameter inference, and all the (optional) provided pre and post processing scripts are also written in R. Most uses of the code require interacting with components written in both languages, and thus making sure that both are installed along with a set of required packages. However, Python alone can be used to do forward simulations of the model using _gempyor_.&#x20;

First, ensure you have python and R installed. You need a working python3.7+ installation. We recommend using the latest stable python release (python 3.12) to benefit from huge speed-ups and future-proof your installation. We also recommend installing Rstudio to interact with the R code and for exploring your model outputs.

{% hint style="info" %}
_**On Mac**_ 🍏

Python 3 is installed by default on recent MacOS installation. If it is not, you might want to check [homebrew](https://brew.sh/) and install the appropriate installation.&#x20;
{% endhint %}

#### Install packages

To install the python portions of the code (_gempyor_ ) and all the necessary dependencies, go to the flepiMoP directory and run the installation, using the following commands:

```bash
cd $FLEPI_PATH
pip install -e flepimop/gempyor_pkg/ # Install Python package gempyor (and dependencies)
```

{% hint style="danger" %}
_**A warning for Windows**_\
Once _gempyor_ is successfully installed locally, you will need to make sure the executable file `gempyor-seir.exe` is runnable via command line. To do this, you will need to add the directory where it was created to PATH. Follow the instructions [here](https://techpp.com/2021/08/26/set-path-variable-in-windows-guide/) to add the directory where this .exe file is located to PATH. This can be done via GUI or CLI.
{% endhint %}

If you just want to [run a forward simulation](quick-start-guide.md#non-inference-run), installing python's _gempyor_ is all you need.

To [run an inference run](quick-start-guide.md#inference-run) and to explore your model outputs using provided post-processing functionality, there are some packages you'll need to **install in R**. Open your **R terminal** (at the bottom of RStudio, or in the R IDE), and run the following command to install the necessary R packages:

<pre class="language-r" data-overflow="wrap"><code class="lang-r"><strong># while in R
</strong><strong>install.packages(c("readr","sf","lubridate","tidyverse","gridExtra","reticulate","truncnorm","xts","ggfortify","flextable","doParallel","foreach","optparse","arrow","devtools","cowplot","ggraph"))
</strong></code></pre>

This step does not need to be repeated unless you use a new computer or delete and reinstall R.&#x20;

Now return to your system terminal. To install _flepiMop_-internal R packages, run the following from command line:

{% code overflow="wrap" %}
```bash
Rscript build/local_install.R  # Install R packages
```
{% endcode %}

Each installation step may take a few minutes to run.&#x20;

{% hint style="info" %}
Note: These installations take place on your local operating system. You will need an active internet connection for installation, but not for other steps of running the model. \
If you are concerned about disrupting your base environment that you use for other projects, try using [Docker](advanced-run-guides/running-with-docker-locally.md) or [conda](advanced-run-guides/quick-start-guide-conda.md) instead. &#x20;
{% endhint %}

## 🚀 Run the code

Everything is now ready 🎉 .&#x20;

The next step depends on what sort of simulation you want to run: One that includes inference (fitting model to data) or only a forward simulation (non-inference). Inference is run from R, while forward-only simulations are run directly from the Python package `gempyor`.

First, navigate to the project folder and make sure to delete any old model output files that are there.

```bash
cd $PROJECT_PATH       # goes to your project repository
rm -r model_output/ # delete the outputs of past run if there are
```

For the following examples we use an example config from _flepimop\_sample_, but you can provide the name of any configuration file you want.&#x20;

To get started, let's start with just running a forward simulation (non-inference).&#x20;

### Non-inference run

Stay in the `PROJECT_PATH` folder, and run a simulation directly from forward-simulation Python package gempyor. Call `gempyor-simulate` providing the name of the configuration file you want to run. For example here, we use `config_sample_2pop.yml` from _flepimop\_sample_.&#x20;

```
gempyor-simulate -c config_sample_2pop.yml
```

This will produce a `model_output` folder, which you can look using provided post-processing functions and scripts.&#x20;

We recommend using `model_output_notebook.Rmd` from _flepimop\_sample_ as a starting point to interact with your model outputs. First, modify the yaml preamble in the notebook, then knit this markdown. This will produce some nice plots of the prevalence of infection states over time. You can edit this markdown to produce any figures you'd like to explore your model output.

The first time you run all this, it's , it's better to run each command individually as described above to be sure each exits successfully. However, eventually you can **put all these steps together in a script**, like below

```
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepiMoP_sample
cd $FLEPI_PATH
pip install --no-deps -e flepimop/gempyor_pkg/
cd $PROJECT_PATH
rm -rf model_output
gempyor-simulate -c config.yml
```

Note that you only have to re-run the installation steps once each time you update any of the files in the flepimop repository (either by pulling changes made by the developers and stored on Github, or by changing them yourself). If you're just running the same or different configuration file, just repeat the final steps

```
rm -rf model_output
gempyor-simulate -c new_config.yml
```

### Inference run

An inference run requires a configuration file that has the `inference` section. Stay in the `$PROJECT_PATH` folder, and run the inference script, providing the name of the configuration file you want to run.  For example here, we use `config_sample_2pop_inference.yml` from _flepimop\_sample_.&#x20;

{% code overflow="wrap" %}
```bash
flepimop-inference-main.R -c config_sample_2pop_inference.yml
```
{% endcode %}

This will run the model and create a lot of output files in `$PROJECT_PATH/model_output/`.&#x20;

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
flepimop-inference-main.R -j 1 -n 1 -k 1 -c config_inference.yml
```

where:

* `n` is the number of parallel inference slots,
* `j` is the number of CPU cores to use on your machine (if `j` > `n`, only `n` cores will actually be used. If `j` <`n`, some cores will run multiple slots in sequence)
* `k` is the number of iterations per slots.

Again, it is helpful to run the model output notebook (`model_output_notebook.Rmd` from _flepimop\_sample)_ to explore your model outputs. Knitting this file for an inference run will also provide an analysis of your fits: the acceptance probabilities, likelihoods overtime, and the fits against the provided ground truth.&#x20;

The first time you run all this, it's , it's better to run each command individually as described above to be sure each exits successfully. However, eventually you can **put all these steps together in a script**, like below

```bash
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepiMoP_sample
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
gempyor-simulate -c new_config_inference.yml
```

### Next steps

These configs and notebooks should be a good starting point for getting started with flepiMoP. To explore other running options, see [How to run: Advanced](advanced-run-guides/).&#x20;
