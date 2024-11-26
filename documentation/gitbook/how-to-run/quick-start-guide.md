---
description: >-
  Instructions to get started with using gempyor and flepiMoP, using some
  provided example configs to help you.
---

# Quick Start Guide

## 🧱 Set up

### Access model files

Follow all the steps in the [Before any run](before-any-run.md) section to ensure you have access to the correct files needed to run your custom model or a sample model with flepiMoP.

Take note of the location of the directory on your local computer where you cloned the flepiMoP model code (which we'll call `FLEPI_PATH`), as well as where you cloned your project files (which we'll call `PROJECT_PATH`).

{% hint style="info" %}
For example, if you cloned your Github repositories into a local folder called `Github` and are using [_flepimop\_sample_](https://github.com/HopkinsIDD/flepimop_sample) as a project repository, your directory names could be\
\
&#xNAN;_**On Mac:**_

/Users/YourName/Github/flepiMoP

/Users/YourName/Github/flepimop\_sample\
\
&#xNAN;_**On Windows:**_\
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

You can check that the variables have been set by either typing `env` to see all defined environmental variables, or typing `echo $FLEPI_PATH` to see the value of `FLEPI_PATH`.

If you're on a **Windows** machine

<pre class="language-bash"><code class="lang-bash"><strong>set FLEPI_PATH=C:\Users\YourName\Github\flepiMoP
</strong>set PROJECT_PATH=C:\Users\YourName\Github\flepimop_sample
</code></pre>

or, if you have already navigated to your parent directory

<pre class="language-bash"><code class="lang-bash"><strong>set FLEPI_PATH=%CD%\flepiMoP
</strong>set PROJECT_PATH=%CD%\flepimop_sample
</code></pre>

You can check that the variables have been set by either typing `set` to see all defined environmental variables, or typing `echo $FLEPI_PATH$` to see the value of `FLEPI_PATH`.

{% hint style="info" %}
If you don't want to bother defining the environmental variables, it's no problem, just remember to use the full or relative path names for navigating to the right files or folders in future steps
{% endhint %}

### Install packages

The code is written in a combination of [R](https://www.r-project.org/) and [Python](https://www.python.org/). The Python part of the model is a package called [_gempyor_](../gempyor/model-description.md), and includes all the code to simulate the epidemic model and the observational model and apply time-dependent interventions. The R component conducts the (optional) parameter inference, and all the (optional) provided pre and post processing scripts are also written in R. Most uses of the code require interacting with components written in both languages, and thus making sure that both are installed along with a set of required packages. However, Python alone can be used to do forward simulations of the model using _gempyor_.

First, ensure you have python and R installed. You need a working python3.7+ installation. We recommend using the latest stable python release (python 3.12) to benefit from huge speed-ups and future-proof your installation. We also recommend installing Rstudio to interact with the R code and for exploring your model outputs.

{% hint style="info" %}
_**On Mac**_ 🍏

Python 3 is installed by default on recent MacOS installation. If it is not, you might want to check [homebrew](https://brew.sh/) and install the appropriate installation.

However, this may result in two versions of Python being installed on your computer. If there are multiple versions of Python (e.g., multiple versions of Python 3), you may need to specify which version to use in the installation. This can be done by following the instructions for using a conda environment, in which case the version of Python to use can be specified in the creation of the virtual environment, e.g., `conda create -c conda-forge -n flepimop-env python=3.12 numba pandas numpy seaborn tqdm matplotlib click confuse pyarrow sympy dask pytest scipy graphviz emcee xarray boto3 slack_sdk`. The conda environment will be activated in the same way and when installing gempyor, the version of pip used in the installation will reflect the Python version used in the conda environment (e.g., 3.12), so you can use `pip install -e flepimop/gempyor_pkg/` in this case.

There is also the possibility that multiple versions of _gempyor_ have been installed on your computer in the various iterations of Python. You will only want to have _gempyor_ installed on the latest version of Python (e.g., Python 3.8+) that you have. You can remove a _gempyor_ iteration installed for a given version of Python using `pip[version] uninstall gempyor` e.g., `pip3.7 uninstall gempyor`. Then, you will need to specify which version of Python to install _gempyor_ on during that step (see above).
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

If you would like to install _gempyor_ directly from GitHub, go to the flepiMoP directory and use the following command:

```bash
cd $FLEPI_PATH
pip install --no-deps "git+https://github.com/HopkinsIDD/flepiMoP.git@main#egg=gempyor&subdirectory=flepimop/gempyor_pkg"
```

If you just want to [run a forward simulation](quick-start-guide.md#non-inference-run), installing python's _gempyor_ is all you need.

To [run an inference run](quick-start-guide.md#inference-run) and to explore your model outputs using provided post-processing functionality, there are some packages you'll need to **install in R**. Open your **R terminal** (at the bottom of RStudio, or in the R IDE), and run the following command to install the necessary R packages:

<pre class="language-r" data-overflow="wrap"><code class="lang-r"><strong># while in R
</strong><strong>>install.packages(c("readr","sf","lubridate","tidyverse","gridExtra","reticulate","truncnorm","xts","ggfortify","flextable","doParallel","foreach","optparse","arrow","devtools","cowplot","ggraph","data.table"))
</strong></code></pre>

{% hint style="info" %}
_**On Linux**_

The R packages "`sf"` and "`ggraph"` require you to have `libgdal-dev` and `libopenblas-dev` installed on your local linux machine.
{% endhint %}

This step does not need to be repeated unless you use a new computer or delete and reinstall R.

Now return to your system terminal. To install _flepiMop_-internal R packages, run the following from command line:

{% code overflow="wrap" %}
```bash
Rscript build/local_install.R  # Install R packages
```
{% endcode %}

After installing the _flepiMoP_ R packages, we need to do one more step to install the command line tools for the inference package. If you are not running in a conda environment, you need to point this installation step to a location that is on your executable search path (i.e., whenever you call a command from the terminal, the places that are searched to find that executable). To find a consistent location, type

```
>which gempyor-simulate
```

The location that is returned will be of the form `EXECUTABLE_SEARCH_PATH/gempyor-simulate`. Then run the following in an R terminal:

```r
# While in R
>library(inference)
>inference::install_cli("EXECUTABLE_SEARCH_PATH")
```

To install the inference package's CLI tools.

Each installation step may take a few minutes to run.

{% hint style="info" %}
Note: These installations take place on your local operating system. You will need an active internet connection for installation, but not for other steps of running the model.\
If you are concerned about disrupting your base environment that you use for other projects, try using [Docker](advanced-run-guides/running-with-docker-locally.md) or [conda](advanced-run-guides/quick-start-guide-conda.md) instead.
{% endhint %}

## 🚀 Run the code

Everything is now ready 🎉 .

The next step depends on what sort of simulation you want to run: One that includes inference (fitting model to data) or only a forward simulation (non-inference). Inference is run from R, while forward-only simulations are run directly from the Python package `gempyor`.

First, navigate to the project folder and make sure to delete any old model output files that are there.

```bash
cd $PROJECT_PATH       # goes to your project repository
rm -r model_output/ # delete the outputs of past run if there are
```

For the following examples we use an example config from _flepimop\_sample_, but you can provide the name of any configuration file you want.

To get started, let's start with just running a forward simulation (non-inference).

### Non-inference run

Stay in the `PROJECT_PATH` folder, and run a simulation directly from forward-simulation Python package gempyor. Call `gempyor-simulate` providing the name of the configuration file you want to run. For example here, we use `config_sample_2pop.yml` from _flepimop\_sample_.

```
gempyor-simulate -c config_sample_2pop.yml
```

This will produce a `model_output` folder, which you can look using provided post-processing functions and scripts.

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

An inference run requires a configuration file that has the `inference` section. Stay in the `$PROJECT_PATH` folder, and run the inference script, providing the name of the configuration file you want to run. For example here, we use `config_sample_2pop_inference.yml` from _flepimop\_sample_.

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

Again, it is helpful to run the model output notebook (`model_output_notebook.Rmd` from _flepimop\_sample)_ to explore your model outputs. Knitting this file for an inference run will also provide an analysis of your fits: the acceptance probabilities, likelihoods overtime, and the fits against the provided ground truth.

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
flepimop-inference-main -c config_inference_new.yml
```

## 📈 Examining model output

If your run is successful, you should see your output files in the model\_output folder. The structure of the files in this folder is described in the [Model Output](../gempyor/output-files.md) section. By default, all the output files are .parquet format (a compressed format which can be imported as dataframes using R's arrow package `arrow::read_parquet` or using the free desktop application [Tad ](https://www.tadviewer.com/)for quick viewing). However, you can add the option `--write-csv` to the end of the commands to run the code (e.g., `> gempyor-simulate -c config.yml --write-csv)` to have everything saved as .csv files instead ;

## 🪜 Next steps

These configs and notebooks should be a good starting point for getting started with flepiMoP. To explore other running options, see [How to run: Advanced](advanced-run-guides/).
