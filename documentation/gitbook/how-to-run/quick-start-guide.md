---
description: |
    Quick instructions on how to install Prerequisites, install flepiMoP itself, and
    then run through a quick example of how to use flepiMoP.
---

# Quick Start Guide

`flepiMoP` is flexible pipeline for modeling epidemics. It has functionality for simulating epidemics as well as doing inference for simulation parameters and post-processing of simulation/inference outputs. It is written in a combination of python and R and uses anaconda to manage installations which allows `flepiMoP` to enforce version constraints across both languages.

## Prerequisites

`flepiMoP` requires the following:

* [`git`](https://git-scm.com/), and
* [`conda`](https://anaconda.org/).

If you do not have `git` installed you can go to [the downloads page](https://git-scm.com/downloads) to find the appropriate installation for your system. It's also recommended, but not required, to have a [GitHub](https://github.com/) account. If you're totally new to `git` and GitHub, GitHub has a very nice introduction to the [basics of git](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git) that is worth reading before continuing.

If you do not have `conda` installed you can go to [the downloads page](https://www.anaconda.com/download) to find the appropriate installation for your system. We would recommend selecting the `Anaconda Distribution` installer of `conda`.

## Installing `flepiMoP`

Navigate to the parent location of where you would like to install `flepiMoP`, a subdirectory called `flepiMoP` will be created there. For example, if you navigate to `~/Desktop` then `flepiMoP` will be installed to `~/Desktop/flepiMoP`.

{% hint style="warning" %}
This installation script is currently only designed for Linux/MacOS operating systems or linux shells for windows. If you need windows native installation please reach out for assistance.
{% endhint %}

```shell
$ curl -LsSf -o flepimop-install "https://raw.githubusercontent.com/HopkinsIDD/flepiMoP/refs/heads/main/bin/lint"
$ chmod +x flepimop-install
```

This installations script will guide you through a series of prompts to determine how and where to install `flepiMoP`. Loosely this script:

1. Determines what directory `flepiMoP` is being installed into,
2. Optionally gets a clone of `flepiMoP` if it is not present at the install location,
3. Creates a conda environment to house the installation,
4. Installs `flepiMoP`'s dependencies and custom packages to this conda environment, and
5. Finally prints out a summary of the installation with helpful debugging information.

For more help on how to use the installation script you can do `./flepimop-install -h` to get help information. For first time users accepting the default prompt will be the best choice (as shown below):

```shell
$ ./flepimop-install
An explicit $USERDIR was not provided, please set one (or press enter to use '/Users/example/Desktop'):
Using '/Users/example/Desktop' for $USERDIR.
An explicit $FLEPI_PATH was not provided, please set one (or press enter to use '/Users/example/Desktop/flepiMoP'):
Using '/Users/example/Desktop/flepiMoP' for $FLEPI_PATH.
Did not find flepiMoP at '/Users/example/Desktop/flepiMoP', do you want to clone the repo? (y/n) y
Cloning on your behalf.
Cloning into '/Users/example/Desktop/flepiMoP'...
remote: Enumerating objects: 28513, done.
remote: Counting objects: 100% (3424/3424), done.
remote: Compressing objects: 100% (845/845), done.
remote: Total 28513 (delta 2899), reused 2786 (delta 2576), pack-reused 25089 (from 2)
Receiving objects: 100% (28513/28513), 145.99 MiB | 26.32 MiB/s, done.
Resolving deltas: 100% (14831/14831), done.
An explicit $FLEPI_CONDA was not provided, please set one (or press enter to use 'flepimop-env'):
Using 'flepimop-env' name for $FLEPI_CONDA.
...
```

Once the prompts are done the installer will output information about the installations that it is doing. After the installation has completed you should see an installation summary similar to:

```shell
flepiMoP installation summary:
> flepiMoP version: ec707d36cd9f8675466c05cbaba295cc4f4a7112
> flepiMoP path: /Users/example/Desktop/flepiMoP
> flepiMoP conda env: flepimop-env
> conda: 24.9.2
> R 4.3.3: /opt/anaconda3/envs/flepimop-env/bin/R
> Python 3.11.12: /opt/anaconda3/envs/flepimop-env/bin/python
> gempyor version: 2.1
> R flepicommon version: 0.0.1
> R flepiconfig version: 3.0.0
> R inference version: 0.0.1

To activate the flepimop conda environment, run:
    conda activate flepimop-env
```

This summary gives a brief overview of the R/python/package versions installed. If you encounter any issues with your installation please include this information with your issue report.

## Activating A `flepiMoP` Installation

To activate `flepiMoP` you need to activate the conda environment that it is installed to:

```shell
$ conda activate flepimop-env
```

Or replacing `flepimop-env` with the appropriate conda environment if you decided on a non-default conda environment. Once you do this you should have the `flepimop` CLI available to you with:

```shell
$ flepimop --help
Usage: flepimop [OPTIONS] COMMAND [ARGS]...

  Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface

Options:
  --help  Show this message and exit.

Commands:
  batch-calibrate  Submit a calibration job to a batch system.
  compartments     Add commands for working with FlepiMoP compartments.
  modifiers
  patch            Merge configuration files.
  simulate         Forward simulate a model using gempyor.
  sync             Sync flepimop files between local and remote locations.
```

### Defining Environment Variables (Optional)

{% hint style="info" %}
If you choose not to define environment variables, remember to use the full or relative path names for navigating to the right directories and provide appropriate flepi/project path arguments in future steps.
{% endhint %}

`flepiMoP` frequently uses two environment variables to refer to specific directories both as a default for CLI arguments and throughout the documentation:

1. `FLEPI_PATH`: Refers to the directory where `flepiMoP` is installed to, and
2. `PROJECT_PATH`: Refers to the directory where `flepiMoP` is being ran from.

Furthermore, you'll likely be navigating between these directories frequently in production usage so having these environment variables set can save some typing.

Continuing with the same paths from the installation example `flepiMoP` was installed to `/Users/example/Desktop/flepiMoP`. On Linux/MacOS or in linux shells on windows setting an environment variable can be done by:

```bash
export FLEPI_PATH=/Users/example/Desktop/flepiMoP
export PROJECT_PATH=/Users/example/Desktop/flepiMoP/examples/tutorials
```

Where `/your/path/to` is the directory containing `flepiMoP`. If you have already navigated to your flepiMoP directory you can just do:

```bash
export FLEPI_PATH=$(pwd)
export PROJECT_PATH=$(pwd)/examples/tutorials
```

You can check that the variables have been set by either typing `env` to see all defined environment variables, or typing `echo $FLEPI_PATH` and `echo $PROJECT_PATH` to see the values of `FLEPI_PATH` and `PROJECT_PATH`.

However, if you're on Windows:

```bash
set FLEPI_PATH=C:\your\path\to\flepiMoP
set PROJECT_PATH=C:\your\path\to\flepiMoP\examples\tutorials
```

Where `/your/path/to` is the directory containing `flepiMoP`. If you have already navigated to your flepiMoP directory you can just do:

```bash
set FLEPI_PATH=%CD%
set PROJECT_PATH=%CD%\examples\tutorials
```

You can check that the variables have been set by either typing `set` to see all defined environment variables, or typing `echo $FLEPI_PATH$` and `echo $PROJECT_PATH$` to see the values of `FLEPI_PATH` and `PROJECT_PATH`.

For more information on the usage of environment variables with `flepiMoP` please refer to the [Environment Variables](./environmental-variables.md) documentation.

## Run `flepiMoP`

Now that `flepiMoP` has been successfully installed on your system you will be able to use the tool to model epidemics.

First, navigate to the `PROJECT_PATH` folder and make sure to delete any old model output files that are there:

```shell
$ cd $PROJECT_PATH
$ rm -r model_output/
```

### Non-Inference Run

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

### Inference Run

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

### Examining Model Output

If your run is successful, you should see your output files in the model\_output folder. The structure of the files in this folder is described in the [Model Output](../gempyor/output-files.md) section. By default, all the output files are .parquet format (a compressed format which can be imported as dataframes using R's arrow package `arrow::read_parquet` or using the free desktop application [Tad ](https://www.tadviewer.com/) for quick viewing). However, you can add the option `--write-csv` to the end of the commands to run the code (e.g.,  `flepimop simulate --write-csv config.yml`) to have everything saved as .csv files instead ;

## Next Steps

These configs and notebooks should be a good starting point for getting started with flepiMoP. To explore other running options, see [How to run: Advanced](advanced-run-guides/).
