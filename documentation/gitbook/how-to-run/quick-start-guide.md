---
description: >-
  Instructions to get started with using gempyor and flepiMoP, directly following the 
  steps described in "Before Any Run", by activating your install and running some 
  sample commands.
---

# Quick Start Guide

## 🧱 Set Up

Before completing this **Quick Start Guide**, make sure you have followed all the steps in the [Before any run](before-any-run.md) section to ensure you have access to the correct files needed to run your model with flepiMoP.

## Activating The Conda Environment

First step in using `flepiMoP` is activating the conda environment that it has been installed in:

```bash
conda activate flepimop-env
```

Assuming that you installed `flepiMoP` to the default conda environment name, but if you choose to install elsewhere please adjust the above command accordingly.

### Define Environment Variables (Optional)

{% hint style="info" %}
If you choose not to define environment variables, remember to use the full or relative path names for navigating to the right directories and provide appropriate flepi/project path arguments in future steps.
{% endhint %}

`flepiMoP` frequently uses two environment variables to refer to specific directories both as a default for CLI arguments and throughout the documentation:

1. `FLEPI_PATH`: Refers to the directory where `flepiMoP` is installed to, and
2. `PROJECT_PATH`: Refers to the directory where `flepiMoP` is being ran from.

Furthermore, you'll likely be navigating between these directories frequently in production usage so having these environment variables set can save some typing.

For example, if you're on a **Mac** or Linux/Unix based operating system and storing the `flepiMoP` code in a directory called `Github`, you define the FLEPI\_PATH and PROJECT\_PATH environmental variables to be your directory locations as follows:

On Linux/MacOS or in linux shells on windows setting an environment variable can be done by:

```bash
export FLEPI_PATH=/your/path/to/flepiMoP
export PROJECT_PATH=/your/path/to/flepiMoP/examples/tutorials
```

Where `/your/path/to` is the directory containing `flepiMoP`. If you have already navigated to your flepiMoP directory you can just do:

```bash
export FLEPI_PATH=$(pwd)
export PROJECT_PATH=$(pwd)/examples/tutorials
```

You can check that the variables have been set by either typing `env` to see all defined environmental variables, or typing `echo $FLEPI_PATH` to see the value of `FLEPI_PATH`.

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

You can check that the variables have been set by either typing `set` to see all defined environmental variables, or typing `echo $FLEPI_PATH$` to see the value of `FLEPI_PATH`.

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
