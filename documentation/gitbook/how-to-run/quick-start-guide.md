---
description: >-
  Instructions to get started with using gempyor and flepiMoP.
  Executing a simple run with provided example configs.
---

# Quick Start Guide

## 🧱 Set up

### Access model files

Follow all the steps in the [Before any run](before-any-run.md) section to ensure you have access to the correct files needed to run your model with flepiMoP.

Take note of the location of the directory on your local computer where you cloned the flepiMoP model code (which we'll call `FLEPI_PATH`).

{% hint style="info" %}
For example, if you cloned your Github repositories into a local folder called `Github` and are using `flepiMoP/examples/tutorials` as a project repository, your directory names could be\
\
_**On Mac:**_

/Users/YourName/Github/flepiMoP

/Users/YourName/Github/fleiMoP/examples/tutorials
\
_**On Windows:**_\
C:\Users\YourName\Github\flepiMoP

C:\Users\YourName\Github\flepiMoP\examples\tutorials
{% endhint %}

### Define environment variables (optional)

Since you'll be navigating frequently between the folder that contains your project code and the folder that contains the core flepiMoP model code, it's helpful to define shortcuts for these file paths. You can do this by creating environmental variables that you can then quickly call instead of writing out the whole file path.

If you're on a **Mac** or Linux/Unix based operating system, define the FLEPI\_PATH and PROJECT\_PATH environmental variables to be your directory locations, for example

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

If you're on a **Windows** machine

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

Stay in the `PROJECT_PATH` folder, and run a simulation directly from forward-simulation Python package `gempyor`. Call `gempyor-simulate` providing the name of the configuration file you want to run. For example here, we use `config_sample_2pop.yml`.

```
gempyor-simulate -c config_sample_2pop.yml
```

This will produce a `model_output` folder, which you can look at using provided post-processing functions and scripts.

We recommend using `model_output_notebook.Rmd` as a starting point to interact with your model outputs. First, modify the YAML preamble in the notebook (make sure the configuration file listed matches the one used in simulation), then knit this markdown. This will produce plots of the prevalence of infection states over time. You can edit this markdown to produce any figures you'd like to explore your model output.

For your first `flepiMoP` run, it's better to run each command individually as described above to be sure each exits successfully. However, eventually you can **put all these steps together in a script**, seen below:

```
export FLEPI_PATH=/Users/YourName/Github/flepiMoP
export PROJECT_PATH=/Users/YourName/Github/flepiMoP/examples/tutorials
cd $FLEPI_PATH
pip install --no-deps -e flepimop/gempyor_pkg/
cd $PROJECT_PATH
rm -rf model_output
gempyor-simulate -c config.yml
```

Note that you only have to re-run the installation steps once each time you update any of the files in the flepimop repository (either by pulling changes made by the developers and stored on Github, or by changing them yourself). If you're just running the same or different configuration file, just repeat the final steps:

```
rm -rf model_output
gempyor-simulate -c new_config.yml
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
Rscript $FLEPI_PATH/flepimop/main_scripts/inference_main.R -c config_inference_new.yml
```

## 📈 Examining model output

If your run is successful, you should see your output files in the `model_output` folder. The structure of the files in this folder is described in the [Model Output](../gempyor/output-files.md) section. By default, all the output files are .parquet format (a compressed format which can be imported as dataframes using R's arrow package `arrow::read_parquet` or using the free desktop application [Tad ](https://www.tadviewer.com/)for quick viewing). However, you can add the option `--write-csv` to the end of the commands to run the code (e.g., `> gempyor-simulate -c config.yml --write-csv)` to have everything saved as .csv files instead ;

## 🪜 Next steps

These configs and notebooks should be a good starting point for getting started with flepiMoP. To explore other running options, see [How to run: Advanced](advanced-run-guides/).
