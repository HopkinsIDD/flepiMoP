---
description: Short tutorial on running locally using an "Anaconda" environment.
---

# Running locally in a conda environment 🐍

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

## 🧱 Setup (do this once)

#### Installing the `conda` environment

One of simplest ways to get everything to work is to build an Anaconda environment. Install (or update) Anaconda on your computer. We find that it is easiest to create your conda environment by installing required python packages, then installing R packages separately once your conda environment has been built as not all R packages can be found on conda.

You can either use the command line (here) or the graphical user interface (you just tick the packages you want). With the command line it's this one-liner:

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash">conda update conda # makes sure you have a recent conda instatllation

# be sure to copy the whole thing as a single line ! copy it to your text editor
<strong>conda create -c conda-forge -n flepimop-env numba pandas numpy seaborn tqdm matplotlib click confuse pyarrow sympy dask pytest scipy graphviz emcee xarray boto3 slack_sdk
</strong></code></pre>

Anaconda will take some time, to come up with a proposal that works with all dependencies. This creates a `conda` environment named `flepimop-env` that has all the necessary python packages.\
\
The next step in preparing your environment is to install the necessary R packages. First, activate your environment, launch R and then install the following packages.

<pre class="language-bash" data-overflow="wrap"><code class="lang-bash"><strong>conda activate flepimop-env # this launches the environment you just created
</strong>
R # to launch R from command line

<strong># while in R
</strong><strong>install.packages(c("readr","sf","lubridate","tidyverse","gridExtra","reticulate","truncnorm","xts","ggfortify","flextable","doParallel","foreach","optparse","arrow","devtools","cowplot","ggraph"))
</strong></code></pre>

If you'd like, you can install `rstudio` as a package as well.

<details>

<summary>Can I run without conda?</summary>

Anaconda is the most reproducible way to run our model. However, you can still proceed without it. You can just carry on with the steps below without creating an environment.

**How to do it?** Just skip every line starting with `conda` and do not use the `--no-deps` flag when installing gempyor (so pip will install the dependencies). When running `local_install.R` there may be failures because some packages are missing. Install them as you usually do from R. The rest is the same as this tutorial.

</details>

## 🚀 Run the model

Activate your conda environment, which we built above.

```bash
conda activate flepimop-env
```

In this `conda` environment, commands with R and python will uses this environment's R and python.

### Define environment variables

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

Other environmental variables can be set at any point in process of setting up your model run. These options are listed in ... **ADD ENVAR PAGE**

For example, some frequently used environmental variables we recommend setting are:

{% code overflow="wrap" %}
```bash
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
```
{% endcode %}

### Run the code

Everything is now ready. 🎉

The next step depends on what sort of simulation you want to run: One that includes inference (fitting model to data) or only a forward simulation (non-inference). Inference is run from R, while forward-only simulations are run directly from the Python package `gempyor`.

In either case, navigate to the project folder and make sure to delete any old model output files that are there.

```bash
cd $PROJECT_PATH       # goes to your project repository
rm -r model_output/ # delete the outputs of past run if there are
```

#### Inference run

An inference run requires a configuration file that has an `inference` section. Stay in the `$PROJECT_PATH` folder, and run the inference script, providing the name of the configuration file you want to run (ex. `config.yml`). 

```bash
flepimop-inference-main.R -c config.yml
```

This will run the model and create [a lot of output files](../../gempyor/output-files.md) in `$PROJECT_PATH/model_output/`.

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
flepimop-inference-main -j 1 -n 1 -k 1 -c config.yml
```

where:

* `n` is the number of parallel inference slots,
* `j` is the number of CPU cores to use on your machine (if `j` > `n`, only `n` cores will actually be used. If `j` < `n`, some cores will run multiple slots in sequence)
* `k` is the number of iterations per slots.

#### Non-inference run

Stay in the `$PROJECT_PATH` folder, and run a simulation directly from forward-simulation Python package `gempyor`. To do this, call `flepimop simulate` providing the name of the configuration file you want to run (ex. `config.yml`). An example config is provided in `$PROJECT_PATH/config_sample_2pop_interventions.yml.`

```
flepimop simulate config.yml
```

{% hint style="warning" %}
It is currently required that all configuration files have an `interventions` section. There is currently no way to simulate a model with no interventions, though this functionality is expected soon. For now, simply create an intervention that has value zero.
{% endhint %}

You can also try to knit the Rmd file in `flepiMoP/flepimop/gempyor_pkg/docs` which will show you how to analyze these files.


