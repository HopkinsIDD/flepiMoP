---
description: Short tutorial on running locally using an "Anaconda" environment.
---

# Running locally in a conda environment 🐍

### Access model files

As is the case for any run, first see the [Before any run](../before-any-run.md) section to ensure you have access to the correct files needed to run. On your local machine, determine the file paths to:

* the directory containing the flepimop code (likely the folder you cloned from Github), which we'll call `FLEPI_PATH`
* the directory containing your project code including input configuration file and population structure (again likely from Github), which we'll call `DATA_PATH`

{% hint style="info" %}
For example, if you clone your Github repositories into a local folder called Github and are using the flepimop\_sample as a project repository, your directory names could be\
\
_**On Mac:**_

\<dir1> = /Users/YourName/Github/flepiMoP

\<dir2> = /Users/YourName/Github/flepimop\_sample\
\
_**On Windows:**_\
\<dir1> = C:\Users\YourName\Github\flepiMoP

\<dir2> = C:\Users\YourName\Github\flepimop\_sample\\

(hint: if you navigate to a directory like `C:\Users\YourName\Github` using `cd C:\Users\YourName\Github`, modify the above `<dir1>` paths to be `.\flepiMoP` and `.\flepimop_sample)`

:warning: Note again that these are best cloned **flat.**
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

First, you'll need to fill in some variables that are used by the model. This can be done in a script (an example is provided at the end of this page). For your first time, it's better to run each command individually to be sure it exits successfully.

First, in `myparentfolder` populate the folder name variables for the paths to the flepimop code folder and the project folder:

```bash
export FLEPI_PATH=$(pwd)/flepiMoP
export DATA_PATH=$(pwd)/flepimop_sample
```

Go into the code directory (making sure it is up to date on your favorite branch) and do the installation required of the repository:

```bash
cd $FLEPI_PATH # move to the flepimop directory
Rscript build/local_install.R # Install R packages
pip install --no-deps -e flepimop/gempyor_pkg/ # Install Python package gempyor
```

Each installation step may take a few minutes to run.

{% hint style="info" %}
Note: These installations take place in your conda environment and not the local operating system. They must be made once while in your environment and need not be done for every time you run a model, provided they have been installed once. You will need an active internet connection for installing the R packages (since some are hosted online), but not for other steps of running the model.
{% endhint %}

<details>

<summary>Help! I have errors in installation</summary>

If you get an error because no cran mirror is selected, just create in your home directory a `.Rprofile` file:

{% code title="~/.Rprofile" lineNumbers="true" %}
```r
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org" 
       options(repos=r)
})
```
{% endcode %}

Perhaps this should be added to the top of the local\_install.R script #todo

When running `local_install.R` the first time, you may get an error:

<pre><code><strong>ERROR: dependency ‘report.generation’ is not available for package ‘inference’
</strong><strong>[...]
</strong><strong>installation of package ‘./R/pkgs//inference’ had non-zero exit status
</strong></code></pre>

and the second time it'll finish successfully (no non-zero exit status at the end). That's because there is a circular dependency in this file (inference requires report.generation which is built after) and will hopefully get fixed.

For subsequent runs, once is enough because the package is already installed once.

</details>

Other environmental variables can be set at any point in process of setting up your model run. These options are listed in ... ADD ENVAR PAGE

For example, some frequently used environmental variables which we recommend setting are:

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
cd $DATA_PATH       # goes to your project repository
rm -r model_output/ # delete the outputs of past run if there are
```

#### Inference run

An inference run requires a configuration file that has an `inference` section. Stay in the `$DATA_PATH` folder, and run the inference script, providing the name of the configuration file you want to run (ex. `config.yml`). In the example data folder (flepimop\_sample), try out the example config XXX.

```bash
Rscript  $FLEPI_PATH/flepimop/main_scripts/inference_main.R -c config.yml
```

This will run the model and create [a lot of output files](../../gempyor/output-files.md) in `$DATA_PATH/model_output/`.

The last few lines visible on the command prompt should be:

> \[\[1]]
>
> \[\[1]]\[\[1]]
>
> \[\[1]]\[\[1]]\[\[1]]
>
> NULL

If you want to quickly do runs with options different from those encoded in the configuration file, you can do that from the command line, for example

```
Rscript $FLEPI_PATH/flepimop/main_scripts/inference_main.R -j 1 -n 1 -k 1 -c config.yml
```

where:

* `n` is the number of parallel inference slots,
* `j` is the number of CPU cores to use on your machine (if `j` > `n`, only `n` cores will actually be used. If `j` <`n`, some cores will run multiple slots in sequence)
* `k` is the number of iterations per slots.

#### Non-inference run

Stay in the `$DATA_PATH` folder, and run a simulation directly from forward-simulation Python package `gempyor`. To do this, call `gempyor-simulate` providing the name of the configuration file you want to run (ex. `config.yml`). An example config is provided in `flepimop_sample/config_sample_2pop_interventions.yml.`

```
gempyor-simulate -c config.yml
```

{% hint style="warning" %}
It is currently required that all configuration files have an `interventions` section. There is currently no way to simulate a model with no interventions, though this functionality is expected soon. For now, simply create an intervention that has value zero.
{% endhint %}

You can also try to knit the Rmd file in `flepiMoP/flepimop/gempyor_pkg/docs` which will show you how to analyze these files.

### Do it all with a script

The following script does all the above commands in an easy script. Save it in `myparentfolder` as `quick_setup.sh`. Then, just go to `myparentfolder` and type `source quick_setup_flu.sh` and it'll do everything for you!

{% code title="quick_setup_flu.sh" lineNumbers="true" %}
```bash
export FLEPI_PATH=$(pwd)/flepiMoP
export DATA_PATH=$(pwd)/flepimop_sample

cd $FLEPI_PATH
Rscript build/local_install.R
pip install --no-deps -e gempyor_pkg/ # before: python setup.py develop --no-deps

cd $DATA_PATH
rm -rf model_output
export CONFIG_PATH=config.yml # set your configuration file path

Rscript $FLEPI_PATH/flepimop/main_scripts/inference_main.R.R -j 1 -n 1 -k 1
```
{% endcode %}
