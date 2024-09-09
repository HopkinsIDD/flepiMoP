# Python guidelines for developers

The "heart" of the pipeline, gempyor, is written in python taking advantage of just-in-time compilation (via numba) and existing optimized librairies (numpy, pandas). If you would like to help us build gempyor, here are some useful information

### Tests and build dependencies

First, you'll need to install the gempyor package with build dependencies:

```bash
pip install "flepimop/gempyor_pkg[test]"
```

which installs the `pytest` and `mock` packages in addition to all other gempyor dependencies so that one can run tests.

If you are running from a conda environment and installing with \`--no-deps\`, then you should make sure that these two packages are installed.

Now you can try to run the gempyor test suite by running, from the `gempyor_pkg` folder:

```bash
pytest
```

If that works, then you are ready to develop gempyor. Feel free to open your first pull request.

If you want more output on tests, e.g capturing standart output (print), you can use:

```bash
pytest -vvvv
```

and to run just some subset of the tests (e.g here just the outcome tests), use:

```bash
pytest -vvvv -k outcomes
```

{% hint style="danger" %}
Before committing, make sure you **format your code** using black (see below) and that the **test passes** (see above).
{% endhint %}

### Formatting

{% hint style="info" %}
Code formatters are necessary, but not sufficient for well formatted code and further style changes may be requested in PRs. Furthermore, the formatting/linting requirements for code contributed to `flepiMoP` are likely to be enhanced in the future and those changes will be reflected here when they come.
{% endhint %}

For python code formatting the [black](https://black.readthedocs.io/en/stable/) code formatter is applied to all edits to python files being merged into `flepiMoP`. For installation and detailed usage guides please refer to the black documentation. For most use cases the following commands are sufficient:

```bash
# See what style changes need to be made
black --diff .
# Reformat the python files automatically
black .
# Check if current work would be allowed to merged into flepiMoP
black --check .
```

### Structure of the main classes

The main classes, such as `Parameter`, `NPI`, `SeedingAndInitialConditions`,`Compartments` should tend to the same struture:

* a `writeDF`
* function to plot
* (TODO: detail pipeline internal API)

### Batch folder

Here are some notes useful to improve the batch submission:

Setup site wide Rprofile.

```
export R_PROFILE=$COVID_PATH/slurm_batch/Rprofile
```

> SLURM copies your environment variables by default. You don't need to tell it to set a variable on the command line for sbatch. Just set the variable in your environment before calling sbatch.

> There are two useful environment variables that SLURM sets up when you use job arrays:

> SLURM\_ARRAY\_JOB\_ID, specifies the array's master job ID number. SLURM\_ARRAY\_TASK\_ID, specifies the job array index number. https://help.rc.ufl.edu/doc/Using\_Variables\_in\_SLURM\_Jobs

SLURM does not support using variables in the #SBATCH lines within a job script (for example, #SBATCH -N=$REPS will NOT work). A very limited number of variables are available in the #SBATCH just as %j for JOB ID. However, values passed from the command line have precedence over values defined in the job script. and you could use variables in the command line. For example, you could set the job name and output/error files can be passed on the sbatch command line:

```
RUNTYPE='test'
RUNNUMBER=5
sbatch --job-name=$RUNTYPE.$RUNNUMBER.run --output=$RUNTYPE.$RUNUMBER.txt --export=A=$A,b=$b jobscript.sbatch
```

However note in this example, the output file doesn't have the job ID which is not available from the command line, only inside the sbatch shell script.

#### File descriptions

launch\_job.py and runner.py for non inference job

inference\_job.py launch a slurm or aws job, where it uses

* \`inference\_runner.sh\` and inference\_copy.sh for aws
* &#x20;batch/inference\_job.run for slurm
