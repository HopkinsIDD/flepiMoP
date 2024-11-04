# Guidelines for contributors

All are welcome to contribute to flepiMoP! The easiest way is to open an issue on GitHub if you encounter a bug or if you have an issue with the framework. We would be very happy to help you out.

If you want to contribute code, fork the [flepiMoP repository](https://github.com/HopkinsIDD/flepiMoP), modify it, and submit your Pull Request (PR). In order to be merged, a pull request need:

* the approval of two reviewers AND
* the continuous integration (CI) tests passing.

### Contributing to the Python code

The "heart" of the pipeline, gempyor, is written in Python taking advantage of just-in-time compilation (via `numba`) and existing optimized libraries (`numpy`, `pandas`). If you would like to help us build gempyor, here is some useful information.

#### Frameworks

We make extensive use of the following packages:

* [click](https://click.palletsprojects.com/en/) for managing the command-line arguments
* [confuse](https://confuse.readthedocs.io/en/latest/usage.html) for accessing the configuration file
* numba to just-in-time compile the core model
* sympy to parse the model equations
* pyarrow as parquet is our main data storage format
* [xarray](https://docs.xarray.dev/en/stable/), which provides labels in the form of dimensions, coordinates and attributes on top of raw NumPy multidimensional arrays, for performance and convenience ;
* emcee for inference, as an option
* graphviz to export transition graph between compartments
* pandas, numpy, scipy, seaborn, matplotlib and tqdm like many Python projects

{% hint style="info" %}
One of the current focus is to switch internal data types from dataframes and numpy array to xarrays!
{% endhint %}

#### Tests and build dependencies

To run the tests suite locally, you'll need to install the gempyor package with build dependencies:

```bash
pip install "flepimop/gempyor_pkg[test]"
```

which installs the `pytest` and `mock` packages in addition to all other gempyor dependencies so that one can run tests.

If you are running from a conda environment and installing with \`--no-deps\`, then you should make sure that these two packages are installed.

Now you can try to run the gempyor test suite by running, from the `flepimop/gempyor_pkg` folder:

```bash
pytest
```

If that works, then you are ready to develop gempyor. Feel free to open your first pull request.

If you want more output on tests, e.g capturing standard output (print), you can use:

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

We try to remain close to Python conventions and to follow the updated rules and best practices. For formatting, we use [black](https://github.com/psf/black), the _Uncompromising Code Formatter_ before submitting pull requests. It provides a consistent style, which is useful when diffing. We use a custom length of 120 characters as the baseline is short for scientific code. Here is the line to use to format your code:

```bash
black --line-length 120 . --exclude renv*
```

{% hint style="warning" %}
Please use type-hints as much as possible, as we are trying to move towards static checks.
{% endhint %}

#### Structure of the main classes

The code is structured so that each of the main classes **owns** a config segment, and only this class should parse and build the related object. To access this information, other classes first need to build the object.

{% hint style="warning" %}
Below, this page is still underconstruction
{% endhint %}

The main classes are:

* `Coordinates:` this is a light class that stores all the coordinates needed by every other class (e.g the time serie
* `Parameter`
* `Compartments`
* `Modifers`
* `Seeding`,
* `InitialConditions`
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
* ;batch/inference\_job.run for slurm
