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
pip install "flepimop/gempyor_pkg[dev]"
```

which installs the `pytest` package in addition to all other gempyor dependencies so that one can run tests. Now you can try to run the gempyor test suite by running, from the `flepimop/gempyor_pkg` folder:

```bash
pytest
```

If that works, then you are ready to develop gempyor. Feel free to open your first pull request. If you want more output on tests, e.g capturing standard output (print), you can use:

```bash
pytest -vvv
```

And to run just some subset of the tests (e.g here just the outcome tests), use:

```bash
pytest -vvv -k outcomes
```

Furthermore, to ensure that the examples provided in the documentation are high quality we run the doctests in CI with:

```bash
pytest --doctest-modules src/gempyor/
```

For more details on how to use `pytest` please refer to their [usage guide](https://docs.pytest.org/en/latest/how-to/usage.html).

### Formatting and Linting 

We try to remain close to Python conventions and to follow the updated rules and best practices. For formatting, we use [black](https://github.com/psf/black), the _Uncompromising Code Formatter_ before submitting pull requests. It provides a consistent style, which is useful when diffing. To get started with black please refer to their [Getting Started guide](https://black.readthedocs.io/en/stable/getting_started.html). We use a custom length of 92 characters as the baseline is short for scientific code. Here is the line to use to format your code:

```bash
black --line-length 92 \
    --extend-exclude 'flepimop/gempyor_pkg/src/gempyor/steps_rk4.py' \
    --verbose .
```

To identify instances of poor Python practices within `gempyor`, we use [pylint](https://www.pylint.org/). `pylint` checks for these instances in the code, then produces a list of labeled errors. Again, we use a custom length of 92 characters as the recommended max line length. To lint your code with these settings, you can run the following line from the `flepiMoP` directory:

```bash
pylint flepimop/gempyor_pkg/src/gempyor/ \
    --fail-under 5 \
    --rcfile flepimop/gempyor_pkg/.pylintrc \
    --verbose
```

For those using a Mac or Linux system for development, these commands are also available for execution by calling `./bin/lint`. Similarly, you can take advantage of the formatting pre-commit hook found at `bin/pre-commit`. To start using it copy this file to your git hooks folder:

```bash
cp -f bin/pre-commit .git/hooks/
```
