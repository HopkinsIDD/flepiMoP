"""
Helper functions for interacting with model I/O.
"""

from collections import Counter
from collections.abc import Iterable
import datetime
import functools
import logging
import numbers
import os
from pathlib import Path
import random
from shlex import quote as shlex_quote
import shutil
import subprocess
import time
from typing import Any, Callable, Literal, overload

import confuse
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import scipy.ndimage
import scipy.stats
import sympy.parsing.sympy_parser
import yaml

from . import file_paths
from ._pydantic_ext import _evaled_expression


logger = logging.getLogger(__name__)

config = confuse.Configuration("flepiMoP", read=False)


def write_df(
    fname: str | bytes | os.PathLike,
    df: pd.DataFrame,
    extension: Literal[None, "", "csv", "parquet"] = "",
) -> None:
    """Writes a pandas DataFrame without its index to a file.

    Writes a pandas DataFrame to either a CSV or Parquet file without its index and can
    infer which format to use based on the extension given in `fname` or based on
    explicit `extension`.

    Args:
        fname: The name of the file to write to.
        df: A pandas DataFrame whose contents to write, but without its index.
        extension: A user specified extension to use for the file if not contained in
            `fname` already.

    Returns:
        None

    Raises:
        NotImplementedError: The given output extension is not supported yet.
    """
    # Decipher the path given
    fname = fname.decode() if isinstance(fname, bytes) else fname
    path = Path(f"{fname}.{extension}") if extension else Path(fname)
    # Write df to either a csv or parquet or raise if an invalid extension
    if path.suffix == ".csv":
        return df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        return df.to_parquet(path, index=False, engine="pyarrow")
    raise NotImplementedError(
        f"Invalid extension provided: '.{path.suffix[1:]}'. Supported extensions are `.csv` or `.parquet`."
    )


def read_df(
    fname: str | bytes | os.PathLike,
    extension: Literal[None, "", "csv", "parquet"] = "",
) -> pd.DataFrame:
    """Reads a pandas DataFrame from either a CSV or Parquet file.

    Reads a pandas DataFrame to either a CSV or Parquet file and can infer which format
    to use based on the extension given in `fname` or based on explicit `extension`. If
    the file being read is a csv with a column called 'subpop' then that column will be
    cast as a string.

    Args:
        fname: The name of the file to read from.
        extension: A user specified extension to use for the file if not contained in
            `fname` already.

    Returns:
        A pandas DataFrame parsed from the file given.

    Raises:
        NotImplementedError: The given output extension is not supported yet.
    """
    # Decipher the path given
    fname = fname.decode() if isinstance(fname, bytes) else fname
    path = Path(f"{fname}.{extension}") if extension else Path(fname)
    # Read df from either a csv or parquet or raise if an invalid extension
    if path.suffix == ".csv":
        return pd.read_csv(
            path, converters={"subpop": lambda x: str(x)}, skipinitialspace=True
        )
    elif path.suffix == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    raise NotImplementedError(
        f"Invalid extension provided: '.{path.suffix[1:]}'. Supported extensions are `.csv` or `.parquet`."
    )


def command_safe_run(
    command: str, command_name: str = "mycommand", fail_on_fail: bool = True
) -> tuple[int, str, str]:
    """
    Runs a shell command and prints diagnostics if command fails.

    Args:
        command: The CLI command to be given.
        command_name: The reference name for you command. Default value is "mycommand".
        fail_on_fail: If True, an exception will be thrown if the command fails (default is True)

    Returns:
        As a tuple; the return code, the standard output, and standard error from running the command.

    Raises:
        RuntimeError: If fail_on_fail=True and the command fails, an error will be thrown.
    """
    import subprocess
    import shlex  # using shlex to split the command because it's not obvious https://docs.python.org/3/library/subprocess.html#subprocess.Popen

    sr = subprocess.Popen(
        shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    (stdout, stderr) = sr.communicate()
    if sr.returncode != 0:
        print(f"'{command_name}' failed failed with returncode '{sr.returncode}'")
        print(f"'{command_name}': '{command}'")
        print("{command_name} command failed with stdout and stderr:")

        print("{command_name} stdout >>>>>>")
        print(stdout.decode())
        print("{command_name} stdout <<<<<<")

        print("{command_name} stderr >>>>>>")
        print(stderr.decode())
        print("{command_name} stderr <<<<<<")
        if fail_on_fail:
            raise RuntimeError(f"The '{command_name}' command failed.")

    return sr.returncode, stdout, stderr


def add_method(cls: Any):
    """
    A function which adds a function to a class.

    Args:
        cls: The class you want to add a method to.

    Returns:
        decorator: The decorator.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


def search_and_import_plugins_class(
    plugin_file_path: str, path_prefix: str, class_name: str, **kwargs: dict[str, Any]
) -> Any:
    """
    Function serving to create a class that finds and imports the necessary modules.

    Args:
        plugin_file_path: Pathway to the module.
        path_prefix: Pathway prefix to the module.
        class_name: Name of the class.
        **kwargs: Further arguments passed to initilization of the class.

    Returns:
        The instance of the class that was instantiated with provided **kwargs.

    Examples:
        Suppose there is a module called `my_plugin.py with a class `MyClass` located at `/path/to/plugin/`.

        Dynamically import and instantiate the class:

        >>> instance = search_and_import_plugins_class('/path/to/plugin', path_prefix, 'MyClass', **params)

        View the instance:

        >>> print(instance)
        <__main__.MyClass object at 0x7f8b2c6b4d60>

    """
    # Look for all possible plugins and import them
    # https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
    # unfortunatelly very complicated, this is cpython only ??
    import sys, os

    full_path = os.path.join(path_prefix, plugin_file_path)
    sys.path.append(os.path.dirname(full_path))
    # the following works, but these above lines seems necessary to pickle // runs
    from pydoc import importfile

    module = importfile(full_path)
    klass = getattr(module, class_name)
    return klass(**kwargs)


### Profile configuration
import cProfile
import pstats
from functools import wraps


def profile(
    output_file: str = None,
    sort_by: str = "cumulative",
    lines_to_print: int = None,
    strip_dirs: bool = False,
):
    """
    A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/

    Args:
        output_file:
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by:
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print:
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs:
            Whether to remove the leading path info from file names.

    Returns:
        Profile of the decorated function.

    Examples:
        >>> @profile(output_file="my_function.prof")
        >>> def my_function():
            # Function body content
            pass
        >>> my_function()
        After running ``my_function``, a file named ``my_function.prof`` will be created in the current WD.
        This file contains the profiling data.
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)
            return retval

        return wrapper

    return inner


def as_list(thing: Any) -> list[Any]:
    """
    Returns argument passed as a list.

    Args:
        thing: The object that you would like to be converted to a list.

    Returns:
        The object converted to a list.
    """
    if type(thing) == list:
        return thing
    return [thing]


### A little timer class
class Timer(object):
    """
    A timer class that starts, ends, and records time in between.

    Attributes:
        name: Name of event.
        tstart: Time start.
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.debug(f"[{self.name}] started")
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        logging.debug(f"[{self.name}] completed in {time.time() - self.tstart:,.2f} s")


class ISO8601Date(confuse.Template):
    """
    Reads in config dates into datetimes.dates.
    """

    def convert(self, value: any, view: confuse.ConfigView):
        """
        Converts the given value to a datetime.date object.

        Args:
            value: The value to be converted. Can be datetime.date object or ISO8601 string.
            view:  A view object from confuse, to be used for error reporting.

        Raises:
            confuse.TemplateError: If `value` is neither a datetime.date nor an ISO8601Date string.
        """
        if isinstance(value, datetime.date):
            return value
        elif isinstance(value, str):
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        else:
            self.fail("must be a date object or ISO8601 date", True)


@add_method(confuse.ConfigView)
def as_date(self) -> datetime.date:
    """
    Evaluates an datetime.date or ISO8601 date string.

    Returns:
        A datetime.date data type of the date associated with the object.
    """

    return self.get(ISO8601Date())


def get_truncated_normal(
    mean: float | int = 0,
    sd: float | int = 1,
    a: float | int = 0,
    b: float | int = 10,
) -> scipy.stats._distn_infrastructure.rv_frozen:
    """
    Returns a truncated normal distribution.

    This function constructs a truncated normal distribution with the specified
    mean, standard deviation, and bounds. The truncated normal distribution is
    a normal distribution bounded within the interval [a, b].

    Args:
        mean: The mean of the truncated normal distribution. Defaults to 0.
        sd: The standard deviation of the truncated normal distribution. Defaults to 1.
        a: The lower bound of the truncated normal distribution. Defaults to 0.
        b: The upper bound of the truncated normal distribution. Defaults to 10.

    Returns:
        rv_frozen: A frozen instance of the truncated normal distribution with the specified parameters.

    Examples:
        Create a truncated normal distribution with specified parameters (truncated between 1 and 10):
        >>> truncated_normal_dist = get_truncated_normal(mean=5, sd=2, a=1, b=10)
        >>> print(truncated_normal_dist)
        rv_frozen(<scipy.stats._distn_infrastructure.rv_frozen object at 0x...>)
    """
    lower = (a - mean) / sd
    upper = (b - mean) / sd
    return scipy.stats.truncnorm(lower, upper, loc=mean, scale=sd)


def get_log_normal(
    meanlog: float | int,
    sdlog: float | int,
) -> scipy.stats._distn_infrastructure.rv_frozen:
    """
    Returns a log normal distribution.

    This function constructs a log normal distribution with the specified
    log mean and log standard deviation.

    Args:
        meanlog: The log of the mean of the log normal distribution.
        sdlog: The log of the standard deviation of the log normal distribution.

    Returns:
        rv_frozen: A frozen instance of the log normal distribution with the
        specified parameters.

    Examples:
        Create a log-normal distribution with specified parameters:
        >>> log_normal_dist = get_log_normal(meanlog=1, sdlog=0.5)
        >>> print(log_normal_dist)
        <scipy.stats._distn_infrastructure.rv_frozen object at 0x...>
    """
    return scipy.stats.lognorm(s=sdlog, scale=np.exp(meanlog), loc=0)


def random_distribution_sampler(
    distribution: Literal[
        "fixed", "uniform", "poisson", "binomial", "truncnorm", "lognorm"
    ],
    **kwargs: dict[str, Any],
) -> Callable[[], float | int]:
    """
    Create function to sample from a random distribution.

    Args:
        distribution: The type of distribution to generate a sampling function for.
        **kwargs: Further parameters that are passed to the underlying function for the
            given distribution.

    Notes:
        The further args expected by each distribution type are:
        - fixed: value,
        - uniform: low, high,
        - poisson: lam,
        - binomial: n, p,
        - truncnorm: mean, sd, a, b,
        - lognorm: meanlog, sdlog.

    Returns:
        A function that can be called to sample from that distribution.

    Raises:
        ValueError: If `distribution` is 'binomial' the given `p` must be in (0,1).
        NotImplementedError: If `distribution` is not one of the type hinted options.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(123)
        >>> uniform_sampler = random_distribution_sampler("uniform", low=0.0, high=3.0)
        >>> uniform_sampler()
        2.089407556793585
        >>> uniform_sampler()
        0.8584180048511384
    """
    if distribution == "fixed":
        # Fixed value is the same as uniform on [a, a)
        return functools.partial(
            np.random.uniform,
            kwargs.get("value"),
            kwargs.get("value"),
        )
    elif distribution == "uniform":
        # Uniform on [low, high)
        return functools.partial(
            np.random.uniform,
            kwargs.get("low"),
            kwargs.get("high"),
        )
    elif distribution == "poisson":
        # Poisson with mean lambda
        return functools.partial(np.random.poisson, kwargs.get("lam"))
    elif distribution == "binomial":
        p = kwargs.get("p")
        if not (0 < p < 1):
            raise ValueError(f"Invalid `p-value`: '{p}' is out of range [0,1].")
        return functools.partial(np.random.binomial, kwargs.get("n"), p)
    elif distribution == "truncnorm":
        # Truncated normal with mean, sd on interval [a, b]
        return get_truncated_normal(
            mean=kwargs.get("mean"),
            sd=kwargs.get("sd"),
            a=kwargs.get("a"),
            b=kwargs.get("b"),
        ).rvs
    elif distribution == "lognorm":
        # Lognormal distribution with meanlog, sdlog
        return get_log_normal(kwargs.get("meanlog"), kwargs.get("sdlog")).rvs
    raise NotImplementedError(f"Unknown distribution [received: '{distribution}'].")


@add_method(confuse.ConfigView)
def as_random_distribution(self):
    """
    Constructs a random distribution object from a distribution config key.

    Args:
        self: Class instance (in this case, a config key) to construct the random distribution from.

    Returns:
        A partial object containing the random distribution.

    Raises:
        ValueError: When values are out of range.
        NotImplementedError: If an unknown distribution is found.

    Examples:
        Say that ``config`` is a ``confuse.ConfigView`` instance.

        To create a uniform distribution between 1 and 10:
        >>> dist_function = config.as_random_distribution()
        >>> sample = dist_function()
        5.436789235794546

        To use a truncated normal distribution:
        >>> config_truncnorm = confuse.ConfigView({
            "distribution": "truncnorm",
            "mean": 0
            "sd": 1,
            "a": -1,
            "b": 1
            })
        >>> truncnorm_dist_function = config_truncnorm.as_random_distribution()
        >>> truncnorm_sample = truncnorm_dist_function()
        0.312745
        ```
    """

    if isinstance(self.get(), dict):
        dist = self["distribution"].get()
        if dist == "fixed":
            return functools.partial(
                np.random.uniform,
                _evaled_expression(self["value"].get(), target_type=float),
                _evaled_expression(self["value"].get(), target_type=float),
            )
        elif dist == "uniform":
            return functools.partial(
                np.random.uniform,
                _evaled_expression(self["low"].get(), target_type=float),
                _evaled_expression(self["high"].get(), target_type=float),
            )
        elif dist == "poisson":
            return functools.partial(np.random.poisson, _evaled_expression(self["lam"].get(), target_type=float))
        elif dist == "binomial":
            p = _evaled_expression(self["p"].get(), target_type=float)
            if (p < 0) or (p > 1):
                raise ValueError(f"Invalid `p-value`: '{p}' is out of range [0,1].")
                # if (self["p"] < 0) or (self["p"] > 1):
                #    raise ValueError(f"""p value { self["p"] } is out of range [0,1]""")
            return functools.partial(
                np.random.binomial,
                _evaled_expression(self["n"].get(), target_type=float),
                # _evaled_expression(self["p"].get(), target_type=float),
                p,
            )
        elif dist == "truncnorm":
            return get_truncated_normal(
                mean=_evaled_expression(self["mean"].get(), target_type=float),
                sd=_evaled_expression(self["sd"].get(), target_type=float),
                a=_evaled_expression(self["a"].get(), target_type=float),
                b=_evaled_expression(self["b"].get(), target_type=float),
            ).rvs
        elif dist == "lognorm":
            return get_log_normal(
                meanlog=_evaled_expression(self["mean_log"].get(), target_type=float),
                sdlog=_evaled_expression(self["sd_log"].get(), target_type=float),
            ).rvs
        else:
            raise NotImplementedError(f"Unknown distribution [received: '{dist}'].")
    else:
        # we allow a fixed value specified directly:
        return functools.partial(
            np.random.uniform,
            _evaled_expression(self, target_type=float),
            _evaled_expression(self, target_type=float),
        )


def list_filenames(
    folder: str | bytes | os.PathLike = ".",
    filters: str | list[str] = [],
) -> list[str]:
    """
    Return the list of all filenames and paths in the provided folder.

    This function lists all files in the specified folder and its subdirectories.
    If filters are provided, only the files containing each of the substrings
    in the filters will be returned.

    Args:
        folder:
            The directory to search for files. Defaults to the current directory.
        filters:
            A string or a list of strings to filter filenames. Only files
            containing all the provided substrings will be returned. Defaults to an
            empty list.

    Returns:
        A list of strings representing the paths to the files that match the filters.

    Examples:
        To get all files containing "hosp":
        >>> gempyor.utils.list_filenames(
            folder="model_output/",
            filters=["hosp"],
        )

        To get only "hosp" files with a ".parquet" extension:
        >>> gempyor.utils.list_filenames(
            folder="model_output/",
            filters=["hosp", ".parquet"],
        )
    """
    filters = [filters] if not isinstance(filters, list) else filters
    filters = filters if len(filters) else [""]
    folder = Path(folder.decode() if isinstance(folder, bytes) else folder)
    files = [
        str(file)
        for file in folder.rglob("*")
        if file.is_file() and all(f in str(file) for f in filters)
    ]
    return files


def extract_slot(file: Path) -> pd.DataFrame:
    """
    Extract the slot number from the filename and add to DataFrame.

    Args:
        file: The filename to extract the slot number from.

    Returns:
        A pandas DataFrame containing the contents of `file` along with a column called
        'slot' that contains the slot number extracted from the filename.

    Raises:
        ValueError: If the 'slot' column already exists in the DataFrame read in.

    Examples:
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from gempyor.utils import extract_slot
        >>> sample = pd.DataFrame(data={"letters": ["a", "b", "c"]})
        >>> file = Path.cwd() / "00017.foobar.csv"
        >>> sample.to_csv(file, index=False)
        >>> extract_slot(file)
        letters  slot
        0       a    17
        1       b    17
        2       c    17
        >>> extract_slot(file).info()
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 2 columns):
        #   Column   Non-Null Count  Dtype
        ---  ------   --------------  -----
        0   letters  3 non-null      object
        1   slot     3 non-null      int64
        dtypes: int64(1), object(1)
        memory usage: 180.0+ bytes
    """
    df = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file)
    if "slot" in df.columns:
        raise ValueError(
            f"Column 'slot' already exists in the DataFrame from file: {file}."
        )
    df["slot"] = int(file.name.split(".")[0])
    return df


def read_directory(
    directory: str | bytes | os.PathLike, filters: str | list[str] | None = None
) -> pd.DataFrame:
    """
    Read all files in a directory into a single DataFrame.

    Args:
        directory: The directory to read files from.
        filters: A string or a list of strings to filter filenames. Only files
            containing all the provided substrings will be read. Defaults to `None` for
            no filters.

    Returns:
        A pandas DataFrame containing the contents of all the files in the directory.
    """
    files = [Path(file) for file in list_filenames(folder=directory, filters=filters or [])]
    dfs: list[pd.DataFrame] | pd.DataFrame = []
    for file in sorted(files):
        dfs.append(extract_slot(file))
    return pd.concat(dfs).reset_index(drop=True)


def rolling_mean_pad(
    data: npt.NDArray[np.number],
    window: int,
) -> npt.NDArray[np.number]:
    """
    Calculates the column-wise rolling mean with centered window.

    Args:
        data: A two dimensional numpy array, typically the row dimension is time and the
            column dimension is subpop.
        window: The window size for the rolling mean.

    Returns:
        A two dimensional numpy array that is the same shape as `data`.

    Examples:
        Below is a brief set of examples showcasing how to smooth a metric, like
        hospitalizations, using this function.
        ```
        >>> import numpy as np
        >>> from gempyor.utils import rolling_mean_pad
        >>> hospitalizations = np.arange(1., 29.).reshape((7, 4))
        >>> hospitalizations
        array([[ 1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.],
            [ 9., 10., 11., 12.],
            [13., 14., 15., 16.],
            [17., 18., 19., 20.],
            [21., 22., 23., 24.],
            [25., 26., 27., 28.]])
        >>> rolling_mean_pad(hospitalizations, 5)
        array([[ 3.4,  4.4,  5.4,  6.4],
            [ 5.8,  6.8,  7.8,  8.8],
            [ 9. , 10. , 11. , 12. ],
            [13. , 14. , 15. , 16. ],
            [17. , 18. , 19. , 20. ],
            [20.2, 21.2, 22.2, 23.2],
            [22.6, 23.6, 24.6, 25.6]])
        ```
    """
    weights = (1.0 / window) * np.ones(window)
    output = scipy.ndimage.convolve1d(data, weights, axis=0, mode="nearest")
    if window % 2 == 0:
        rows, cols = data.shape
        i = rows - 1
        output[i, :] = 0.0
        window -= 1
        weight = 1.0 / window
        for l in range(-((window - 1) // 2), 1 + (window // 2)):
            i_star = min(max(i + l, 0), i)
            for j in range(cols):
                output[i, j] += weight * data[i_star, j]
    return output


def print_disk_diagnosis():
    """
    Reads and prints AWS disk diagnostic information.
    """
    import os
    from os import path
    from shutil import disk_usage

    def bash(command: str) -> str:
        """
        Executes a shell command and returns its output.

        Args:
            command: The shell command to be executed.

        Returns:
            The output of the shell command.
        """
        output = os.popen(command).read()
        return output

    print("START AWS DIAGNOSIS ================================")
    total_bytes, used_bytes, free_bytes = disk_usage(path.realpath("/"))
    print(
        f"shutil.disk_usage: {total_bytes/ 1000000} Mb total, {used_bytes / 1000000} Mb used, {free_bytes / 1000000} Mb free..."
    )
    print("------------")
    print(f"df -hT: '{bash('df -hT')}'")
    print("------------")
    print(f"df -i: '{bash('df -i')}'")
    print("------------")
    print(f"free -h: '{bash('free -h')}'")
    print("------------")
    print(f"lsblk: '{bash('lsblk')}'")
    print("END AWS DIAGNOSIS ================================")


def create_resume_out_filename(
    flepi_run_index: str,
    flepi_prefix: str,
    flepi_slot_index: str,
    flepi_block_index: str,
    filetype: str,
    liketype: str,
) -> str:
    """
    Compiles run output information.

    Args:
        flepi_run_index: Index of the run.
        flepi_prefix: File prefix.
        flepi_slot_index: Index of the slot.
        flepi_block_index: Index of the block.
        filetype: File type.
        liketype: Chimeric or global.

    Returns:
        The path to a corresponding output file.

    Examples:
        Generate an output file with specified parameters:
        >>> filename = create_resume_out_filename(
            flepi_run_index="test_run",
            flepi_prefix="model_output/run_id/",
            flepi_slot_index="1",
            flepi_block_index="2",
            filetype="seed",
            liketype="chimeric"
            )
        >>> print(filename)
        "experiment/001/normal/intermediate/000000123.000000000.1.parquet"
    """
    prefix = f"{flepi_prefix}/{flepi_run_index}"
    inference_filepath_suffix = f"{liketype}/intermediate"
    inference_filename_prefix = "{:09d}.".format(int(flepi_slot_index))
    index = "{:09d}.{:09d}".format(1, int(flepi_block_index) - 1)
    extension = "parquet"
    if filetype == "seed":
        extension = "csv"
    return file_paths.create_file_name(
        run_id=flepi_run_index,
        prefix=prefix,
        inference_filename_prefix=inference_filename_prefix,
        inference_filepath_suffix=inference_filepath_suffix,
        index=index,
        ftype=filetype,
        extension=extension,
    )


def create_resume_input_filename(
    resume_run_index: str,
    flepi_prefix: str,
    flepi_slot_index: str,
    filetype: str,
    liketype: str,
) -> str:
    """
    Compiles run input information.

    Args:
        resume_run_index: Index of the run.
        flepi_prefix: File prefix.
        flepi_slot_index: Index of the slot.
        filetype: File type.
        liketype: Chimeric or global.

    Returns:
        The path to the a corresponding input file.

    Examples:
        Generate an input file with specified parameters:
        >>> filename = create_resume_input_filename(
            resume_run_index="2",
            flepi_prefix="model_output/run_id/",
            flepi_slot_index="1",
            filetype="seed",
            liketype="chimeric"
            )
        >>> print(filename)
        "experiment/002/normal/final/789.csv"
    """
    prefix = f"{flepi_prefix}/{resume_run_index}"
    inference_filepath_suffix = f"{liketype}/final"
    index = flepi_slot_index
    extension = "parquet"
    if filetype == "seed":
        extension = "csv"
    return file_paths.create_file_name(
        run_id=resume_run_index,
        prefix=prefix,
        inference_filepath_suffix=inference_filepath_suffix,
        index=index,
        ftype=filetype,
        extension=extension,
    )


def get_filetype_for_resume(
    resume_discard_seeding: str, flepi_block_index: str
) -> list[str]:
    """
    Retrieves a list of parquet file types that are relevant for resuming a process based on
    specific environment variable settings.
    This function dynamically determines the list
    based on the current operational context given by the environment.

    Args:
        resume_discard_seeding: Determines whether seeding-related file types should be included (str).
        flepi_block_index: Determines a specific operational mode or block of the process (str).

    Returns:
        List of file types.

    Examples:
        Determine file types for block index 1 with seeding data NOT discarded:
        >>> filetypes = get_filetype_for_resume(resume_discard_seeding="false", flepi_block_index="1")
        >>> print(filetypes)
        ["seed", "spar", "snpi", "hpar", "hnpi", "init"]

        Determine file types for block index 2 with seeding data discarded:
        >>> filtypes = get_filetype_for_resume(resume_discard_seeding="true", flepi_block_index="2")
        >>> print(filetypes)
        ["seed", "spar", "snpi", "hpar", "hnpi", "host", "llik", "init"]
    """
    if flepi_block_index == "1":
        if resume_discard_seeding == "true":
            return ["spar", "snpi", "hpar", "hnpi", "init"]
        else:
            return ["seed", "spar", "snpi", "hpar", "hnpi", "init"]
    else:
        return ["seed", "spar", "snpi", "hpar", "hnpi", "host", "llik", "init"]


def create_resume_file_names_map(
    resume_discard_seeding: str,
    flepi_block_index: str,
    resume_run_index: str,
    flepi_prefix: str,
    flepi_slot_index: str,
    flepi_run_index: str,
    last_job_output: str,
) -> dict[str, str]:
    """
    Generates a mapping of input file names to output file names for a resume process based on
    parquet file types and environmental conditions. The function adjusts the file name mappings
    based on the operational block index and the location of the last job output.

    Args:
        resume_discard_seeding:  Determines whether seeding-related file types should be included.
        flepi_block_index: Determines a specific operational mode or block of the process.
        resume_run_index: Resume run index.
        flepi_prefix: File prefix.
        flepi_slot_index: Index of the slot.
        flepi_run_index: flepiMoP run index.
        last_job_output: Adjusts the keys in the mapping to be prefixed with this path.

    Returns:
        A dictionary where keys are input file paths and values are corresponding
        output file paths.

    The mappings depend on:
    - Parquet file types appropriate for resuming a process, as determined by the environment.
    - Whether the files are for 'global' or 'chimeric' types, as these liketypes influence the
      file naming convention.
    - The operational block index ('FLEPI_BLOCK_INDEX'), which can alter the input file names for
      block index '1'.
    - The presence and value of 'LAST_JOB_OUTPUT' environment variable, which if set to an S3 path,
      adjusts the keys in the mapping to be prefixed with this path.

    Raises:
        No explicit exceptions are raised within the function, but it relies heavily on external
        functions and environment variables which if improperly configured could lead to unexpected
        behavior.

    Examples:
        Generate a mapping of file names for a given resume process:
        >>> file_names_map = create_resume_file_names_map(
            resume_discard_seeding="false",
            flepi_block_index="1",
            resume_run_index="1",
            flepi_prefix="model_output/run_id/",
            flepi_slot_index="1",
            flepi_run_index="test_run",
            last_job_output="s3://bucket/path/")
        >>> print(file_names_map)
        {
        's3://bucket/path/model_output/run_id/1_type1_global_1.in': 'model_output/run_id/test_run_type1_global_1_1.out',
        's3://bucket/path/model_output/run_id/1_type1_chimeric_1.in': 'model_output/run_id/test_run_type1_chimeric_1_1.out',
        's3://bucket/path/model_output/run_id/1_type2_global_1.in': 'model_output/run_id/test_run_type2_global_1_1.out',
        's3://bucket/path/model_output/run_id/1_type2_chimeric_1.in': 'model_output/run_id/test_run_type2_chimeric_1_1.out'
        }
        # Note: this output is toy output implemented with toy file names.

    Notes:
        - The paths may be modified by the 'LAST_JOB_OUTPUT' if it is set and points to an S3 location.
    """
    file_types = get_filetype_for_resume(
        resume_discard_seeding=resume_discard_seeding, flepi_block_index=flepi_block_index
    )
    resume_file_name_mapping = dict()
    liketypes = ["global", "chimeric"]
    for filetype in file_types:
        for liketype in liketypes:
            output_file_name = create_resume_out_filename(
                flepi_run_index=flepi_run_index,
                flepi_prefix=flepi_prefix,
                flepi_slot_index=flepi_slot_index,
                flepi_block_index=flepi_block_index,
                filetype=filetype,
                liketype=liketype,
            )
            input_file_name = output_file_name
            if flepi_block_index == "1":
                input_file_name = create_resume_input_filename(
                    resume_run_index=resume_run_index,
                    flepi_prefix=flepi_prefix,
                    flepi_slot_index=flepi_slot_index,
                    filetype=filetype,
                    liketype=liketype,
                )
            resume_file_name_mapping[input_file_name] = output_file_name
    if last_job_output.find("s3://") >= 0:
        old_keys = list(resume_file_name_mapping.keys())
        for k in old_keys:
            new_key = os.path.join(last_job_output, k)
            resume_file_name_mapping[new_key] = resume_file_name_mapping[k]
            del resume_file_name_mapping[k]
    return resume_file_name_mapping


def download_file_from_s3(name_map: dict[str, str]) -> None:
    """
    Downloads files from AWS S3 based on a mapping of S3 URIs to local file paths. The function
    checks if the directory for the first output file exists and creates it if necessary. It
    then iterates over each S3 URI in the provided mapping, downloads the file to the corresponding
    local path, and handles errors if the S3 URI format is incorrect or if the download fails.

    Args:
        name_map:
            A dictionary where keys are S3 URIs (strings) and values
            are the local file paths (strings) where the files should
            be saved.

    Returns:
        This function does not return a value; its primary effect is the side effect of
        downloading files and potentially creating directories.

    Raises:
        ValueError: If an S3 URI does not start with 's3://', indicating an invalid format.
        ClientError: If an error occurs during the download from S3, such as a permissions issue,
                     a missing file, or network-related errors. These are caught and logged but not
                     re-raised, to allow the function to attempt subsequent downloads.

    Examples:
        >>> name_map = {
            "s3://mybucket/data/file1.txt": "/local/path/to/file1.txt",
            "s3://mybucket/data/file2.txt": "/local/path/to/file2.txt"
        }
        >>> download_file_from_s3(name_map)
        # This would download 'file1.txt' and 'file2.txt' from 'mybucket' on S3 to the specified local paths.

        # If an S3 URI is malformed:
        >>> name_map = {
            "http://wrongurl.com/data/file1.txt": "/local/path/to/file1.txt"
        }
        >>> download_file_from_s3(name_map)
        # This will raise a ValueError indicating the invalid S3 URI format.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            (
                "No module named `boto3` found, which is required for "
                "`gempyor.utils.download_file_from_s3`. Please install the aws target."
            )
        )
    s3 = boto3.client("s3")
    first_output_filename = next(iter(name_map.values()))
    output_dir = os.path.dirname(first_output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for s3_uri in name_map:
        try:
            if s3_uri.startswith("s3://"):
                bucket = s3_uri.split("/")[2]
                object = s3_uri[len(bucket) + 6 :]
                s3.download_file(bucket, object, name_map[s3_uri])
            else:
                raise ValueError(f"Invalid S3 URI format [received: '{s3_uri}'].")
        except ClientError as e:
            raise Exception(f"'{e}': could not download filefrom S3.")


def move_file_at_local(name_map: dict[str, str]) -> None:
    """
    Moves files locally according to a given mapping.
    This function takes a dictionary where the keys are source file paths and
    the values are destination file paths. It ensures that the destination
    directories exist and then copies the files from the source paths to the
    destination paths.

    Args:
        name_map: A dictionary mapping source file paths to destination file paths.
    """
    for src, dst in name_map.items():
        os.path.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)


def _dump_formatted_yaml(cfg: confuse.Configuration) -> str:
    """
    Dump confuse configuration to a formatted YAML string.

    Args:
        cfg: The confuse configuration object.

    Returns:
        A formatted YAML string representation of the configuration.

    Examples:
        >>> from gempyor.utils import _dump_formatted_yaml
        >>> import confuse
        >>> conf = confuse.Configuration("foobar")
        >>> data = {
        ...     "name": "Test Config",
        ...     "compartments": {
        ...         "infection_stage": ["S", "E", "I", "R"]
        ...     },
        ...     "seir": {
        ...         "parameters": {
        ...             "beta": {"value": 3.4},
        ...             "gamma": {"value": 5.6},
        ...         },
        ...         "transitions": {
        ...             "source": ["S"],
        ...             "destination": ["E"],
        ...             "rate": ["beta * gamma"],
        ...             "proportional_to": [["S"], ["I"]],
        ...             "proportion_exponent": [1, 1],
        ...         },
        ...     },
        ... }
        >>> conf.set(data)
        >>> print(_dump_formatted_yaml(conf))
        name: "Test Config"
        compartments:
            infection_stage: [S, E, I, R]
        seir:
            parameters:
                beta:
                    value: 3.4
                gamma:
                    value: 5.6
            transitions:
                source: [S]
                destination: [E]
                rate: ["beta * gamma"]
                proportional_to: [[S], [I]]
                proportion_exponent: [1, 1]
    """

    class CustomDumper(yaml.Dumper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_representer(list, self._represent_list)
            self.add_representer(str, self._represent_str)

        def _represent_list(self, dumper, data):
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

        def _represent_str(self, dumper, data):
            if " " in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    return yaml.dump(
        yaml.safe_load(cfg.dump()), Dumper=CustomDumper, indent=4, sort_keys=False
    )


def _shutil_which(
    cmd: str,
    mode: int = os.F_OK | os.X_OK,
    path: str | bytes | os.PathLike | None = None,
    check: bool = True,
) -> str | None:
    """
    A thin wrapper around `shutil.which` with extra validation.

    Args:
        cmd: The name of the command to search for.
        mode: The permission mask required of possible files.
        path: A path describing the locations to search, or `None` to use
            the PATH environment variable.
        check: If `True` an `OSError` will be raised if a `cmd` is not found.
            Similar in spirit to the `check` arg of `subprocess.run`.

    Returns:
        Either the full path to the `cmd` found, or `None` if `cmd` is not
        found and `check` is `False`.

    Raises:
        OSError: If `cmd` is not found and `check` is `True`.

    Examples:
        >>> import os
        >>> from shutil import which
        >>> _shutil_which("python") == which("python")
        True
        >>> _shutil_which("does_not_exist", check=False)
        >>> try:
        ...     _shutil_which("does_not_exist", check=True)
        ... except Exception as e:
        ...     print(type(e))
        ...     print(str(e).replace(os.environ.get("PATH"), "..."))
        ...
        <class 'OSError'>
        Did not find 'does_not_exist' on path '...'.

    See Also:
        [`shutil.which`](https://docs.python.org/3/library/shutil.html#shutil.which)
    """
    result = shutil.which(cmd, mode=mode, path=path)
    if check and result is None:
        path = os.environ.get("PATH") if path is None else path
        raise OSError(f"Did not find '{cmd}' on path '{path}'.")
    return result


def _git_head(repository: Path) -> str:
    """
    Get the sha commit hash for the head of a git repository.

    Args:
        repository: A directory under version control with git to get the sha commit of.

    Returns:
        The sha commit of head for `repository`.

    Examples:
        >>> import os
        >>> from pathlib import Path
        >>> _git_head(Path(os.environ["FLEPI_PATH"]))
        'efe896b1a5e4f8e33667c170cd5319d6ef1e3db5'
    """
    git_cmd = _shutil_which("git")
    proc = subprocess.run(
        [git_cmd, "rev-parse", "HEAD"],
        cwd=repository.expanduser().absolute(),
        capture_output=True,
        check=True,
    )
    return proc.stdout.decode().strip()


def _git_checkout(repository: Path, branch: str) -> subprocess.CompletedProcess:
    """
    Checkout a new branch from a given git repository.

    Args:
        repository: A directory under version control with git to
            checkout a new branch in.
        branch: The name of the new branch to checkout.

    Examples:
        >>> import os
        >>> from pathlib import Path
        >>> _git_checkout(Path(os.environ["FLEPI_PATH"]), "my-new-branch")
    """
    git_cmd = _shutil_which("git")
    return subprocess.run(
        [git_cmd, "checkout", "-b", shlex_quote(branch)],
        cwd=repository.expanduser().absolute(),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _format_cli_options(
    options: dict[str, str | Iterable[str]] | None,
    always_single: bool = False,
    iterable_formatting: Literal["split", "comma"] = "split",
) -> list[str]:
    """
    Convert a dictionary of CLI options into a formatted list.

    Args:
        options: A dictionary where the keys correspond to the option name and the
            values correspond to the option value. Values are escaped for shell.
        always_single: If `True` all options will be formatted with a single dash prefix
            always, useful for commands that don't support long options like
            `pdftotext`. If `False` the option will be formatted with a double dash
            prefix if the key is longer than one character.
        iterable_formatting: The formatting to use for iterable values. If 'split' each
            value will be formatted as a separate option. If 'comma' the values will be
            joined with a comma and formatted as a single option.

    Returns:
        A list of options that can be passed to
        [`subprocess.run`](https://docs.python.org/3/library/subprocess.html#subprocess.run)
        or similar functions.

    Examples:
        >>> from gempyor.utils import _format_cli_options
        >>> _format_cli_options({"name": "foo bar fizz buzz"})
        ["--name='foo bar fizz buzz'"]
        >>> _format_cli_options({"o": "/path/to/output.log"})
        ['-o=/path/to/output.log']
        >>> _format_cli_options({"opt1": "```", "opt2": "$( echo 'Hello!')"})
        ["--opt1='```'", '--opt2=\'$( echo \'"\'"\'Hello!\'"\'"\')\'']
        >>> _format_cli_options({"output": "/path/to/output.log"}, always_single=True)
        ['-output=/path/to/output.log']
        >>> _format_cli_options({"person": ["Alice", "Bob", "Charlie"]})
        ['--person=Alice', '--person=Bob', '--person=Charlie']
        >>> _format_cli_options(
        ...     {"person": ["Alice", "Bob", "Charlie"]},
        ...     iterable_formatting="comma",
        ... )
        ['--person=Alice,Bob,Charlie']
    """
    opts = []
    for k, v in (options or {}).items():
        new_opts = []
        opt_name = f"{'-' if (always_single or len(k) == 1) else '--'}{k}"
        if isinstance(v, str):
            new_opts.append(f"{opt_name}={shlex_quote(v)}")
        elif iterable_formatting == "comma":
            new_opts.append(f"{opt_name}={','.join(shlex_quote(w) for w in v)}")
        else:
            new_opts.extend(f"{opt_name}={shlex_quote(w)}" for w in v)
        opts.extend(new_opts)
    return opts


def _random_seeds_list(a: int, b: int, n: int) -> list[int]:
    """
    Generate a list of unique random integers within a specified range.

    Args:
        a: The lower bound of the range, inclusive.
        b: The upper bound of the range, inclusive.
        n: The number of unique random integers to generate.

    Returns:
        A list of `n` unique integers between `a` and `b`.

    Raises:
        ValueError: If the lower bound is not less than the upper bound.
        ValueError: If the range is too small to accommodate `n` unique integers.

    Examples:
        >>> import random
        >>> from gempyor.utils import _random_seeds_list
        >>> random.seed(42)
        >>> _random_seeds_list(1, 100, 5)
        [82, 15, 4, 95, 36]
        >>> _random_seeds_list(1, 100, 5)
        [32, 29, 18, 95, 14]
        >>> _random_seeds_list(1, 5, 5)
        [5, 1, 4, 2, 3]
        >>> _random_seeds_list(1, 20, 0)
        []
        >>> try:
        ...     _random_seeds_list(1, 5, 6)
        ... except Exception as e:
        ...     print(e)
        ...
        Range [1, 5] is too small for 6 unique random integers.
    """
    random_seeds = []
    if a >= b:
        raise ValueError(f"The lower bound, {a}, must be less than the upper bound, {b}.")
    if n < 0:
        raise ValueError("The number of random integers to generate must be non-negative.")
    if (b - a) + 1 < n:
        raise ValueError(f"Range [{a}, {b}] is too small for {n} unique random integers.")
    while len(random_seeds) < n:
        seed = random.randint(a, b)
        if seed not in random_seeds:
            random_seeds.append(seed)
    return random_seeds


def _nslots_random_seeds(nslots: int) -> list[int]:
    """
    Generate a list of unique random integers for the number of slots.

    This function generates a list of `nslots` unique random integers between
    [`nslots` + 1, 2^32 + 1]. Intended for use in generating random seeds for individual
    slots when simulations are done in parallel.

    Args:
        nslots: The number of slots to generate random integers for.

    Examples:
        >>> import random
        >>> from gempyor.utils import _nslots_random_seeds
        >>> random.seed(42)
        >>> _nslots_random_seeds(5)
        [2746317219, 478163333, 107420375, 3184935169, 1181241949]
        >>> _nslots_random_seeds(5)
        [1051802518, 958682852, 599310831, 3163119791, 440213421]
        >>> _nslots_random_seeds(1)
        [2906402159]
        >>> _nslots_random_seeds(1)
        [3181143733]
    """
    return _random_seeds_list(nslots + 1, 2**32 - 1, nslots)


def _duplicate_strings(it: Iterable[str]) -> list[str]:
    """
    Efficiently find duplicate strings in an iterable.

    Args:
        it: An iterable of strings.

    Returns:
        A list of strings that appear more than once in `it`.

    Examples:
        >>> from gempyor.utils import _duplicate_strings
        >>> _duplicate_strings(["a", "b", "c"])
        []
        >>> _duplicate_strings(["a", "b", "c", "b"])
        ['b']
        >>> _duplicate_strings(["a", "b", "c", "b", "a"])
        ['a', 'b']
        >>> _duplicate_strings("foobar")
        ['o']
    """
    return sorted([k for k, v in Counter(it).items() if v > 1])


@overload
def _trim_s3_path(path: str) -> str: ...


@overload
def _trim_s3_path(path: Path) -> Path: ...


def _trim_s3_path(path: str | Path) -> str | Path:
    """
    Strips the 's3:' prefix from a given path.

    Args:
        path: The path to be trimmed. Can be a string or a Path object.

    Returns:
        The trimmed path as the same type as the input.

    Examples:
        >>> from pathlib import Path
        >>> from gempyor.utils import _trim_s3_path
        >>> _trim_s3_path("/abc/def/ghi.txt")
        '/abc/def/ghi.txt'
        >>> _trim_s3_path(Path("/abc/def/ghi.txt"))
        PosixPath('/abc/def/ghi.txt')
        >>> _trim_s3_path("s3://foo/bar.txt")
        '//foo/bar.txt'
        >>> _trim_s3_path(Path("s3://foo/bar.txt"))
        PosixPath('s3:/foo/bar.txt')
    """
    return path.lstrip("s3:") if isinstance(path, str) else path
