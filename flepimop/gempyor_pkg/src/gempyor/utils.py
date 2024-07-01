import os
import datetime
import functools
import numbers
import time
import confuse
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.stats
import sympy.parsing.sympy_parser
import subprocess
import shutil
import logging
import boto3
from gempyor import file_paths
from typing import List, Dict
from botocore.exceptions import ClientError
from pathlib import Path

logger = logging.getLogger(__name__)

config = confuse.Configuration("flepiMoP", read=False)


def write_df(
    fname: str | bytes | os.PathLike, 
    df: pd.DataFrame, 
    extension: str = "",
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
        f"Invalid extension {extension}. Must be 'csv' or 'parquet'"
    )


def read_df(fname: str, extension: str = "") -> pd.DataFrame:
    """Load a dataframe from a file, agnostic to whether it is a parquet or a csv. The extension
    can be provided as an argument or it is infered"""
    fname = str(fname)
    if extension:  # Empty strings are falsy in python
        fname = f"{fname}.{extension}"
    extension = fname.split(".")[-1]
    if extension == "csv":
        # The converter prevents e.g leading geoid (0600) to be converted as int; and works when the column is absent
        df = pd.read_csv(fname, converters={"subpop": lambda x: str(x)}, skipinitialspace=True)
    elif extension == "parquet":
        df = pa.parquet.read_table(fname).to_pandas()
    else:
        raise NotImplementedError(f"Invalid extension {extension}. Must be 'csv' or 'parquet'")
    return df


def command_safe_run(command, command_name="mycommand", fail_on_fail=True):
    import subprocess
    import shlex  # using shlex to split the command because it's not obvious https://docs.python.org/3/library/subprocess.html#subprocess.Popen

    sr = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = sr.communicate()
    if sr.returncode != 0:
        print(f"{command_name} failed failed with returncode {sr.returncode}")
        print(f"{command_name}:  {command}")
        print("{command_name} command failed with stdout and stderr:")

        print("{command_name} stdout >>>>>>")
        print(stdout.decode())
        print("{command_name} stdout <<<<<<")

        print("{command_name} stderr >>>>>>")
        print(stderr.decode())
        print("{command_name} stderr <<<<<<")
        if fail_on_fail:
            raise Exception(f"{command_name} command failed")

    return sr.returncode, stdout, stderr


def add_method(cls):
    "Decorator to add a method to a class"

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


def search_and_import_plugins_class(plugin_file_path: str, path_prefix: str, class_name: str, **kwargs):
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


def profile(output_file=None, sort_by="cumulative", lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by and modified the profile decorator of Giampaolo Rodola:
    http://code.activestate.com/recipes/577817-profile-decorator/
    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
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


def as_list(thing):
    if type(thing) == list:
        return thing
    return [thing]


### A little timer class
class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.debug(f"[{self.name}] started")
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        logging.debug(f"[{self.name}] completed in {time.time() - self.tstart:,.2f} s")


class ISO8601Date(confuse.Template):
    def convert(self, value, view):
        if isinstance(value, datetime.date):
            return value
        elif isinstance(value, str):
            return datetime.datetime.strptime(value, "%Y-%m-%d").date()
        else:
            self.fail("must be a date object or ISO8601 date", True)


@add_method(confuse.ConfigView)
def as_date(self):
    "Evaluates an datetime.date or ISO8601 date string, raises ValueError on parsing errors."

    return self.get(ISO8601Date())


@add_method(confuse.ConfigView)
def as_evaled_expression(self):
    "Evaluates an expression string, returning a float. Raises ValueError on parsing errors."

    value = self.get()
    if isinstance(value, numbers.Number):
        return value
    elif isinstance(value, str):
        try:
            return float(sympy.parsing.sympy_parser.parse_expr(value))
        except TypeError as e:
            raise ValueError(e) from e
    else:
        raise ValueError(f"expected numeric or string expression [got: {value}]")


def get_truncated_normal(*, mean=0, sd=1, a=0, b=10):
    "Returns the truncated normal distribution"

    return scipy.stats.truncnorm((a - mean) / sd, (b - mean) / sd, loc=mean, scale=sd)


def get_log_normal(meanlog, sdlog):
    "Returns the log normal distribution"
    return scipy.stats.lognorm(s=sdlog, scale=np.exp(meanlog), loc=0)


@add_method(confuse.ConfigView)
def as_random_distribution(self):
    "Constructs a random distribution object from a distribution config key"

    if isinstance(self.get(), dict):
        dist = self["distribution"].get()
        if dist == "fixed":
            return functools.partial(
                np.random.uniform, self["value"].as_evaled_expression(), self["value"].as_evaled_expression(),
            )
        elif dist == "uniform":
            return functools.partial(
                np.random.uniform, self["low"].as_evaled_expression(), self["high"].as_evaled_expression(),
            )
        elif dist == "poisson":
            return functools.partial(np.random.poisson, self["lam"].as_evaled_expression())
        elif dist == "binomial":
            p = self["p"].as_evaled_expression()
            if (p < 0) or (p > 1):
                raise ValueError(f"""p value { p } is out of range [0,1]""")
                # if (self["p"] < 0) or (self["p"] > 1):
                #    raise ValueError(f"""p value { self["p"] } is out of range [0,1]""")
            return functools.partial(
                np.random.binomial,
                self["n"].as_evaled_expression(),
                # self["p"].as_evaled_expression(),
                p,
            )
        elif dist == "truncnorm":
            return get_truncated_normal(
                mean=self["mean"].as_evaled_expression(),
                sd=self["sd"].as_evaled_expression(),
                a=self["a"].as_evaled_expression(),
                b=self["b"].as_evaled_expression(),
            ).rvs
        elif dist == "lognorm":
            return get_log_normal(
                meanlog=self["meanlog"].as_evaled_expression(), sdlog=self["sdlog"].as_evaled_expression(),
            ).rvs
        else:
            raise NotImplementedError(f"unknown distribution [got: {dist}]")
    else:
        # we allow a fixed value specified directly:
        return functools.partial(np.random.uniform, self.as_evaled_expression(), self.as_evaled_expression(),)


def list_filenames(folder: str = ".", filters: list = []) -> list:
    """
    return the list of all filename and path in the provided folders.
    If filters [list] is provided, then only the files that contains each of the
    substrings in filter will be returned. Example to get all hosp file:
    ```
        gempyor.utils.list_filenames(folder="model_output/", filters=["hosp"])
    ```
        and be sure we only get parquet:
    ```
        gempyor.utils.list_filenames(folder="model_output/", filters=["hosp" , ".parquet"])
    ```
    """
    from pathlib import Path

    fn_list = []
    for f in Path(str(folder)).rglob(f"*"):
        if f.is_file():  # not a folder
            f = str(f)
            if not filters:
                fn_list.append(f)
            else:
                if all(c in f for c in filters):
                    fn_list.append(str(f))
                else:
                    pass
    return fn_list


def rolling_mean_pad(data, window):
    """
    Calculates rolling mean with centered window and pads the edges.

    Args:
        data: A NumPy array !!! shape must be (n_days, nsubpops).
        window: The window size for the rolling mean.

    Returns:
        A NumPy array with the padded rolling mean (n_days, nsubpops).
    """
    padding_size = (window - 1) // 2
    padded_data = np.pad(data, ((padding_size, padding_size), (0, 0)), mode="edge")

    # Allocate space for the result
    result = np.zeros_like(data)

    # Perform convolution along the days axis (axis 0) using a loop
    for i in range(data.shape[0]):
        # Extract the current day's data from the padded array
        window_data = padded_data[i : i + window, :]
        # Calculate the rolling mean for this day's data
        result[i, :] = np.mean(window_data, axis=0)

    return result


def print_disk_diagnosis():
    import os
    from os import path
    from shutil import disk_usage

    def bash(command):
        output = os.popen(command).read()
        return output

    print("START AWS DIAGNOSIS ================================")
    total_bytes, used_bytes, free_bytes = disk_usage(path.realpath("/"))
    print(
        f"shutil.disk_usage: {total_bytes/ 1000000} Mb total, {used_bytes / 1000000} Mb used, {free_bytes / 1000000} Mb free..."
    )
    print("------------")
    print(f"df -hT: {bash('df -hT')}")
    print("------------")
    print(f"df -i: {bash('df -i')}")
    print("------------")
    print(f"free -h: {bash('free -h')}")
    print("------------")
    print(f"lsblk: {bash('lsblk')}")
    print("END AWS DIAGNOSIS ================================")


def create_resume_out_filename(
    flepi_run_index: str, flepi_prefix: str, flepi_slot_index: str, flepi_block_index: str, filetype: str, liketype: str
) -> str:
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
    resume_run_index: str, flepi_prefix: str, flepi_slot_index: str, filetype: str, liketype: str
) -> str:
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


def get_filetype_for_resume(resume_discard_seeding: str, flepi_block_index: str) -> List[str]:
    """
    Retrieves a list of parquet file types that are relevant for resuming a process based on
    specific environment variable settings. This function dynamically determines the list
    based on the current operational context given by the environment.

    The function checks two environment variables:
    - `resume_discard_seeding`: Determines whether seeding-related file types should be included.
    - `flepi_block_index`: Determines a specific operational mode or block of the process.
    """
    if flepi_block_index == "1":
        if resume_discard_seeding == "true":
            return ["spar", "snpi", "hpar", "hnpi", "init"]
        else:
            return ["seed", "spar", "snpi", "hpar", "hnpi", "init"]
    else:
        return ["seed", "spar", "snpi", "hpar", "hnpi", "host", "llik", "init"]


def create_resume_file_names_map(
    resume_discard_seeding,
    flepi_block_index,
    resume_run_index,
    flepi_prefix,
    flepi_slot_index,
    flepi_run_index,
    last_job_output,
) -> Dict[str, str]:
    """
    Generates a mapping of input file names to output file names for a resume process based on
    parquet file types and environmental conditions. The function adjusts the file name mappings
    based on the operational block index and the location of the last job output.

    The mappings depend on:
    - Parquet file types appropriate for resuming a process, as determined by the environment.
    - Whether the files are for 'global' or 'chimeric' types, as these liketypes influence the
      file naming convention.
    - The operational block index ('FLEPI_BLOCK_INDEX'), which can alter the input file names for
      block index '1'.
    - The presence and value of 'LAST_JOB_OUTPUT' environment variable, which if set to an S3 path,
      adjusts the keys in the mapping to be prefixed with this path.

    Returns:
        Dict[str, str]: A dictionary where keys are input file paths and values are corresponding
                        output file paths. The paths may be modified by the 'LAST_JOB_OUTPUT' if it
                        is set and points to an S3 location.

    Raises:
        No explicit exceptions are raised within the function, but it relies heavily on external
        functions and environment variables which if improperly configured could lead to unexpected
        behavior.
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
            if os.environ.get("FLEPI_BLOCK_INDEX") == "1":
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


def download_file_from_s3(name_map: Dict[str, str]) -> None:
    """
    Downloads files from AWS S3 based on a mapping of S3 URIs to local file paths. The function
    checks if the directory for the first output file exists and creates it if necessary. It
    then iterates over each S3 URI in the provided mapping, downloads the file to the corresponding
    local path, and handles errors if the S3 URI format is incorrect or if the download fails.

    Parameters:
        name_map (Dict[str, str]): A dictionary where keys are S3 URIs (strings) and values
                                   are the local file paths (strings) where the files should
                                   be saved.

    Returns:
        None: This function does not return a value; its primary effect is the side effect of
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
                raise ValueError(f"Invalid S3 URI format {s3_uri}")
        except ClientError as e:
            print(f"An error occurred: {e}")
            print("Could not download file from s3")


def move_file_at_local(name_map: Dict[str, str]) -> None:
    """
    Moves files locally according to a given mapping.

    This function takes a dictionary where the keys are source file paths and 
    the values are destination file paths. It ensures that the destination 
    directories exist and then copies the files from the source paths to the 
    destination paths.

    Parameters:
    name_map (Dict[str, str]): A dictionary mapping source file paths to 
                               destination file paths.

    Returns:
    None
    """
    for src, dst in name_map.items():
        os.path.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
