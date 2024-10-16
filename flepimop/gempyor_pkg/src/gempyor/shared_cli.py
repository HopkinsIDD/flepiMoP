"""
An internal module to share common CLI elements. Defines the overall cli group,
supported options for config file overrides, and custom click decorators.
"""

import multiprocessing
import pathlib
import warnings
from typing import List, Union


import click

from .utils import config, as_list

__all__ = []

@click.group()
def cli():
    """Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface"""
    pass

# click decorator to handle configuration file(s) as arguments
# use as `@argument_config_files` before a cli command definition
config_files_argument = click.Argument(
    ["config_files"], nargs=-1, type=click.Path(exists=True)
)

# List of `click` options that will be applied by `option_config_files`
# n.b., the help for these options will be presented in the order defined here
config_file_options = {
    "config_filepath" : click.Option(
        ["-c", "--config", "config_filepath"],
        envvar="CONFIG_PATH",
        type=click.Path(exists=True),
        required=False,  # deprecated = ["-c", "--config"],
        # preferred = "CONFIG_FILES...",
        help="Deprecated: configuration file for this simulation",
    ),
    "seir_modifiers_scenarios": click.Option(
        ["-s", "--seir_modifiers_scenarios"],
        envvar="FLEPI_SEIR_SCENARIO",
        type=str,
        default=[],
        multiple=True,
        help="override/select the transmission scenario(s) to run",
    ),
    "outcome_modifiers_scenarios": click.Option(
        ["-d", "--outcome_modifiers_scenarios"],
        envvar="FLEPI_OUTCOME_SCENARIO",
        type=str,
        default=[],
        multiple=True,
        help="override/select the outcome scenario(s) to run",
    ),
    "jobs": click.Option(
        ["-j", "--jobs"],
        envvar="FLEPI_NJOBS",
        type=click.IntRange(min=1),
        default=multiprocessing.cpu_count(),
        show_default=True,
        help="the parallelization factor",
    ),
    "nslots": click.Option(
        ["-n", "--nslots"],
        envvar="FLEPI_NUM_SLOTS",
        type=click.IntRange(min=1),
        help="override the # of simulation runs in the config file",
    ),
    "in_run_id": click.Option(
        ["--in-id", "in_run_id"],
        envvar="FLEPI_RUN_INDEX",
        type=str,
        show_default=True,
        help="Unique identifier for the run",
    ),
    "out_run_id": click.Option(
        ["--out-id", "out_run_id"],
        envvar="FLEPI_RUN_INDEX",
        type=str,
        show_default=True,
        help="Unique identifier for the run",
    ),
    "in_prefix": click.Option(
        ["--in-prefix"],
        envvar="FLEPI_PREFIX",
        type=str,
        default=None,
        show_default=True,
        help="unique identifier for the run",
    ),
    "first_sim_index": click.Option(
        ["-i", "--first_sim_index"],
        envvar="FIRST_SIM_INDEX",
        type=click.IntRange(min=1),
        default=1,
        show_default=True,
        help="The index of the first simulation",
    ),
    "stoch_traj_flag": click.Option(
        ["--stochastic/--non-stochastic", "stoch_traj_flag"],
        envvar="FLEPI_STOCHASTIC_RUN",
        default=False,
        help="Run stochastic simulations?",
    ),
    "write_csv": click.Option(
        ["--write-csv/--no-write-csv"],
        default=False,
        show_default=True,
        help="write csv output?",
    ),
    "write_parquet": click.Option(
        ["--write-parquet/--no-write-parquet"],
        default=True,
        show_default=True,
        help="write parquet output?",
    ),
}


# adapted from https://stackoverflow.com/a/78533451
def click_helpstring(params: Union[click.Parameter, List[click.Parameter]]):
    """
    A decorator that dynamically appends `click.Parameter`s to the docstring of the decorated function.

    Args:
        params (Union[click.Parameter, List[click.Parameter]]): A parameter or a list of parameters whose corresponding argument values and help strings will be appended to the docstring of the decorated function.

    Returns:
        Callable: The original function with an updated docstring.
    """
    if not isinstance(params, list):
        params = [params]

    def decorator(func):
        # Generate the additional docstring with args from the specified functions
        additional_doc = "\n\nCommand Line Interface arguments:\n"
        for param in params:
            paraminfo = param.to_info_dict()
            additional_doc += f"\n{paraminfo['name']}: {paraminfo['type']['param_type']}\n"

        if func.__doc__ is None:
            func.__doc__ = ""
        func.__doc__ += additional_doc

        return func

    return decorator

# TODO: create a custom command decorator cls ala: https://click.palletsprojects.com/en/8.1.x/advanced/#command-aliases
# to also apply the `@click_helpstring` decorator to the command. Possibly to also default the params argument, assuming
# enough commands have consistent option set?

@click_helpstring([config_files_argument] + list(config_file_options.values()))
def parse_config_files(
    config_files: list[pathlib.Path],
    config_filepath: pathlib.Path = None,
    seir_modifiers_scenarios: list[str] = [],
    outcome_modifiers_scenarios: list[str] = [],
    in_run_id: str = None,
    out_run_id: str = None,
    in_prefix: str = None,
    nslots: int = None,
    jobs: int = None,
    write_csv: bool = False,
    write_parquet: bool = True,
    first_sim_index: int = 1,
    stoch_traj_flag: bool = False,
) -> None:
    """
    Parse configuration file(s) and override with command line arguments

    Args: (see auto generated CLI items below)

    Returns: None (side effect: updates the global configuration object)
    """
    config.clear()
    if config_filepath:
        if not len(config_files):
            config_files = [config_filepath]
        else:
            warnings.warn(
                "Found CONFIG_FILES... ignoring -(-c)onfig option / CONFIG_FILE environment variable.",
                DeprecationWarning,
            )

    if not len(config_files):
        raise ValueError("No configuration file(s) provided")

    for config_file in reversed(config_files):
        config.set_file(config_file)

    for option in ("seir_modifiers", "outcome_modifiers"):
        value = locals()[f"{option}_scenarios"]
        if value:
            if config[option].exists():
                config[option]["scenarios"] = as_list(value)
            else:
                raise ValueError(f"Specified {option}_scenarios when no {option} in configuration file(s): {value}")

    for option in (
        "nslots",
        "in_run_id",
        "out_run_id",
        "in_prefix",
        "jobs",
        "write_csv",
        "write_parquet",
        "first_sim_index",
        "stoch_traj_flag",
    ):
        if (value := locals()[option]) is not None:
            config[option] = value
