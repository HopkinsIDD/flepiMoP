"""
An internal module to share common CLI elements. Defines the overall cli group,
supported options for config file overrides, and custom click decorators.
"""

import multiprocessing
import pathlib
import warnings
from typing import Callable, Any

import click

from .utils import config, as_list

__all__ = []

@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface"""
    pass

output_option = click.Option(
    ["-o", "--output-file"],
    type=click.Path(allow_dash=True),
    is_flag=False, flag_value="-",
    default="transition_graph.pdf", show_default=True,
    help="output file path",
)

# click decorator to handle configuration file(s) as arguments
# use as `@argument_config_files` before a cli command definition
config_files_argument = click.Argument(
    ["config_files"], nargs=-1, type=click.Path(exists=True)
)

# List of standard `click` options that override/update config settings
# n.b., the help for these options will be presented in the order defined here
config_file_options = {
    "config_filepath" : click.Option(
        ["-c", "--config", "config_filepath"],
        envvar="CONFIG_PATH",
        type=click.Path(exists=True),
        required=False,
        default=[],
        # deprecated = ["-c", "--config"],
        # preferred = "CONFIG_FILES...",
        multiple=True,
        help="Deprecated: configuration file(s) for this simulation",
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

def click_helpstring(
    params: click.Parameter | list[click.Parameter],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Dynamically append `click.Parameter`s to the docstring of a decorated function.

    Args:
        params: A parameter or a list of parameters whose corresponding argument
            values and help strings will be appended to the docstring of the
            decorated function.

    Returns:
        The original function with an updated docstring.

    Notes:
        Adapted from https://stackoverflow.com/a/78533451.

    Examples:
        >>> import click
        >>> opt = click.Option("-n", "--number", type=int, help="Your favorite number.")
        >>> @click_helpstring(opt)
        ... def test_cli(num: int) -> None:
        ...     print(f"Your favorite number is {num}!")
        ...
        >>> print(test_cli.__doc__)


        Command Line Interface arguments:

        n: Int
    """
    if not isinstance(params, list):
        params = [params]

    def decorator(func):
        # Generate the additional docstring with args from the specified functions
        additional_doc = "\n\tCommand Line Interface arguments:\n"
        for param in params:
            paraminfo = param.to_info_dict()
            additional_doc += f"\n\t\t{paraminfo['name']}: {paraminfo['type']['param_type']}\n"

        if func.__doc__ is None:
            func.__doc__ = ""
        func.__doc__ += additional_doc

        return func

    return decorator

# TODO: create a custom command decorator cls ala: https://click.palletsprojects.com/en/8.1.x/advanced/#command-aliases
# to also apply the `@click_helpstring` decorator to the command. Possibly to also default the params argument, assuming
# enough commands have consistent option set?

# TODO: have parse_config_files check with the click.Parameter validators:
# https://stackoverflow.com/questions/59096020/how-to-unit-test-function-that-requires-an-active-click-context-in-python

mock_context = click.Context(click.Command('mock'), info_name="Mock context for non-click use of parse_config_files")

@click_helpstring([config_files_argument] + list(config_file_options.values()))
def parse_config_files(ctx = mock_context, **kwargs) -> None:
    """
    Parse configuration file(s) and override with command line arguments

    Args:
        **kwargs: see auto generated CLI items below. Unmatched keys will be ignored + a warning will be issued

    Returns: None (side effect: updates the global configuration object)
    """
    parsed_args = {config_files_argument.name}.union({option.name for option in config_file_options.values()})

    # warn re unrecognized arguments
    if parsed_args.difference(kwargs.keys()):
        warnings.warn(f"Unused arguments: {parsed_args.difference(kwargs.keys())}")

    # initialize the config, including handling missing / double-specified config files
    config_args = {k for k in parsed_args if k.startswith("config")}
    found_configs = [k for k in config_args if kwargs.get(k)]
    config_src = []
    if len(found_configs) != 1:
        if not found_configs:
            raise ValueError(f"No config files provided.")
        else:
            error_dict = {k: kwargs[k] for k in found_configs}
            raise ValueError(f"Exactly one config file source option must be provided; got {error_dict}.")
    else:
        config_key = found_configs[0]
        config_validator = config_file_options[config_key] if config_key in config_file_options else config_files_argument
        config_src = config_validator.type_cast_value(ctx, kwargs[config_key])
        config.clear()
        for config_file in reversed(config_src):
            config.set_file(config_file)
        config["config_src"] = config_src

    # deal with the scenario overrides
    scen_args = {k for k in parsed_args if k.endswith("scenarios") and kwargs.get(k)}
    for option in scen_args:
        key = option.replace("_scenarios", "")
        value = config_file_options[option].type_cast_value(ctx, kwargs[option])
        if config[key].exists():
            config[key]["scenarios"] = as_list(value)
        else:
            raise ValueError(f"Specified {option} when no {key} in configuration file(s): {config_src}")

    # update the config with the remaining options
    other_args = parsed_args - config_args - scen_args
    for option in other_args:
        if (value := kwargs.get(option)) is not None:
            config[option] = config_file_options[option].type_cast_value(ctx, value)
