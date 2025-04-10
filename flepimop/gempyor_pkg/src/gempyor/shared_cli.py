"""
An internal module to share common CLI elements. Defines the overall cli group,
supported options for config file overrides, and custom click decorators.
"""

__all__ = []


import multiprocessing
import pathlib

from typing import Any, Callable
import warnings

import click
import confuse

from .logging import get_script_logger
from .utils import config, as_list


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface"""
    pass


# click decorator to handle configuration file(s) as arguments
# use as `@argument_config_files` before a cli command definition
config_files_argument = click.Argument(
    ["config_files"], nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path)
)

# List of standard `click` options that override/update config settings
# n.b., the help for these options will be presented in the order defined here
config_file_options = {
    "config_filepath": click.Option(
        ["-c", "--config", "config_filepath"],
        envvar="CONFIG_PATH",
        type=click.Path(exists=True, path_type=pathlib.Path),
        required=False,
        # deprecated = ["-c", "--config"],
        # preferred = "CONFIG_FILES...",
        multiple=True,
        help="Deprecated: configuration file(s) for this simulation",
    ),
    "populations": click.Option(
        ["-p", "--populations", "populations"],
        type=click.STRING,
        required=False,
        multiple=True,
        help="Population(s) to run use in simulation.",
    ),
    "seir_modifiers_scenarios": click.Option(
        ["-s", "--seir_modifiers_scenarios"],
        envvar="FLEPI_SEIR_SCENARIO",
        type=click.STRING,
        default=[],
        multiple=True,
        help="override/select the transmission scenario(s) to run",
    ),
    "outcome_modifiers_scenarios": click.Option(
        ["-d", "--outcome_modifiers_scenarios"],
        envvar="FLEPI_OUTCOME_SCENARIO",
        type=click.STRING,
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
        type=click.STRING,
        show_default=True,
        help="Unique identifier for the run",
    ),
    "out_run_id": click.Option(
        ["--out-id", "out_run_id"],
        envvar="FLEPI_RUN_INDEX",
        type=click.STRING,
        show_default=True,
        help="Unique identifier for the run",
    ),
    "in_prefix": click.Option(
        ["--in-prefix"],
        envvar="FLEPI_PREFIX",
        type=click.STRING,
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
    "method": click.Option(
        ["-m", "--method"],
        type=click.STRING,
        help="If provided, overrides seir::integration::method",
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
verbosity_options = {
    "verbosity": click.Option(
        param_decls=["-v", "--verbose", "verbosity"],
        count=True,
        help="The verbosity level to use for this command.",
    ),
    "dry_run": click.Option(
        param_decls=["--dry-run", "dry_run"],
        type=bool,
        default=False,
        is_flag=True,
        help="Should this command be run using dry run?",
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

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Generate the additional docstring with args from the specified functions
        additional_doc = "\n\tCommand Line Interface arguments:\n"
        for param in params:
            paraminfo = param.to_info_dict()
            additional_doc += f"\n\t{paraminfo['name']}: {paraminfo['type']['param_type']}"

        if func.__doc__ is None:
            func.__doc__ = ""

        func.__doc__ += additional_doc.lstrip()

        return func

    return decorator


# TODO: create a custom command decorator cls ala: https://click.palletsprojects.com/en/8.1.x/advanced/#command-aliases
# to also apply the `@click_helpstring` decorator to the command. Possibly to also default the params argument, assuming
# enough commands have consistent option set?

mock_context = click.Context(
    click.Command("mock"),
    info_name="Mock context for non-click use of parse_config_files",
)


@click_helpstring([config_files_argument] + list(config_file_options.values()))
def parse_config_files(
    cfg: confuse.Configuration = config, ctx: click.Context = mock_context, **kwargs
) -> confuse.Configuration:
    """
    Parse configuration file(s) and override with command line arguments

    Args:
        cfg: the configuration object to update; defaults to global `utils.config`
        ctx: the click context (used for type casting); defaults to a mock context for non-click use; pass actual context if using in a click command
        **kwargs: see auto generated CLI items below. Unmatched keys will be ignored + a warning will be issued

    Returns: returns the object passed via `cfg`; n.b. this object is also side-effected
    """

    def _parse_option(param: click.Parameter, value: Any) -> Any:
        """internal parser to autobox values"""
        if (param.multiple or param.nargs == -1) and not isinstance(value, (list, tuple)):
            value = [value]
        return param.type_cast_value(ctx, value)

    parsed_args = {config_files_argument.name}.union(
        {option.name for option in config_file_options.values()}
    )

    # warn re unrecognized arguments
    if unknownargs := [
        k for k in parsed_args.difference(kwargs.keys()) if kwargs.get(k) is not None
    ]:
        warnings.warn(f"Unused arguments: {unknownargs}")

    # initialize the config, including handling missing / double-specified config files
    config_args = {k for k in parsed_args if k.startswith("config")}
    found_configs = [k for k in config_args if kwargs.get(k)]
    config_src = []
    if len(found_configs) != 1:
        if not found_configs:
            click.echo("No configuration provided! See help for required usage:\n")
            click.echo(ctx.get_help())
            ctx.exit()
        else:
            error_dict = {k: kwargs[k] for k in found_configs}
            raise ValueError(
                f"Exactly one config file source option must be provided; got {error_dict}."
            )
    else:
        config_key = found_configs[0]
        config_validator = (
            config_file_options[config_key]
            if config_key in config_file_options
            else config_files_argument
        )
        config_src = _parse_option(config_validator, kwargs[config_key])
        cfg.clear()
        cfg_data = {}
        for config_file in config_src:
            tmp = confuse.Configuration("tmp")
            tmp.set_file(config_file)
            if intersect := set(tmp.keys()) & set(cfg_data.keys()):
                intersect = ", ".join(sorted(list(intersect)))
                raise ValueError(
                    "Configuration files contain overlapping keys, "
                    f"{intersect}, introduced by {config_file}."
                )
            for k in tmp.keys():
                cfg_data[k] = tmp[k].get()
        cfg.set(cfg_data)
        cfg["config_src"] = [str(k) for k in config_src]

    # deal with the scenario overrides
    scen_args = {k for k in parsed_args if k.endswith("_scenarios")}
    for option in {s for s in scen_args if kwargs.get(s)}:
        key = option.replace("_scenarios", "")
        value = _parse_option(config_file_options[option], kwargs[option])
        if cfg[key].exists():
            cfg[key]["scenarios"] = (
                list(value) if isinstance(value, tuple) else as_list(value)
            )
        else:
            raise ValueError(
                f"Specified {option} when no {key} in configuration file(s): {config_src}"
            )

    # update the config with the remaining options
    other_args = parsed_args - config_args - scen_args

    if method := kwargs.pop("method", None):
        cfg["seir"]["integration"]["method"].set(
            _parse_option(config_file_options["method"], method)
        )

    if (populations := kwargs.pop("populations", None)) is not None:
        if populations:
            cfg["subpop_setup"]["selected"].set(
                list(_parse_option(config_file_options["populations"], populations))
            )

    for option in other_args:
        if (value := kwargs.get(option)) is not None:
            # auto box the value if the option expects a multiple
            cfg[option] = _parse_option(config_file_options[option], value)

    return cfg


def log_cli_inputs(kwargs: dict[str, Any], verbosity: int | None = None) -> None:
    """
    Log CLI inputs for user debugging.

    This function only logs debug messages so the verbosity has to be set quite high
    to see the output of this function.

    Args:
        kwargs: The CLI arguments given as a dictionary of key word arguments.
        verbosity: The verbosity level of the CLI tool being used or `None` to infer
            from the given `kwargs`.

    Examples:
        >>> from gempyor.shared_cli import log_cli_inputs
        >>> log_cli_inputs({"abc": 123, "def": True}, 3)
        2024-11-05 09:27:58,884:DEBUG:gempyor.shared_cli> CLI was given 2 arguments:
        2024-11-05 09:27:58,885:DEBUG:gempyor.shared_cli> abc = 123.
        2024-11-05 09:27:58,885:DEBUG:gempyor.shared_cli> def = True.
        >>> log_cli_inputs({"abc": 123, "def": True}, 2)
        >>> from pathlib import Path
        >>> kwargs = {
        ...     "input_file": Path("config.in"),
        ...     "method": "stochastic",
        ...     "cluster": "longleaf",
        ...     "verbosity": 3,
        ... }
        >>> log_cli_inputs(kwargs)
        2024-11-05 09:29:21,666:DEBUG:gempyor.shared_cli> CLI was given 4 arguments:
        2024-11-05 09:29:21,667:DEBUG:gempyor.shared_cli> input_file = /Users/twillard/Desktop/GitHub/HopkinsIDD/flepiMoP/flepimop/gempyor_pkg/config.in.
        2024-11-05 09:29:21,667:DEBUG:gempyor.shared_cli> method = stochastic.
        2024-11-05 09:29:21,668:DEBUG:gempyor.shared_cli> cluster    = longleaf.
        2024-11-05 09:29:21,668:DEBUG:gempyor.shared_cli> verbosity  = 3.
    """
    verbosity = kwargs.get("verbosity") if verbosity is None else verbosity
    if verbosity is None:
        return
    logger = get_script_logger(__name__, verbosity)
    longest_key = -1
    total_keys = 0
    for k, _ in kwargs.items():
        longest_key = len(k) if len(k) > longest_key else longest_key
        total_keys += 1
    logger.debug("CLI was given %u arguments:", total_keys)
    for k, v in kwargs.items():
        v = v.absolute() if isinstance(v, pathlib.Path) else v
        logger.debug("%s = %s", k.ljust(longest_key, " "), v)
