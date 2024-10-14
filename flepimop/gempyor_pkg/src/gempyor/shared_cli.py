import click
from functools import reduce
from gempyor.utils import config, as_list
import multiprocessing
import warnings

__all__ = []


@click.group()
def cli():
    """Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface"""
    pass


argument_config_files = click.argument(
    "config_files", nargs=-1, type=click.Path(exists=True)
)
""" `click` Argument decorator to handle configuration file(s) """

config_file_options = [
    click.option(
        "--write-csv/--no-write-csv",
        default=False,
        show_default=True,
        help="write csv output?",
    ),
    click.option(
        "--write-parquet/--no-write-parquet",
        default=True,
        show_default=True,
        help="write parquet output?",
    ),
    click.option(
        "-j",
        "--jobs",
        envvar="FLEPI_NJOBS",
        type=click.IntRange(min=1),
        default=multiprocessing.cpu_count(),
        show_default=True,
        help="the parallelization factor",
    ),
    click.option(
        "-n",
        "--nslots",
        envvar="FLEPI_NUM_SLOTS",
        type=click.IntRange(min=1),
        help="override the # of simulation runs in the config file",
    ),
    click.option(
        "--in-id",
        "in_run_id",
        envvar="FLEPI_RUN_INDEX",
        type=str,
        show_default=True,
        help="Unique identifier for the run",
    ),
    click.option(
        "--out-id",
        "out_run_id",
        envvar="FLEPI_RUN_INDEX",
        type=str,
        show_default=True,
        help="Unique identifier for the run",
    ),
    click.option(
        "--in-prefix",
        "in_prefix",
        envvar="FLEPI_PREFIX",
        type=str,
        default=None,
        show_default=True,
        help="unique identifier for the run",
    ),
    click.option(
        "-i",
        "--first_sim_index",
        envvar="FIRST_SIM_INDEX",
        type=click.IntRange(min=1),
        default=1,
        show_default=True,
        help="The index of the first simulation",
    ),
    click.option(
        "--stochastic/--non-stochastic",
        "stoch_traj_flag",
        envvar="FLEPI_STOCHASTIC_RUN",
        default=False,
        help="Run stochastic simulations?",
    ),
    click.option(
        "-s",
        "--seir_modifiers_scenarios",
        envvar="FLEPI_SEIR_SCENARIO",
        type=str,
        default=None,
        multiple=True,
        help="override/select the transmission scenario(s) to run",
    ),
    click.option(
        "-d",
        "--outcome_modifiers_scenarios",
        envvar="FLEPI_OUTCOME_SCENARIO",
        type=str,
        default=None,
        multiple=True,
        help="override/select the outcome scenario(s) to run",
    ),
    click.option(
        "-c",
        "--config",
        "config_filepath",
        envvar="CONFIG_PATH",
        type=click.Path(exists=True),
        required=False,  # deprecated = ["-c", "--config"],
        # preferred = "CONFIG_FILES...",
        help="Deprecated: configuration file for this simulation",
    ),
]
""" List of `click` options that will be applied by `option_config_files` """


def option_config_files(function: callable) -> callable:
    """`click` Option decorator to apply handlers for all options"""
    reduce(lambda f, option: option(f), config_file_options, function)
    return function


def parse_config_files(
    config_files: list[str],
    config_filepath: str,
    in_run_id: str,
    out_run_id: str,
    seir_modifiers_scenarios: list[str],
    outcome_modifiers_scenarios: list[str],
    in_prefix: str,
    nslots: int,
    jobs: int,
    write_csv: bool,
    write_parquet: bool,
    first_sim_index: int,
    stoch_traj_flag: bool,
) -> None:
    """Parse the configuration file(s) and override with command line arguments"""
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
        if config[option].exists():
            if (value := locals()[f"{option}_scenarios"]) is not None:
                config[option]["scenarios"] = value
        else:
            if (value := locals()[f"{option}_scenarios"]) is not None:
                config[option] = {"scenarios": as_list(value)}
            else:
                config[option] = {"scenarios": [None]}

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
