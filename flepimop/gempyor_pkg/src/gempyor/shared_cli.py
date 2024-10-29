import click
from functools import reduce
from gempyor.utils import config, as_list
import multiprocessing
import warnings

@click.group()
def cli():
    """Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface"""
    pass

argument_config_files = click.argument("config_files", nargs = -1, type = click.Path(exists = True), required = True)

config_file_options = [
    click.option(
        "--write-csv/--no-write-csv",
        default=False, show_default=True,
        help="write csv output?",
    ),
    click.option(
        "--write-parquet/--no-write-parquet",
        default=True, show_default=True,
        help="write parquet output?",
    ),
    click.option(
        "-j", "--jobs", envvar = "FLEPI_NJOBS",
        type = click.IntRange(min = 1), default = multiprocessing.cpu_count(), show_default=True,
        help = "the parallelization factor",
    ),
    click.option(
        "-n", "--nslots", envvar = "FLEPI_NUM_SLOTS",
        type = click.IntRange(min = 1),
        help = "override the # of simulation runs in the config file",
    ),
    click.option(
        "--in-id", "in_run_id", envvar="FLEPI_RUN_INDEX",
        type=str, show_default=True,
        help="Unique identifier for the run",
    ),
    click.option(
        "--out-id", "out_run_id", envvar="FLEPI_RUN_INDEX",
        type = str, show_default = True,
        help = "Unique identifier for the run",
    ),
    click.option(
        "--in-prefix", "in_prefix", envvar="FLEPI_PREFIX",
        type=str, default=None, show_default=True,
        help="unique identifier for the run",
    ),
    click.option(
        "-i", "--first_sim_index", envvar = "FIRST_SIM_INDEX",
        type = click.IntRange(min = 1),
        default = 1, show_default = True,
        help = "The index of the first simulation",
    ),
    click.option(
        "--stochastic/--non-stochastic", "stoch_traj_flag", envvar = "FLEPI_STOCHASTIC_RUN",
        default = False,
        help = "Run stochastic simulations?",
    ),
    click.option(
        "-s", "--seir_modifiers_scenarios", envvar = "FLEPI_SEIR_SCENARIO",
        type = str, default = None, multiple = True,
        help = "override/select the transmission scenario(s) to run",
    ),
    click.option(
        "-d", "--outcome_modifiers_scenarios",
        envvar = "FLEPI_OUTCOME_SCENARIO",
        type = str, default = None, multiple = True,
        help = "override/select the outcome scenario(s) to run"
    ),
    click.option(
        "-c", "--config", "config_filepath", envvar = "CONFIG_PATH",
        type = click.Path(exists = True),
        required = False, # deprecated = ["-c", "--config"],
        # preferred = "CONFIG_FILES...",
        help = "Deprecated: configuration file for this simulation"
    )
]

def option_config_files(function):
    reduce(lambda f, option: option(f), config_file_options, function)
    return function

def parse_config_files(
    config_files,
    config_filepath,
    in_run_id,
    out_run_id,
    seir_modifiers_scenarios,
    outcome_modifiers_scenarios,
    in_prefix,
    nslots,
    jobs,
    write_csv,
    write_parquet,
    first_sim_index,
    stoch_traj_flag
) -> None:
    # parse the config file(s)
    config.clear()
    if config_filepath:
        if not len(config_files):
            config_files = [config_filepath]
        else:
            warnings.warn("Found CONFIG_FILES... ignoring -(-c)onfig option / CONFIG_FILE environment variable.", DeprecationWarning)

    for config_file in reversed(config_files):
        config.set_file(config_file)

    # override the config file with any command line arguments
    if config["seir_modifiers"].exists():
        if seir_modifiers_scenarios:
            config["seir_modifiers"]["scenarios"] = seir_modifiers_scenarios
    else:
        config["seir_modifiers"] = {"scenarios": as_list(seir_modifiers_scenarios)}
    

    if config["outcome_modifiers"].exists():
        if outcome_modifiers_scenarios:
            config["outcome_modifiers"]["scenarios"] = outcome_modifiers_scenarios
    else:
        config["outcome_modifiers"] = {"scenarios": as_list(outcome_modifiers_scenarios)}
    
    if nslots:
        config["nslots"] = nslots
    
    if in_run_id:
        config["in_run_id"] = in_run_id
    
    if out_run_id:
        config["out_run_id"] = out_run_id
    
    if in_prefix:
        config["in_prefix"] = in_prefix
    
    if jobs:
        config["jobs"] = jobs
    
    if write_csv:
        config["write_csv"] = write_csv
    
    if write_parquet:    
        config["write_parquet"] = write_parquet
    
    if first_sim_index:
        config["first_sim_index"] = first_sim_index
    
    if stoch_traj_flag:
        config["stoch_traj_flag"] = stoch_traj_flag
