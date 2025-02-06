import click
import yaml

from .shared_cli import (
    config_files_argument,
    config_file_options,
    parse_config_files,
    cli,
    mock_context,
)
from .utils import _dump_formatted_yaml, config

# register the commands from the other modules
from . import batch, compartments, simulate
from .NPI import base

# Guidance for extending the CLI:
# - to add a new small command to the CLI, add a new function with the @cli.command() decorator here (e.g. patch below)
# - to add something with lots of module logic in it, define that in the module (e.g. .compartments, .simulate above)
# - ... and then import that module above to add it to the CLI


# add some basic commands to the CLI
@cli.command(
    params=[config_files_argument] + list(config_file_options.values()),
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.pass_context
def patch(ctx: click.Context = mock_context, **kwargs) -> None:
    """
    Merge configuration files.

    This command will merge multiple config files together by overriding the top level
    keys in config files. The order of the config files is important, as the last file
    has the highest priority and the first has the lowest.

    A brief example of the command is shown below using the sample config files from the
    `examples/tutorials` directory. The command will merge the two files together and
    print the resulting configuration to the console.

    \b
    ```bash
        $ flepimop patch config_sample_2pop_modifiers_part.yml config_sample_2pop_outcomes_part.yml > config_sample_2pop_patched.yml
        $ cat config_sample_2pop_patched.yml
        write_csv: false
        stoch_traj_flag: false
        jobs: 14
        write_parquet: true
        first_sim_index: 1
        config_src: [config_sample_2pop_modifiers_part.yml, config_sample_2pop_outcomes_part.yml]
        seir_modifiers:
            scenarios: [Ro_lockdown, Ro_all]
            modifiers:
                Ro_lockdown:
                    method: SinglePeriodModifier
                    parameter: Ro
                    period_start_date: 2020-03-15
                    period_end_date: 2020-05-01
                    subpop: all
                    value: 0.4
                Ro_relax:
                    method: SinglePeriodModifier
                    parameter: Ro
                    period_start_date: 2020-05-01
                    period_end_date: 2020-08-31
                    subpop: all
                    value: 0.8
                Ro_all:
                    method: StackedModifier
                    modifiers: [Ro_lockdown, Ro_relax]
        outcome_modifiers:
            scenarios: [test_limits]
            modifiers:
                test_limits:
                    method: SinglePeriodModifier
                    parameter: incidCase::probability
                    subpop: all
                    period_start_date: 2020-02-01
                    period_end_date: 2020-06-01
                    value: 0.5
        outcomes:
            method: delayframe
            outcomes:
                incidCase:
                    source:
                        incidence:
                            infection_stage: I
                    probability:
                        value: 0.5
                    delay:
                        value: 5
                incidHosp:
                    source:
                        incidence:
                            infection_stage: I
                    probability:
                        value: 0.05
                    delay:
                        value: 7
                    duration:
                        value: 10
                        name: currHosp
                incidDeath:
                    source: incidHosp
                    probability:
                        value: 0.2
                    delay:
                        value: 14
    ```
    """
    parse_config_files(config, ctx, **kwargs)
    print(_dump_formatted_yaml(config))


if __name__ == "__main__":
    cli()
