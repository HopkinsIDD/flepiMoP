import click
import yaml

from .shared_cli import (
    config_files_argument,
    config_file_options,
    parse_config_files,
    cli,
    mock_context,
)
from .utils import config

# register the commands from the other modules
from . import compartments, simulate
from .NPI import base

# Guidance for extending the CLI:
# - to add a new small command to the CLI, add a new function with the @cli.command() decorator here (e.g. patch below)
# - to add something with lots of module logic in it, define that in the module (e.g. .compartments, .simulate above)
# - ... and then import that module above to add it to the CLI


# add some basic commands to the CLI
@cli.command(params=[config_files_argument] + list(config_file_options.values()))
@click.pass_context
def patch(ctx: click.Context = mock_context, **kwargs) -> None:
    """Merge configuration files

    This command will merge multiple config files together by overriding the top level
    keys in config files. The order of the config files is important, as the last file
    has the highest priority and the first has the lowest.

    A brief example:

    \b
    ```bash
    $ cd $(mktemp -d)
    $ cat << EOF > config1.yml
    compartments:
        infection_stage: ['S', 'I', 'R']
    seir:
        parameters:
            beta:
                value: 1.2
    EOF
    $ cat << EOF > config2.yml
    name: 'more parameters'
    seir:
        parameters:
            beta:
                value: 3.4
            gamma:
                value: 5.6
    EOF
    $ flepimop patch config1.yml config2.yml
    ...: UserWarning: Configuration files contain overlapping keys: {'seir'}.
    warnings.warn(f"Configuration files contain overlapping keys: {intersect}.")
    compartments:
        infection_stage:
            - S
            - I
            - R
    config_src:
        - config1.yml
        - config2.yml
    first_sim_index: 1
    jobs: 14
    name: more parameters
    outcome_modifiers_scenarios: []
    seir:
        parameters:
            beta:
                value: 3.4
            gamma:
                value: 5.6
    seir_modifiers_scenarios: []
    stoch_traj_flag: false
    write_csv: false
    write_parquet: true
    $ flepimop patch config2.yml config1.yml
    ...: UserWarning: Configuration files contain overlapping keys: {'seir'}.
    warnings.warn(f"Configuration files contain overlapping keys: {intersect}.")
    compartments:
        infection_stage:
            - S
            - I
            - R
    config_src:
        - config2.yml
        - config1.yml
    first_sim_index: 1
    jobs: 14
    name: more parameters
    outcome_modifiers_scenarios: []
    seir:
        parameters:
            beta:
                value: 1.2
    parameters: null
    seir_modifiers_scenarios: []
    stoch_traj_flag: false
    write_csv: false
    write_parquet: true
    ```
    """
    parse_config_files(config, ctx, **kwargs)
    print(yaml.dump(yaml.safe_load(config.dump()), indent=4))


if __name__ == "__main__":
    cli()
