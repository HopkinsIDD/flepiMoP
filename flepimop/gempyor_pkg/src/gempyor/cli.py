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
@cli.command(
    params=[config_files_argument]
    + list(config_file_options.values())
    + [
        click.Option(
            ["--indent"],
            type=click.IntRange(min=1),
            required=False,
            default=2,
            help="Indentation level for the output YAML.",
        )
    ],
)
@click.pass_context
def patch(ctx: click.Context = mock_context, **kwargs) -> None:
    """Merge configuration files"""
    parse_config_files(config, ctx, **kwargs)
    print(yaml.dump(yaml.safe_load(config.dump()), indent=kwargs["indent"]))


if __name__ == "__main__":
    cli()
