from click import pass_context, Context

from .shared_cli import (
    config_files_argument,
    config_file_options,
    parse_config_files,
    cli,
    mock_context,
)
from .utils import config

# register the commands from the other modules
from . import compartments, simulate, batch


# Guidance for extending the CLI:
# - to add a new small command to the CLI, add a new function with the @cli.command() decorator here (e.g. patch below)
# - to add something with lots of module logic in it, define that in the module (e.g. .compartments, .simulate above)
# - ... and then import that module above to add it to the CLI


# add some basic commands to the CLI
@cli.command(params=[config_files_argument] + list(config_file_options.values()))
@pass_context
def patch(ctx: Context = mock_context, **kwargs) -> None:
    """Merge configuration files"""
    parse_config_files(config, ctx, **kwargs)
    print(config.dump())


if __name__ == "__main__":
    cli()
