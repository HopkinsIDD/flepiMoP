from .shared_cli import (
    argument_config_files,
    option_config_files,
    parse_config_files,
    cli,
)
from .utils import config

# register the commands from the other modules
from .compartments import compartments
from .simulate import simulate

# Guidance for extending the CLI:
# - to add a new small command to the CLI, add a new function with the @cli.command() decorator here (e.g. patch below)
# - to add something with lots of module logic in it, define that in the module (e.g. .compartments, .simulate above)
# - ... and then import that module above to add it to the CLI


# add some basic commands to the CLI
@cli.command()
@argument_config_files
@option_config_files
def patch(**kwargs) -> None:
    """Merge configuration files"""
    parse_config_files(**kwargs)
    print(config.dump())


if __name__ == "__main__":
    cli()
