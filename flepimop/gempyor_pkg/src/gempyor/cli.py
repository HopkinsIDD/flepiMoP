from gempyor.shared_cli import argument_config_files, option_config_files, parse_config_files, cli
from gempyor.utils import config

# Guidance for extending the CLI:
# - to add a new small command to the CLI, can just add a new function with the @cli.command() decorator here (e.g. patch below)
# - to add something with lots of module logic in it, should define that in the module (e.g. compartments for a group command, or simulate for a single command)
# - ... and then import that module here to add it to the CLI

# add some basic commands to the CLI
@cli.command()
@argument_config_files
@option_config_files
def patch(**kwargs):
    """Merge configuration files"""
    parse_config_files(**kwargs)
    print(config.dump())

# register the commands from the other modules
from .compartments import compartments
from .simulate import simulate

if __name__ == "__main__":
    cli()
