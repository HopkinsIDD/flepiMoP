import click
from gempyor.shared_cli import argument_config_files, option_config_files, parse_config_files, cli
from gempyor.utils import config
from gempyor.compartments import compartments

@cli.command()
@argument_config_files
@option_config_files
def patch(**kwargs):
    """Merge configuration files"""
    parse_config_files(**kwargs)
    print(config.dump())

if __name__ == "__main__":
    cli()
