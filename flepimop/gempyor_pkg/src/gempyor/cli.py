import click
from .compartments import compartments
from gempyor.utils import config


@click.group()
@click.option(
    "-c",
    "--config",
    "config_filepath",
    envvar=["CONFIG_PATH"],
    type=click.Path(exists=True),
    help="configuration file for this simulation",
)
def cli(config_filepath):
    print(config_filepath)
    config.clear()
    config.read(user=False)
    config.set_file(config_filepath)


cli.add_command(compartments)


if __name__ == "__main__":
    cli()
