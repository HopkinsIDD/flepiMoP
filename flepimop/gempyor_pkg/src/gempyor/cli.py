import click
import sys
# from .cli_shared import argument_config_files, parse_config_files, config
# TODO ...is this importing a global config object? do not like
from gempyor.utils import config
from gempyor.compartments import Compartments

argument_config_files = click.argument(
    "config_files",
    nargs = -1,
    type = click.Path(exists = True),
    required = True
)

def parse_config_files(config_files) -> None:
    config.clear()
    for config_file in reversed(config_files):
        config.set_file(config_file)

@click.group()
def cli():
    """Flexible Epidemic Modeling Platform (FlepiMoP) Command Line Interface"""
    pass

@cli.command()
@argument_config_files
def patch(config_files):
    """Merge configuration files"""
    parse_config_files(config_files)
    print(config.dump())


@click.group()
def compartments():
    """Commands for working with FlepiMoP compartments"""
    pass

# TODO: CLI arguments
@compartments.command()
@argument_config_files
def plot(config_files):
    """Plot compartments"""
    parse_config_files(config_files)
    assert config["compartments"].exists()
    assert config["seir"].exists()
    comp = Compartments(seir_config=config["seir"], compartments_config=config["compartments"])

    # TODO: this should be a command like build compartments.
    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = comp.get_transition_array()

    comp.plot(output_file="transition_graph", source_filters=[], destination_filters=[])

    print("wrote file transition_graph")


@compartments.command()
@argument_config_files
def export(config_files):
    """Export compartments"""
    parse_config_files(config_files)
    assert config["compartments"].exists()
    assert config["seir"].exists()
    comp = Compartments(seir_config=config["seir"], compartments_config=config["compartments"])
    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = comp.get_transition_array()
    comp.toFile("compartments_file.csv", "transitions_file.csv", write_parquet=False)
    print("wrote files 'compartments_file.csv', 'transitions_file.csv' ")

cli.add_command(compartments)

if __name__ == "__main__":
    cli()
