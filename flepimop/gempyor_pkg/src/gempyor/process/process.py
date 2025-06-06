from pathlib import Path

import yaml
import click
from pydantic import ValidationError

from ..shared_cli import (
    config_files_argument,
    parse_config_files,
    cli,
    mock_context,
)
from ..utils import config
from ..file_paths import create_file_name_for_push

from ._process import process_from_yaml

__all__ = ["process"]

process_options = {
    "process": click.Option(
        ["-p", "--process", "process"],
        type=click.STRING,
        help="process target to use from CONFIG_FILES",
    ),
    "arguments": click.Option(
        ["-a", "--argument", "arguments"],
        type=click.STRING, multiple=True,
        help="any arguments to pass to the processing target",
    ),
    "dryrun": click.Option(
        ["-n", "--dry-run", "dryrun"],
        is_flag=True,
        help="perform a dry run of the operation",
    ),
    "verbosity": click.Option(
        ["-v", "--verbose", "verbosity"],
        count=True,
        help="The verbosity level to use for this command.",
    ),
}

@cli.command(
    name="process",
    params=[config_files_argument] + list(process_options.values()),
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.pass_context
def process(ctx: click.Context = mock_context, **kwargs) -> int:
    """
    Execute pre- or post-processing steps according to a `process` configuration
    """

    config_files: list[Path] = kwargs.pop("config_files")
    if not config_files:
        ctx.fail("No configuration files provided." + "\n" + ctx.get_help())
    else:
        try:
            verbosity = kwargs.pop("verbosity")
            return process_from_yaml(config_files, kwargs, verbosity).returncode
        except ValidationError as e:
            ctx.fail(f"Configuration error in `sync`: {e}")
