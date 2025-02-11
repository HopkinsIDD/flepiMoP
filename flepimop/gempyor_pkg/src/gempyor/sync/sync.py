import os
from abc import ABC, abstractmethod
from typing import Literal, Annotated, Union, Dict
from pathlib import Path
from subprocess import run, CompletedProcess

import click

from ..shared_cli import (
    config_files_argument,
    parse_config_files,
    cli,
    mock_context,
)
from ..utils import config
from ..file_paths import create_file_name_for_push

sync_options = {
    "protocol": click.Option(
        ["-p", "--protocol"],
        type=click.STRING,
        help="sync protocol to use from configuration file",
    ),
    "source": click.Option(
        ["-s", "--source"],
        type=click.Path(),
        help="source directory to 'push' changes from",
    ),
    "target": click.Option(
        ["-t", "--target"],
        type=click.Path(),
        help="target directory to 'push' changes to",
    ),
    "filter": click.Option(
        ["-f", "--filter"],
        type=click.STRING,
        multiple=True,
        help="filter to apply to files; see `man rsync` for details",
    ),
}

@cli.command(
    name="sync",
    params=[config_files_argument] + list(sync_options.values()),
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.pass_context
def sync(ctx: click.Context = mock_context, **kwargs) -> int:
    """Sync flepimop files between local and remote locations."""
    config_files = kwargs.pop("config_files")
    if not config_files:
        ctx.fail("No configuration files provided." + "\n" + ctx.get_help())
    print("invoking bare sync -- assuming default value from config file")
