from pathlib import Path

import yaml
import click

from ..shared_cli import (
    config_files_argument,
    parse_config_files,
    cli,
    mock_context,
)
from ..utils import config
from ..file_paths import create_file_name_for_push

from ._sync import sync_from_yaml

sync_options = {
    "protocol": click.Option(
        ["-p", "--protocol", "protocol"],
        type=click.STRING,
        help="sync protocol to use from configuration file",
    ),
    "source": click.Option(
        ["-s", "--source", "source_override"],
        type=click.Path(),
        help="source directory to 'push' changes from",
    ),
    "target": click.Option(
        ["-t", "--target", "target_override"],
        type=click.Path(),
        help="target directory to 'push' changes to",
    ),
    "filter": click.Option(
        ["-f", "--filter", "filter_override"],
        type=click.STRING,
        multiple=True,
        help="filter to apply to files; supports basic include/exclude filters per `man rsync`",
    ),
    "nofilter": click.Option(
        ["--no-filter", "nofilter"],
        is_flag=True,
        help="ignore all filters in config file",
    ),
    "reverse": click.Option(
        ["--reverse"],
        is_flag=True,
        help="reverse the source and target directories",
    ),
    "dryrun": click.Option(
        ["-n", "--dry-run", "dryrun"],
        is_flag=True,
        help="perform a dry run of the sync operation",
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

    config_files : list[Path] = kwargs.pop("config_files")
    if not config_files:
        ctx.fail("No configuration files provided." + "\n" + ctx.get_help())
    else:
        if kwargs['nofilter']:
            if kwargs['filter_override']:
                ctx.fail("Cannot use both `--no-filter` and `--filter` options together." + "\n" + ctx.get_help())
            else:
                kwargs['filter_override'] = []
        else:
            if not kwargs['filter_override']:
                kwargs['filter_override'] = None
        
        syncdef = sync_from_yaml(config_files)
        res = syncdef.execute(kwargs)
        return res.returncode