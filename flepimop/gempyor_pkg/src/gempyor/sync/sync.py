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

from ._sync import sync_from_yaml

sync_options = {
    "protocol": click.Option(
        ["-p", "--protocol", "protocol"],
        type=click.STRING,
        help="sync protocol to use from CONFIG_FILES",
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
    "sfilter": click.Option(
        ["-e", "--fsuffix", "filter_suffix"],
        type=click.STRING,
        multiple=True,
        help="Add a filter to the end of the filter list in CONFIG_FILES (same as `-f` if that list is empty)",
    ),
    "pfilter": click.Option(
        ["-a", "--fprefix", "filter_prefix"],
        type=click.STRING,
        multiple=True,
        help="Add a filter to the beginning of the filter list in CONFIG_FILES (same as `-f` if that list is empty)",
    ),
    "filter": click.Option(
        ["-f", "--filter", "filter_override"],
        type=click.STRING,
        multiple=True,
        help="replace the filter list in CONFIG_FILES; can be specified multiple times",
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
    "mkpath": click.Option(
        ["--mkpath"],
        is_flag=True,
        help="Before syncing, manually ensure destination directory exists.",
    ),
    "dryrun": click.Option(
        ["-n", "--dry-run", "dryrun"],
        is_flag=True,
        help="perform a dry run of the sync operation",
    ),
    "verbosity": click.Option(
        ["-v", "--verbose", "verbosity"],
        count=True,
        help="The verbosity level to use for this command.",
    ),
}


@cli.command(
    name="sync",
    params=[config_files_argument] + list(sync_options.values()),
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.pass_context
def sync(ctx: click.Context = mock_context, **kwargs) -> int:
    """
    Sync flepimop files between local and remote locations. For the filter options,
    see `man rsync` for more information - sync supports basic include / exclude filters,
    and follows the rsync precendence rules: earlier filters have higher precedence.

    All of the filter options (-a, -e, -f) can be specified multiple times to add multiple filters.
    For the prefix and suffix filters, they are first assembled into a list in the order specified
    and then added to the beginning or end of the filter list in the config file. So e.g. `-a "+ foo" -a "- bar"`
    adds [`+ foo`, `- bar`] to the beginning of the filter list, meaning the include filter `+ a` has higher precedence
    than the exclude filter `- bar`.

    """

    config_files: list[Path] = kwargs.pop("config_files")
    if not config_files:
        ctx.fail("No configuration files provided." + "\n" + ctx.get_help())
    else:
        if kwargs["nofilter"]:
            if kwargs["filter_override"] or kwargs["filter_prefix"] or kwargs["filter_suffix"]:
                ctx.fail(
                    "Cannot use both `--no-filter` and `-f|a|e` options together."
                    + "\n"
                    + ctx.get_help()
                )
            else:
                kwargs["filter_override"] = []
        else:
            if not kwargs["filter_override"]:
                kwargs["filter_override"] = None

        try:
            verbosity = kwargs.pop("verbosity")
            return sync_from_yaml(config_files, kwargs, verbosity).returncode
        except ValidationError as e:
            ctx.fail(f"Configuration error in `sync`: {e}")
