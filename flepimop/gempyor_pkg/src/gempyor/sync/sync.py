"""The entry point for the `flepimop sync` command."""

__all__ = ("sync",)


from pathlib import Path
from typing import Final

import click
from pydantic import ValidationError

from ..logging import get_script_logger
from ..shared_cli import (
    cli,
    config_files_argument,
    log_cli_inputs,
    mock_context,
    verbosity_options,
)
from ._sync import sync_from_yaml


_SYNC_OPTIONS: Final = {
    "protocol": click.Option(
        ["-p", "--protocol", "protocol"],
        type=click.STRING,
        help="sync protocol to use from CONFIG_FILES",
    ),
    "source": click.Option(
        ["-s", "--source", "source_override"],
        type=click.STRING,
        help="source directory to 'push' changes from",
    ),
    "target": click.Option(
        ["-t", "--target", "target_override"],
        type=click.STRING,
        help="target directory to 'push' changes to",
    ),
    "sourceappend": click.Option(
        ["--source-append", "source_append"],
        type=click.STRING,
        default=None,
        show_default=True,
        help="Append to the source instead of replacing it.",
    ),
    "targetappend": click.Option(
        ["--target-append", "target_append"],
        type=click.STRING,
        default=None,
        show_default=True,
        help="Append to the target instead of replacing it.",
    ),
    "sfilter": click.Option(
        ["-e", "--fsuffix", "filter_suffix"],
        type=click.STRING,
        multiple=True,
        help=(
            "Add a filter to the end of the filter list in "
            "CONFIG_FILES (same as `-f` if that list is empty)"
        ),
    ),
    "pfilter": click.Option(
        ["-a", "--fprefix", "filter_prefix"],
        type=click.STRING,
        multiple=True,
        help=(
            "Add a filter to the beginning of the filter list in "
            "CONFIG_FILES (same as `-f` if that list is empty)"
        ),
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
} | verbosity_options


@cli.command(
    name="sync",
    params=[config_files_argument] + list(_SYNC_OPTIONS.values()),
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.pass_context
def sync(  # pylint: disable=inconsistent-return-statements
    ctx: click.Context = mock_context, **kwargs
) -> int:
    """
    Sync flepimop files between local and remote locations. For the filter options,
    see `man rsync` for more information - sync supports basic include / exclude
    filters and follows the rsync precedence rules: earlier filters have higher
    precedence.

    All of the filter options (-a, -e, -f) can be specified multiple times to add
    multiple filters. For the prefix and suffix filters, they are first assembled into a
    list in the order specified and then added to the beginning or end of the filter
    list in the config file. So e.g. `-a "+ foo" -a "- bar"` adds [`+ foo`, `- bar`] to
    the beginning of the filter list, meaning the include filter `+ a` has higher
    precedence than the exclude filter `- bar`.
    """
    log_cli_inputs(kwargs)
    config_files: list[Path] = kwargs.pop("config_files")
    if not config_files:
        ctx.fail("No configuration files provided." + "\n" + ctx.get_help())
    else:
        if kwargs["nofilter"]:
            if (
                kwargs["filter_override"]
                or kwargs["filter_prefix"]
                or kwargs["filter_suffix"]
            ):
                ctx.fail(
                    "Cannot use both `--no-filter` and `-f|a|e` options together."
                    + "\n"
                    + ctx.get_help()
                )
            else:
                kwargs["filter_override"] = []
        elif not kwargs["filter_override"]:
            kwargs["filter_override"] = None
        try:
            verbosity = kwargs.pop("verbosity")
            return sync_from_yaml(config_files, kwargs, verbosity).returncode
        except ValidationError as e:
            ctx.fail(f"Configuration error in `sync`: {e}")
