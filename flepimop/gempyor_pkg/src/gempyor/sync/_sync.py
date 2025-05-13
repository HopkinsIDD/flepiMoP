"""Implementation of sync protocols and config parsing for `gempyor`."""

__all__ = ("sync_from_yaml", "sync_from_dict")


import re
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Annotated, Any, Final, Literal, Pattern

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from ..logging import get_script_logger
from ..utils import _shutil_which
from ._sync_filter import FilterParts, ListSyncFilter, WithFilters


_RSYNC_HOST_REGEX: Final[Pattern[str]] = re.compile(r"^(?P<host>[^:]+):(?P<path>.+)$")


class SyncOptions(BaseModel):
    """
    Override options for sync protocols.

    Attributes:
        protocol: The sync protocol to override if not `None`.
        source_override: Override for the sync source if not `None`.
        target_override: Override for the sync target if not `None`.
        source_append: Append to the source path if not `None`.
        target_append: Append to the target path if not `None`.
        filter_override: Override the sync filters. This is strict override, not an
            append/prepend.
        filter_prefix: Filters to prepend to the filter list.
        filter_suffix: Filters to append to the filter list.
        dry_run: If `True` perform a dry run of the sync operation, otherwise perform
            the sync operation.
        reverse: If `True` perform the sync operation in reverse (swap source and
            target).
        mkpath: If `True` create the target directory if it does not exist.
    """

    protocol: str | None = None
    source_override: str | None = None
    target_override: str | None = None
    source_append: str | None = None
    target_append: str | None = None
    filter_override: ListSyncFilter | None = None
    filter_prefix: ListSyncFilter = []
    filter_suffix: ListSyncFilter = []
    # n.b. a filter_override = [] would be a valid override, e.g. to clear filters

    dry_run: bool = False
    reverse: bool = False
    mkpath: bool = False
    sep: str = "/"

    # allow potentially other fields for external modules
    model_config = ConfigDict(extra="allow")

    def _true_path(self, original: str, override: str | None, append: str | None) -> str:
        """
        Determine the true path based on the original, override, and append paths.

        This method will swap the `original` with the `override` if the `override`
        is not `None`. If the `append` is not `None`, it will be appended to the
        resulting path.

        Args:
            original: The original path.
            override: The override path.
            append: The append path.

        Returns:
            The true path based on the original, override, and append paths.
        """
        path = original if override is None else override
        if append is not None:
            path += append if path.endswith(self.sep) else f"{self.sep}{append}"
        return path

    def source(self, source: str) -> str:
        """
        Get the source path based on the override and append options.

        Returns:
            The resolved source path.
        """
        return self._true_path(source, self.source_override, self.source_append)

    def target(self, target: str) -> str:
        """
        Get the target path based on the override and append options.

        Returns:
            The resolved target path.
        """
        return self._true_path(target, self.target_override, self.target_append)


class SyncABC(BaseModel, ABC):
    """Abstract base class for a sync protocol."""

    model_config = ConfigDict(extra="forbid")

    @singledispatchmethod
    def execute(self, sync_options, verbosity: int = 0) -> CompletedProcess:
        """
        Perform the sync operation potentially with modifying options.

        Args:
            sync_options: The options to override the sync operation.
            verbosity: The verbosity level of the sync operation.

        Returns:
            A completed process object with the result of the sync operation.

        Raises:
            ValueError: If the `sync_options` is not a `SyncOptions` or `dict`.
        """
        raise ValueError(
            "Invalid `execute(sync_options = ...)`; must be a `SyncOptions` or `dict`"
        )

    @execute.register
    def _sync_impl(self, sync_options: SyncOptions, verbosity: int = 0) -> CompletedProcess:
        return self._sync_pydantic(sync_options, verbosity=verbosity)

    @execute.register
    def _sync_dict(self, sync_options: dict, verbosity: int = 0) -> CompletedProcess:
        return self._sync_pydantic(SyncOptions(**sync_options), verbosity=verbosity)

    @abstractmethod
    def _sync_pydantic(
        self, sync_options: SyncOptions, verbosity: int = 0
    ) -> CompletedProcess:
        """
        Perform the sync operation with the given options.

        Args:
            sync_options: The options to override the sync operation.
            verbosity: The verbosity level of the sync operation.

        Returns:
            A completed process object with the result of the sync operation.
        """


def _rsync_ensure_path(target: str, verbosity: int, dry_run: bool) -> CompletedProcess:
    """
    Ensure the target path exists for rsync.

    Args:
        target: The target path to ensure exists.
        verbosity: The verbosity level of the sync operation.
        dry_run: If `True`, perform a dry run of the command.

    Returns:
        A completed process object with the result of the command executed.

    Examples:
        >>> from gempyor.sync._sync import _rsync_ensure_path
        >>> _rsync_ensure_path("/foo/bar", 0, True)
        (DRY RUN): mkdir -p /foo/bar
        CompletedProcess(args=['echo', '(DRY RUN): mkdir -p /foo/bar'], returncode=0)
        >>> _rsync_ensure_path("user@host:/fizz/buzz", 0, True)
        (DRY RUN): ssh user@host mkdir -p /fizz/buzz
        CompletedProcess(args=['echo', '(DRY RUN): ssh user@host mkdir -p /fizz/buzz'], returncode=0)
    """
    logger = get_script_logger(__name__, verbosity)
    cmd = ["mkdir", "-p"]
    if (tarmatch := _RSYNC_HOST_REGEX.match(target)) is not None:
        cmd = ["ssh", tarmatch.group("host")] + cmd + [tarmatch.group("path")]
        logger.info("Ensuring target directory %s exists with command: %s", target, cmd)
    else:
        cmd += [target]
        logger.info("Ensuring target directory %s exists with command: %s", target, cmd)
    if dry_run:
        cmd = ["echo", " ".join(["(DRY RUN):"] + cmd)]
        logger.debug("Executing command %s instead since dry run.", str(cmd))
    return run(cmd, check=True)


class RsyncModel(SyncABC, WithFilters):
    """
    `SyncABC` Implementation of `rsync` based approach to synchronization.

    Attributes:
        type: The type of sync protocol, which is always "rsync".
        target: The target directory to sync to.
        source: The source directory to sync from.
    """

    type: Literal["rsync"]
    target: str
    source: str

    @staticmethod
    def _formatter(f: FilterParts) -> list[str]:
        return ["--filter", f"{f[0]} {f[1]}"]

    def _sync_pydantic(
        self, sync_options: SyncOptions, verbosity: int = 0
    ) -> CompletedProcess:
        logger = get_script_logger(__name__, verbosity)
        inner_paths = [sync_options.source(self.source), sync_options.target(self.target)]
        logger.debug("Resolved paths: %s", str(inner_paths))
        if sync_options.reverse:
            inner_paths.reverse()
            logger.debug("Reversed paths, now resolved: %s", str(inner_paths))
        if not Path(inner_paths[0]).exists():
            logger.error("Source path %s does not exist", inner_paths[0])
        if sync_options.mkpath:
            proc = _rsync_ensure_path(inner_paths[1], verbosity, sync_options.dry_run)
            if proc.returncode != 0:
                logger.error(
                    "Failed to ensure target directory exists with command, "
                    "received return code %u with output: %s",
                    proc.returncode,
                    proc.stdout,
                )
                return proc
        inner_filter = self.format_filters(
            sync_options.filter_override,
            sync_options.filter_prefix,
            sync_options.filter_suffix,
        )
        logger.debug("Resolved filters: %s", str(inner_filter))
        cmd = (
            [_shutil_which("rsync"), "--archive", "--compress", "--prune-empty-dirs"]
            + inner_filter
            + (["--verbose"] if verbosity > 1 else [])
            + (["--dry-run"] if sync_options.dry_run else [])
            + [str(ip) for ip in inner_paths]
        )
        logger.info("Executing command: %s", str(cmd))
        return run(cmd, check=True)


class S3SyncModel(SyncABC, WithFilters):
    """
    Implementation of `aws s3 sync` based approach to synchronization.

    Attributes:
        type: The type of sync protocol, which is always "s3sync".
        target: The target S3 bucket or path.
        source: The source S3 bucket or path.

    Examples:
        >>> from gempyor.sync._sync import S3SyncModel
        >>> s3_sync = S3SyncModel(
        ...     type="s3sync",
        ...     target="s3://mybucket/target/",
        ...     source="s3://mybucket/source/",
        ... )
        >>> s3_sync.type
        's3sync'
        >>> s3_sync.target
        's3://mybucket/target/'
        >>> s3_sync.source
        's3://mybucket/source/'
        >>> S3SyncModel(type="s3sync", target="target", source="source")
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for S3SyncModel
        Value error, At least one of `source` or `target` must be an s3 bucket, as indicated by a `s3://` prefix [type=value_error, input_value={'type': 's3sync', 'targe...et', 'source': 'source'}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error
    """

    type: Literal["s3sync"]
    target: str
    source: str

    @model_validator(mode="after")
    def _check_at_least_one_bucket(self) -> "S3SyncModel":
        """
        Check that one of the source or target is an S3 bucket.

        Raises:
            ValueError: If neither source nor target is an S3 bucket.
        """
        if self.target.startswith("s3://") or self.source.startswith("s3://"):
            return self
        raise ValueError(
            "At least one of `source` or `target` must be "
            "an s3 bucket, as indicated by a `s3://` prefix"
        )

    @staticmethod
    def _formatter(f: FilterParts) -> list[str]:
        return ["--exclude" if f[0] == "-" else "--include", f'"{f[1]}"']

    def _sync_pydantic(
        self, sync_options: SyncOptions, verbosity: int = 0
    ) -> CompletedProcess:
        logger = get_script_logger(__name__, verbosity)
        inner_paths = [
            (p if p.endswith("/") else f"{p}/")
            for p in (sync_options.source(self.source), sync_options.target(self.target))
        ]
        logger.debug("Resolved paths: %s", str(inner_paths))
        if sync_options.reverse:
            inner_paths.reverse()
            logger.debug("Reversed paths, now resolved: %s", str(inner_paths))
        inner_filter = self.format_filters(
            sync_options.filter_override,
            sync_options.filter_prefix,
            sync_options.filter_suffix,
            reverse=True,
        )
        logger.debug("Resolved filters: %s", str(inner_filter))
        cmd = (
            [_shutil_which("aws"), "s3", "sync"]
            + (["--dryrun"] if sync_options.dry_run else [])
            + inner_filter
            + inner_paths
        )
        logger.info("Executing command: %s", str(cmd))
        return run(cmd, check=True)


class GitModel(SyncABC):
    """
    Implementation of `git` based approach to synchronization.

    Attributes:
        type: The type of sync protocol, which is always "git".
        mode: The mode of synchronization, either "push" or "pull".
    """

    type: Literal["git"]
    mode: Literal["push", "pull"]

    def _sync_pydantic(
        self, sync_options: SyncOptions, verbosity: int = 0
    ) -> CompletedProcess:
        logger = get_script_logger(__name__, verbosity)
        mode = self.mode
        logger.debug("Resolved mode: %s", str(mode))
        if sync_options.reverse:
            mode = "pull" if self.mode == "push" else "push"
            logger.debug("Reversed mode, now resolved: %s", str(mode))
        cmd = ["git", mode] + (["--dry-run"] if sync_options.dry_run else [])
        logger.info("Executing command: %s", str(cmd))
        return run(cmd, check=True)


SyncModel = Annotated[RsyncModel | S3SyncModel | GitModel, Field(discriminator="type")]


class SyncProtocols(SyncABC):
    """
    Implementation of a collection of sync protocols.

    Attributes:
        sync: A dictionary of sync protocols, where the keys are protocol names and the
            values are `SyncModel` instances.
    """

    sync: dict[str, SyncModel] = {}

    model_config = ConfigDict(extra="ignore")

    def _sync_pydantic(
        self, sync_options: SyncOptions, verbosity: int = 0
    ) -> CompletedProcess:
        logger = get_script_logger(__name__, verbosity)
        if not self.sync:
            logger.info("No protocols to sync.")
            return run(["echo", "No protocols to sync"], check=True)
        target_protocol = (
            sync_options.protocol if sync_options.protocol else list(self.sync.keys())[0]
        )
        logger.debug("Resolved protocol: %s.", str(target_protocol))
        if proto := self.sync.get(target_protocol):
            logger.info("Executing protocol: %s.", str(proto))
            return proto.execute(sync_options, verbosity)
        logger.error(
            "Protocol '%s' not found, available protocols are: %s.",
            target_protocol,
            ", ".join(self.sync.keys()),
        )
        return run(
            [
                "echo",
                f"Protocol '{target_protocol}' not found, available "
                f"protocols are: {', '.join(self.sync.keys())}.",
            ],
            check=True,
        )


def sync_from_yaml(
    yamlfiles: list[Path], opts: dict[str, Any], verbosity: int = 0
) -> CompletedProcess:
    """
    Parse a list of yaml files into a SyncABC object and execute the sync operation.

    Args:
        yamlfiles: the list of yaml files to parse. Later files will take precedence
            over earlier files.
        opts: the options to override the sync operation.
        verbosity: the verbosity level of the sync operation.

    Returns:
        A completed process object with the result of the sync operation.
    """
    logger = get_script_logger(__name__, verbosity)
    logger.debug("Parsing YAML files for sync protocols: %s", str(yamlfiles))
    syncdef: dict[Literal["sync"], dict[str, Any]] = {"sync": {}}
    for yamlfile in yamlfiles:
        with yamlfile.open("r") as f:
            look = yaml.safe_load(f)
        if (sync := look.get("sync")) is not None:
            logger.debug("Found %u sync protocols in %s.", len(sync), str(yamlfile))
            syncdef["sync"].update(sync)
    logger.info(
        "Parsed %u sync protocols from %u YAML files.", len(syncdef["sync"]), len(yamlfiles)
    )
    logger.debug("Parsed sync protocols: %s", ", ".join(syncdef["sync"].keys()))
    return sync_from_dict(syncdef, opts, verbosity)


def sync_from_dict(
    syncdef: dict[Literal["sync"], dict[str, Any]], opts: dict[str, Any], verbosity: int = 0
) -> CompletedProcess:
    """
    Parse a dictionary into a SyncABC object and execute the sync operation.

    Args:
        syncdef: the dictionary to parse.
        opts: the options to override the sync operation.
        verbosity: the verbosity level of the sync operation.

    Returns:
        A completed process object with the result of the sync operation.
    """
    return SyncProtocols(**syncdef).execute(SyncOptions(**opts), verbosity)
