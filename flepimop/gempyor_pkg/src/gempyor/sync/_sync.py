"""Implementation of sync protocols and config parsing for `gempyor`."""

__all__ = ("sync_from_yaml", "sync_from_dict")


import re
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from .._pydantic_ext import _override_or_val
from ._sync_filter import FilterParts, ListSyncFilter, WithFilters


_RSYNC_HOST_REGEX: Final = re.compile(r"^(?P<host>[^:]+):(?P<path>.+)$")


def _echo_failed(cmd: list[str]) -> CompletedProcess:
    """
    Runs a command and returns the result, echoing the command on failure.

    Args:
        cmd: The command to run.

    Returns:
        A completed process object either with the result of the command or echoing the
        command on failure.

    Raises:
        ValueError: If the command list is empty.

    Examples:
        >>> from gempyor.sync._sync import _echo_failed
        >>> _echo_failed(["which", "ls"])
        /bin/ls
        CompletedProcess(args='which ls', returncode=0)
        >>> _echo_failed(["which", "does-not-exist"])
        `w h i c h   d o e s - n o t - e x i s t` failed with return code 1
        CompletedProcess(args=['echo', '`w h i c h   d o e s - n o t - e x i s t` failed with return code 1'], returncode=0)
        >>> _echo_failed(["./does-not-exist"])
        /bin/sh: ./does-not-exist: No such file or directory
        `. / d o e s - n o t - e x i s t` failed with return code 127
        CompletedProcess(args=['echo', '`. / d o e s - n o t - e x i s t` failed with return code 127'], returncode=0)
    """
    if not cmd:
        raise ValueError("The command cannot be empty.")
    try:
        res = run(cmd, shell=True)
        if res.returncode != 0:
            return run(
                ["echo", f"`{' '.join(res.args)}` failed with return code {res.returncode}"]
            )
        return res
    except FileNotFoundError:
        return run(["echo", f"command `{cmd[0]}` not found"])


class SyncOptions(BaseModel):
    """
    Override options for sync protocols.

    Attributes:
        protocol: The sync protocol to override if not `None`.
        target_override: Override for the sync target if not `None`.
        source_override: Override for the sync source if not `None`.
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
    target_override: Path | None = None
    source_override: Path | None = None
    filter_override: ListSyncFilter | None = None
    filter_prefix: ListSyncFilter = []
    filter_suffix: ListSyncFilter = []
    # n.b. a filter_override = [] would be a valid override, e.g. to clear filters

    dry_run: bool = False
    reverse: bool = False
    mkpath: bool = False

    # allow potentially other fields for external modules
    model_config = ConfigDict(extra="allow")


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
        return self._sync_pydantic(sync_options, verbosity)

    @execute.register
    def _sync_dict(self, sync_options: dict, verbosity: int = 0) -> CompletedProcess:
        return self._sync_pydantic(SyncOptions(**sync_options), verbosity)

    @abstractmethod
    def _sync_pydantic(
        self, sync_options: SyncOptions = SyncOptions(), verbosity: int = 0
    ) -> CompletedProcess:
        """
        Perform the sync operation with the given options.

        Args:
            sync_options: The options to override the sync operation.
            verbosity: The verbosity level of the sync operation.

        Returns:
            A completed process object with the result of the sync operation.
        """


class RsyncModel(SyncABC, WithFilters):
    """
    `SyncABC` Implementation of `rsync` based approach to synchronization.

    Attributes:
        type: The type of sync protocol, which is always "rsync".
        target: The target directory to sync to.
        source: The source directory to sync from.
    """

    type: Literal["rsync"]
    target: Path
    source: Path

    @staticmethod
    def _formatter(f: FilterParts) -> list[str]:
        return [f"-f'{f[0]} {f[1]}'"]

    @staticmethod
    def _dry_run(dry: bool) -> list[str]:
        return ["-v", "--dry-run"] if dry else []

    def _cmd(self) -> list[str]:
        return ["rsync", "-avz"]

    def _ensure_path(self, target: Path, dry_run: bool) -> CompletedProcess:
        """
        Ensure the target path exists
        """
        echo = ["echo", "(DRY RUN):"] if dry_run else []
        cmd = ["mkdir", "-p"]
        if tarmatch := _RSYNC_HOST_REGEX.match(str(target)):
            cmd = cmd + [tarmatch.group("path")]
            return run(echo + ["ssh", tarmatch.group("host")] + cmd)
        return run(echo + cmd + [str(target)])

    def _sync_pydantic(
        self, sync_options: SyncOptions = SyncOptions(), verbosity: int = 0
    ) -> CompletedProcess:
        inner_paths = [
            str(_override_or_val(sync_options.source_override, self.source)) + "/",
            str(_override_or_val(sync_options.target_override, self.target)) + "/",
        ]
        if sync_options.reverse:
            inner_paths.reverse()
        if sync_options.mkpath:
            proc = self._ensure_path(inner_paths[1], sync_options.dry_run)
            if proc.returncode != 0:
                return proc

        inner_filter = self.format_filters(
            sync_options.filter_override,
            sync_options.filter_prefix,
            sync_options.filter_suffix,
        )
        testcmd = (
            self._cmd() + inner_filter + self._dry_run(sync_options.dry_run) + inner_paths
        )
        if verbosity > 0:
            print(" ".join(["executing: "] + testcmd))
        return _echo_failed(testcmd)


def _trim_s3_path(path: str | Path) -> str | Path:
    if isinstance(path, str):
        return path.lstrip("s3:")
    return path


class S3SyncModel(SyncABC, WithFilters):
    """
    Implementation of `aws s3 sync` based approach to synchronization.

    Attributes:
        type: The type of sync protocol, which is always "s3sync".
        target: The target S3 bucket or path.
        source: The source S3 bucket or path.
    """

    type: Literal["s3sync"]
    target: Annotated[Path, BeforeValidator(_trim_s3_path)]
    source: Annotated[Path, BeforeValidator(_trim_s3_path)]

    @model_validator(mode="after")
    def _check_at_least_one_bucket(self):
        srcs3 = self.source.root == "//"
        tars3 = self.target.root == "//"
        if srcs3 or tars3:
            return self
        raise ValueError(
            "At least one of `source` or `target` must be "
            "an s3 bucket, as indicated by a `//` prefix"
        )

    @computed_field
    @property
    def s3(self) -> Literal["source", "target", "both"]:
        """
        Determine which of the source or target is an S3 bucket.

        Returns:
            "source" if the source is an S3 bucket, "target" if the target is an S3
            bucket, or "both" if both are S3 buckets.
        """
        srcs3 = self.source.root == "//"
        tars3 = self.target.root == "//"
        if srcs3 and tars3:
            return "both"
        if srcs3:
            return "source"
        return "target"

    @staticmethod
    def _formatter(f: FilterParts) -> list[str]:
        return ["--exclude" if f[0] == "-" else "--include", f'"{f[1]}"']

    @staticmethod
    def _dry_run(dry: bool) -> list[str]:
        return ["--dryrun"] if dry else []

    @staticmethod
    def _cmd() -> list[str]:
        return ["aws", "s3", "sync"]

    def _sync_pydantic(
        self, sync_options: SyncOptions = SyncOptions(), verbosity: int = 0
    ) -> CompletedProcess:
        inner_paths = [
            str(_override_or_val(sync_options.source_override, self.source)) + "/",
            str(_override_or_val(sync_options.target_override, self.target)) + "/",
        ]
        match self.s3:
            case "source":
                inner_paths[0] = "s3:" + inner_paths[0]
            case "target":
                inner_paths[1] = "s3:" + inner_paths[1]
            case "both":
                inner_paths = ["s3:" + p for p in inner_paths]

        if sync_options.reverse:
            inner_paths.reverse()
        inner_filter = self.format_filters(
            sync_options.filter_override,
            sync_options.filter_prefix,
            sync_options.filter_suffix,
            reverse=True,
        )
        testcmd = (
            self._cmd() + self._dry_run(sync_options.dry_run) + inner_filter + inner_paths
        )
        if verbosity > 0:
            print(" ".join(["executing: "] + testcmd))
        return _echo_failed(testcmd)


class GitModel(SyncABC):
    """
    Implementation of `git` based approach to synchronization.

    Attributes:
        type: The type of sync protocol, which is always "git".
        mode: The mode of synchronization, either "push" or "pull".
    """

    type: Literal["git"]
    mode: Literal["push", "pull"]

    @staticmethod
    def _dry_run(dry: bool) -> list[str]:
        return ["--dry-run"] if dry else []

    def _sync_pydantic(
        self, sync_options: SyncOptions = SyncOptions(), verbosity: int = 0
    ) -> CompletedProcess:
        inner_mode = (
            self.mode
            if not sync_options.reverse
            else ("push" if self.mode == "pull" else "pull")
        )
        testcmd = ["git", inner_mode] + self._dry_run(sync_options.dry_run)
        if verbosity > 0:
            print(" ".join(testcmd))
        return _echo_failed(testcmd)


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
        self, sync_options: SyncOptions = SyncOptions(), verbosity: int = 0
    ) -> CompletedProcess:
        if not self.sync:
            return run(["echo", "No protocols to sync"])
        tarproto = (
            sync_options.protocol if sync_options.protocol else list(self.sync.keys())[0]
        )
        if proto := self.sync.get(tarproto):
            return proto.execute(sync_options, verbosity)
        return run(
            [
                "echo",
                f"No protocol `{tarproto}` to sync;",
                f"available protocols are: {', '.join(self.sync.keys())}",
            ]
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
    syncdef: dict[Literal["sync"], dict[str, Any]] = {"sync": {}}
    for yamlfile in yamlfiles:
        with yamlfile.open("r") as f:
            look = yaml.safe_load(f)
        if "sync" in look:
            syncdef["sync"].update(look["sync"])
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
