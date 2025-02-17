import re
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Literal, Annotated, Union, Dict, List, Any
from pathlib import Path
from subprocess import run, CompletedProcess
from itertools import chain

import yaml
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator, computed_field, ValidationError

__all__ = ["SyncABC", "sync_from_yaml"]

# filters can begin with a `+` or `-` followed by a space, but not just those symbols or a space
FilterRegex = r'^([\+\-] |[^\+\- ])'
frcompiled = re.compile(r'^([\+\-] )?')

# once 3.12, use type parametrization
# def _ensure_list[T](value: T | list[T]) -> list[T]:
def _ensure_list(value: Any) -> list | None:
    if value is None:
        return None
    else:
        return [value] if not isinstance(value, list) else value

SyncFilter = Annotated[str, Field(pattern=FilterRegex)]

ListSyncFilter = Annotated[List[SyncFilter], BeforeValidator(_ensure_list)]

def _filter_mode(filter : SyncFilter) -> Literal["+", "-"]:
    return "-" if filter.startswith("- ") else "+"

def _filter_pattern(filter : SyncFilter) -> str:
    return frcompiled.sub("", filter)

def _filter_parse(filter : SyncFilter) -> tuple[Literal["+", "-"], str]:
    return (_filter_mode(filter), _filter_pattern(filter))

# once 3.12, use type parametrization
# def _override_or_val[T](override : T | None, value : T) -> T:
def _override_or_val(override : Any, value : Any) -> Any:
    return value if override is None else override

def _echo_failed(cmd : list[str]) -> CompletedProcess:
    try:
        res = run(cmd)
        if res.returncode != 0:
            return run(
                ["echo", "`{}` failed with return code {}".format(" ".join(res.args), res.returncode)],
                stdout=res.stdout,
                stderr=res.stderr
            )
        else:
            return res
    except FileNotFoundError as e:
        return run(["echo", "command `{}` not found".format(cmd[0])])

class SyncOptions(BaseModel):
    """
    The potential overriding options for a sync operation
    :param target_override: optional: override the sync target
    :param source_override: optional: override the sync source
    :param filter_override: optional: override the sync filters; n.b. this is strict override, not an append/prepend
    :param dryrun: optional (default: false) perform a dry run of the sync operation
    :param reverse: optional (default: false) perform the sync operation in reverse (e.g. swap source and target)
    """

    protocol: str | None = None
    target_override : Path | None = None
    source_override : Path | None = None
    filter_override : ListSyncFilter | None = None
    # n.b. a filter_override = [] would be a valid override, e.g. to clear filters

    dryrun : bool = False
    reverse : bool = False

    # allow potentially other fields for external modules
    model_config = ConfigDict(extra='allow')


class SyncABC(ABC):
    """
    Defines an (abstract) object capable of sync files to / from a remote resource
    :method execute: perform the sync operation, potentially with modifying options
    """

    model_config = ConfigDict(extra='forbid')

    @singledispatchmethod
    def execute(self, sync_options, verbosity : int = 0) -> CompletedProcess:
        """
        Perform the sync operation
        :param sync_options: optional: the options to override the sync operation
        """
        raise ValueError("Invalid `execute(sync_options = ...)`; must be a `SyncOptions` or `dict`")

    @execute.register
    def _sync_impl(self, sync_options : SyncOptions, verbosity : int = 0) -> CompletedProcess:
        return self._sync_pydantic(sync_options, verbosity)

    @execute.register
    def _sync_dict(self, sync_options : dict, verbosity : int = 0) -> CompletedProcess:
        return self._sync_pydantic(SyncOptions(**sync_options), verbosity)
    
    @abstractmethod
    def _sync_pydantic(self, sync_options : SyncOptions = SyncOptions(), verbosity : int = 0) -> CompletedProcess:
        ...


class RsyncModel(BaseModel, SyncABC):
    """
    `SyncABC` Implementation of `rsync` based approach to synchronization
    """
    
    type : Literal["rsync"]
    target : Path
    source : Path
    filters : ListSyncFilter = []

    @staticmethod
    def _format_filters(filters : ListSyncFilter) -> list[str]:
        return ["-f{} {}".format("-" if mode == "-" else "+", filt) for (mode, filt) in (_filter_parse(f) for f in filters)]

    @staticmethod
    def _dryrun(dry : bool) -> list[str]:
        return ["-v", "--dry-run"] if dry else []

    @staticmethod
    def _cmd() -> list[str]:
        return ["rsync", "-avz"]

    def _sync_pydantic(self, sync_options : SyncOptions = SyncOptions(), verbosity : int = 0) -> CompletedProcess:
        inner_paths = [str(_override_or_val(sync_options.source_override, self.source)) + "/", str(_override_or_val(sync_options.target_override, self.target)) + "/"]
        if sync_options.reverse:
            inner_paths.reverse()
        inner_filter = self._format_filters(_override_or_val(sync_options.filter_override, self.filters))
        testcmd = self._cmd() + inner_filter + self._dryrun(sync_options.dryrun) + inner_paths
        if verbosity > 0:
            print(" ".join(testcmd))
        return _echo_failed(testcmd)


class S3SyncModel(BaseModel, SyncABC):
    """
    Implementation of `aws s3 sync` based approach to synchronization
    """
    
    type : Literal["s3sync"]
    target : Path
    source : Path
    filters : ListSyncFilter = []

    @computed_field
    @property
    def s3(self) -> Literal["source", "target", "both"]:
        srcs3 = self.source.root == "//"
        tars3 = self.target.root == "//"
        if srcs3 and tars3:
            return "both"
        elif srcs3:
            return "source"
        elif tars3:
            return "target"
        else:
            raise ValidationError("Neither source nor target are S3 paths; at least one must begin with `//`")

    @staticmethod
    def _format_filters(filters : ListSyncFilter) -> list[str]:
        return list(chain.from_iterable(
            [["--exclude" if mode == "-" else "--include", '"{}"'.format(filt)] for (mode, filt) in (_filter_parse(f) for f in reversed(filters))]
        ))

    @staticmethod
    def _dryrun(dry : bool) -> list[str]:
        return ["--dryrun"] if dry else []

    @staticmethod
    def _cmd() -> list[str]:
        return ["aws", "s3", "sync"]

    def _sync_pydantic(self, sync_options : SyncOptions = SyncOptions(), verbosity : int = 0) -> CompletedProcess:
        inner_paths = [str(_override_or_val(sync_options.source_override, self.source)) + "/", str(_override_or_val(sync_options.target_override, self.target)) + "/"]
        match self.s3:
            case "source":
                inner_paths[0] = "s3:" + inner_paths[0]
            case "target":
                inner_paths[1] = "s3:" + inner_paths[1]
            case "both":
                inner_paths = ["s3:" + p for p in inner_paths]
        
        if sync_options.reverse:
            inner_paths.reverse()
        inner_filter = self._format_filters(_override_or_val(sync_options.filter_override, self.filters))
        testcmd = self._cmd() + self._dryrun(sync_options.dryrun) + inner_filter + inner_paths
        if verbosity > 0:
            print(" ".join(testcmd))
        return _echo_failed(testcmd)


class GitModel(BaseModel, SyncABC):
    """
    Implementation of `git` based approach to synchronization
    """

    type : Literal["git"]
    mode : Literal["push", "pull"]

    @staticmethod
    def _dryrun(dry : bool) -> list[str]:
        return ["--dry-run"] if dry else []

    def _sync_pydantic(self, sync_options : SyncOptions = SyncOptions(), verbosity : int = 0) -> CompletedProcess:
        inner_mode = self.mode if not sync_options.reverse else ("push" if self.mode == "pull" else "pull")
        testcmd = ["git", inner_mode] + self._dryrun(sync_options.dryrun)
        if verbosity > 0:
            print(" ".join(testcmd))
        return _echo_failed(testcmd)


SyncModel = Annotated[
    Union[RsyncModel, S3SyncModel, GitModel],
    Field(discriminator='type')
]

class SyncProtocols(BaseModel, SyncABC):
    sync : dict[str, SyncModel] = {}

    model_config = ConfigDict(extra='ignore')

    def _sync_pydantic(self, sync_options: SyncOptions = SyncOptions(), verbosity : int = 0) -> CompletedProcess:
        if not self.sync:
            return run(["echo", "No protocols to sync"])
        else:
            tarproto = sync_options.protocol if sync_options.protocol else list(self.sync.keys())[0]
            if proto := self.sync.get(tarproto):
                return proto.execute(sync_options, verbosity)
            else:
                return run(["echo", "No protocol `{}` to sync;".format(tarproto), "available protocols are: {}".format(", ".join(self.sync.keys()))])

def sync_from_yaml(yamlfiles : List[Path]) -> SyncABC:
    """
    Parse a list of yaml files into a SyncABC object

    :param yamlfiles: the list of yaml files to parse
      n.b. the order of the files is important: later files have precedence over earlier files
      so protocols in later files will override protocols in earlier files, though sync options
      can be specified across multiple files
    """
    syncdef = { 'sync' : {} }
    for yf in yamlfiles:
        with open(yf, 'r') as handle:
            look = yaml.safe_load(handle)
            if 'sync' in look:
                syncdef['sync'].update(look['sync'])
    
    return SyncProtocols(**syncdef)