import re
from abc import ABC, abstractmethod
from typing import Literal, Annotated, Union, Dict, List, Optional, overload
from pathlib import Path
from subprocess import run, CompletedProcess

from pydantic import BaseModel, Field, ConfigDict

FilterRegex = r'^([+-] )?'
frcompiled = re.compile(FilterRegex)

SyncFilter = Annotated[str, Field(pattern=FilterRegex)]

def _filter_mode(filter : SyncFilter) -> Literal["+", "-"]:
    return "-" if filter.startswith("- ") else "+"

def _filter_pattern(filter : SyncFilter) -> str:
    return frcompiled.sub("", filter)

def _filter_parse(filter : SyncFilter) -> tuple[Literal["+", "-"], str]:
    return (_filter_mode(filter), _filter_pattern(filter))

class SyncOptions(BaseModel):
    """
    The potential overriding options for a sync operation
    :param target_override: optional: override the sync target
    :param source_override: optional: override the sync source
    :param filter_override: optional: override the sync filters; n.b. this is strict override, not an append/prepend
    :param dryrun: optional (default: false) perform a dry run of the sync operation
    :param reverse: optional (default: false) perform the sync operation in reverse (e.g. swap source and target)
    """

    target_override : Optional[Path] = None
    source_override : Optional[Path] = None
    filter_override : Optional[List[SyncFilter]] = None

    dryrun : bool = False
    reverse : bool = False

    # allow potentially other fields for external modules
    model_config = ConfigDict(extra='allow')

class SyncABC(ABC):
    """
    Defines an (abstract) object capable of sync files to / from a remote resource
    :method sync: perform the sync operation, potentially with modifying options
    """

    model_config = ConfigDict(extra='forbid')

    @abstractmethod
    def sync(self, sync_options : SyncOptions = SyncOptions()) -> CompletedProcess:
        """
        Perform the sync operation
        """
        ...

class RsyncModel(BaseModel, SyncABC):
    """
    `SyncABC` Implementation of `rsync` based approach to synchronization
    """
    
    type : Literal["rsync"]
    target : Path
    source : Path
    filters : List[SyncFilter] = []

    @staticmethod
    def _format_filters(filters : list[SyncFilter]) -> list[str]:
        return ["-f'{} {}'".format("-" if mode == "exclude" else "+", filt) for (mode, filt) in (_filter_parse(f) for f in filters)]

    def sync(self, sync_options : SyncOptions = SyncOptions()) -> CompletedProcess:
        inner_tar = self.target if not sync_options.reverse else self.source
        inner_src = self.source if not sync_options.reverse else self.target
        inner_filter = self.filters if not sync_options.filter_override else sync_options.filter_override
        testcmd = ["rsync", "-avz"] + (["-v", "--dry-run"] if sync_options.dryrun else []) + self._format_filters(inner_filter) + [inner_src, inner_tar]
        return run(["echo"] + testcmd)

class S3SyncModel(BaseModel, SyncABC):
    """
    Implementation of `aws s3 sync` based approach to synchronization
    """
    
    type : Literal["s3sync"]
    target : Path
    source : Path
    filter : List[SyncFilter] = []

    @staticmethod
    def _format_filters(filters : list[SyncFilter]) -> list[str]:
        return ["-f'{} {}'".format("-" if mode == "exclude" else "+", filt) for (mode, filt) in (_filter_parse(f) for f in filters)]

    def sync(self, sync_options : SyncOptions = SyncOptions()) -> CompletedProcess:
        inner_tar = self.target if not sync_options.reverse else self.source
        inner_src = self.source if not sync_options.reverse else self.target
        inner_filter = self.filters if not sync_options.filter_override else sync_options.filter_override
        testcmd = ["rsync", "-avz"] + (["-v", "--dry-run"] if sync_options.dryrun else []) + self._format_filters(inner_filter) + [inner_src, inner_tar]
        return run(["echo"] + testcmd)
    
        testcmd = ["aws", "s3", "sync", "-av", "--delete", "--dry-run" if dryrun else "", self.source, self.target].join(" ")
        return run(["echo", testcmd])


class GitModel(BaseModel, SyncABC):
    """
    Implementation of `git` based approach to synchronization
    """

    type : Literal["git"]
    mode : Literal["push", "pull"]

    def sync(self, sync_options : dict) -> CompletedProcess:
        testcmd = ["git", "status"].join(" ")
        return run(["echo", testcmd])


SyncModel = Annotated[
    Union[RsyncModel, S3SyncModel, GitModel],
    Field(discriminator='type')
]

class SyncProtocols(BaseModel, SyncABC):
    protocols : Dict[str, SyncModel]

    def sync(self, dryrun: bool, reverse : bool, sync_options: dict) -> CompletedProcess:
        if not self.protocols:
            return run(["echo", "No protocols to sync"])
        elif protoname := sync_options['protocol']:
            if protocol := self.protocols.get(protoname):
                return protocol.sync(dryrun, sync_options)
            else:
                return run(["echo", f"No protocol named {protoname} found"])
        else:
            res = { k: v.sync(dryrun, sync_options) for k, v in self.protocols.items() }
            accargs = []
            allargs = [accargs := accargs + list(v.args) for v in res.values()]
            return CompletedProcess(
                args = allargs,
                returncode = min([ v.returncode for v in res.values() ]),
                stdout = "\n".join([ v.stdout for v in res.values() ]),
                stderr = "\n".join([ v.stderr for v in res.values() ]),
            )


