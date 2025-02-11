import os
from abc import ABC, abstractmethod
from typing import Literal, Annotated, Union, Dict
from pathlib import Path
from subprocess import run, CompletedProcess

from pydantic import BaseModel, Field, ConfigDict

class SyncABC(ABC):
    """
    Defines a remote location to sync files to / from
    """

    model_config = ConfigDict(extra='forbid')

    @abstractmethod
    def sync(self, dryrun : bool, sync_options : dict) -> CompletedProcess: ...

class RsyncModel(BaseModel, SyncABC):
    """
    Implementation of `rsync` based approach to synchronization
    """
    
    type : Literal["rsync"]
    target : Path
    source : Path

    def sync(self, dryrun: bool, sync_options : dict) -> CompletedProcess:
        testcmd = ["rsync", "-av", "--delete", "--dry-run" if dryrun else "", self.source, self.target].join(" ")
        run(["echo", testcmd])

class S3SyncModel(BaseModel, SyncABC):
    """
    Implementation of `aws s3 sync` based approach to synchronization
    """
    
    type : Literal["s3sync"]
    target : Path
    source : Path

    def sync(self, dryrun: bool, sync_options : dict) -> CompletedProcess:
        testcmd = ["aws", "s3", "sync", "-av", "--delete", "--dry-run" if dryrun else "", self.source, self.target].join(" ")
        run(["echo", testcmd])


class GitModel(BaseModel, SyncABC):
    """
    Implementation of `git` based approach to synchronization
    """

    type : Literal["git"]

    def sync(self, dryrun: bool, sync_options : dict) -> CompletedProcess:
        testcmd = ["git", "status"].join(" ")
        run(["echo", testcmd])


SyncModel = Annotated[
    Union[RsyncModel, S3SyncModel, GitModel],
    Field(discriminator='type')
]

class SyncProtocols(BaseModel, SyncABC):
    protocols : Dict[str, SyncModel]

    def sync(self, dryrun: bool, sync_options: dict) -> CompletedProcess:
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


