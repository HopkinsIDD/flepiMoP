import os
from abc import ABC, abstractmethod
from typing import Literal, Annotated, Union, Dict
from pathlib import Path
from subprocess import run, CompletedProcess

import click
from pydantic import BaseModel, Field

import shutil
from ..shared_cli import (
    config_files_argument,
    parse_config_files,
    cli,
    mock_context,
)
from ..utils import config
from ..file_paths import create_file_name_for_push

class SyncABC(ABC):
    """
    Defines a remote location to sync files to / from
    """

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
        pass

class S3SyncModel(BaseModel, SyncABC):
    """
    Implementation of `aws s3 sync` based approach to synchronization
    """
    
    type : Literal["s3sync"]
    target : Path
    source : Path

    def sync(self, dryrun: bool, sync_options : dict) -> CompletedProcess:
        pass


class GitModel(BaseModel, SyncABC):
    """
    Implementation of `git` based approach to synchronization
    """

    type : Literal["git"]

    def sync(self, dryrun: bool, sync_options : dict) -> CompletedProcess:
        pass


SyncModel = Annotated[
    Union[RsyncModel, S3SyncModel, GitModel],
    Field(discriminator='type')
]

class SyncProtocols(BaseModel, SyncABC):
    protocols : dict[str, SyncModel]

    def sync(self, dryrun: bool, sync_options: dict) -> CompletedProcess:
        pass
