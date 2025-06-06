import re
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from typing import Literal, Annotated, Union, Any
from pathlib import Path
from subprocess import run, CompletedProcess

import yaml
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    BeforeValidator,
    computed_field,
    model_validator,
)

from .._pydantic_ext import _ensure_list, _override_or_val

__all__ = ["process_from_yaml", "process_from_dict"]


def _echo_failed(cmd: list[str]) -> CompletedProcess:
    try:
        res = run(" ".join(cmd), shell=True)
        if res.returncode != 0:
            return run(
                [
                    "echo",
                    "`{}` failed with return code {}".format(
                        "".join(res.args), res.returncode
                    ),
                ],
                stdout=res.stdout,
                stderr=res.stderr,
            )
        else:
            return res
    except FileNotFoundError as e:
        return run(["echo", "command `{}` not found".format(cmd[0])])

class ProcessArgs(BaseModel):
    arguments : Annotated[list[str], BeforeValidator(_ensure_list)] = []
    process : str | None = None
    dryrun : bool = False

class ProcessABC(BaseModel, ABC):
    """
    Defines an (abstract) object capable of pre-/post-processing
    :method execute: perform the operation, potentially with modifying options
    """

    model_config = ConfigDict(extra="forbid")

    @singledispatchmethod
    def execute(self, arguments, verbosity: int = 0) -> CompletedProcess:
        """
        Perform the sync operation
        :param sync_options: optional: the options to override the sync operation
        """
        raise ValueError(
            "Invalid `execute(options = ...)`; must be a `ProcessArgs` or `list`"
        )

    @execute.register
    def _process_impl(self, arguments : ProcessArgs, verbosity: int = 0) -> CompletedProcess:
        return self._process_pydantic(arguments, verbosity)

    @execute.register
    def _process_dict(self, arguments : dict, verbosity: int = 0) -> CompletedProcess:
        return self._process_pydantic(ProcessArgs(arguments = arguments), verbosity)

    @execute.register
    def _process_str(self, arguments : str, verbosity: int = 0) -> CompletedProcess:
        return self._process_pydantic(ProcessArgs(arguments = arguments), verbosity)

    @abstractmethod
    def _process_pydantic(self, arguments : ProcessArgs, verbosity: int = 0) -> CompletedProcess: ...

class BashProcess(ProcessABC):
    """
    `SyncABC` Implementation of `rsync` based approach to synchronization
    """

    type: Literal["bash"]
    command: str

    def _process_pydantic(
        self, arguments : ProcessArgs, verbosity: int = 0
    ) -> CompletedProcess:
        cmd = ["echo"] if arguments.dryrun else []
        cmd += [self.command] + arguments.arguments
        if verbosity > 0:
            print(" ".join(["executing: "] + cmd))
        return _echo_failed(cmd)

class RScriptProcess(ProcessABC):
    """
    Implementation of `Rscript $targetscript config.yml [arguments]` process
    """

    type: Literal["rscript"]
    script: str # TODO add regex confirming that script ends in R or r?

    def _process_pydantic(
        self, arguments : ProcessArgs, verbosity: int = 0
    ) -> CompletedProcess:
        cmd = ["echo"] if arguments.dryrun else []
        cmd += ["Rscript", self.script] + arguments.arguments
        if verbosity > 0:
            print(" ".join(["executing: "] + cmd))
        return _echo_failed(cmd)

ProcessProtocol = Annotated[Union[BashProcess, RScriptProcess], Field(discriminator="type")]

class ProcessProtocols(ProcessABC):
    process: dict[str, ProcessProtocol] = {}

    model_config = ConfigDict(extra="ignore")

    def _process_pydantic(
        self, arguments: ProcessArgs, verbosity: int = 0
    ) -> CompletedProcess:
        if not self.process:
            return run(["echo", "No process(es) to execute"])
        else:
            tarproto = (
                arguments.process
                if arguments.process
                else list(self.process.keys())[0]
            )
            if proto := self.process.get(tarproto):
                return proto.execute(arguments, verbosity)
            else:
                return run(
                    [
                        "echo",
                        "No process `{}` to execute;".format(tarproto),
                        "available process(es) are: {}".format(", ".join(self.process.keys())),
                    ]
                )


def process_from_yaml(
    yamlfiles: list[Path], opts: dict[str, Any], verbosity: int = 0
) -> CompletedProcess:
    """
    Parse a list of yaml files into a ProcessABC object

    :param yamlfiles: the list of yaml files to parse
      n.b. the order of the files is important: later files have precedence over earlier files
      so protocols in later files will override protocols in earlier files, though sync options
      can be specified across multiple files
    """
    procdef: dict[Literal["process"], dict[str, Any]] = {"process": {}}
    for yf in yamlfiles:
        with open(yf, "r") as handle:
            look = yaml.safe_load(handle)
            if "process" in look:
                procdef["process"].update(look["process"])

    return process_from_dict(procdef, opts, verbosity)


def process_from_dict(
    procdef: dict[Literal["process"], dict[str, Any]],
    opts: dict[str, Any],
    verbosity: int = 0,
) -> CompletedProcess:
    """
    Parse a dictionary into a ProcessABC object

    :param syncdef: the dictionary to parse
    """
    return ProcessProtocols(**procdef).execute(ProcessArgs(**opts), verbosity)
