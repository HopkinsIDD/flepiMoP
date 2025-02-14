"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including:
* Objects for representing job resources and job sizes,
* A system for registering and getting batch systems, and
* Functionalities for writing job metadata to a manifest file.
"""

__all__ = (
    "BatchSystem",
    "JobResources",
    "JobResult",
    "JobSize",
    "JobSubmission",
    "LocalBatchSystem",
    "SlurmBatchSystem",
    "get_batch_system",
    "register_batch_system",
    "write_manifest",
)


from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from getpass import getuser
from itertools import product
import json
from logging import Logger
import math
from pathlib import Path
import platform
import re
import shutil
from stat import S_IXUSR
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Callable, Literal, overload
import warnings

import click
import confuse
from pydantic import BaseModel, Field, PositiveInt, computed_field, model_validator

from ._click import DurationParamType, MemoryParamType
from ._jinja import _jinja_environment
from .file_paths import run_id
from .info import get_cluster_info
from .logging import get_script_logger
from .shared_cli import (
    cli,
    config_files_argument,
    config_file_options,
    log_cli_inputs,
    mock_context,
    parse_config_files,
    verbosity_options,
)
from .utils import (
    _dump_formatted_yaml,
    _format_cli_options,
    _git_checkout,
    _git_head,
    _shutil_which,
    config,
)


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


_JOB_NAME_REGEX = re.compile(r"^[a-z]{1}([a-z0-9\_\-]+)?$", flags=re.IGNORECASE)
_SAMPLES_SIMULATIONS_RATIO: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.6

_batch_systems = []


class JobResources(BaseModel):
    """
    A batch submission job resources request.

    Attributes:
        nodes: The number of nodes to request.
        cpus: The number of CPUs to request per a node.
        memory: The amount of memory to request per a node in MB.

    Raises:
        ValueError: If any of the attributes are less than 1.

    Examples:
        >>> from gempyor.batch import JobResources
        >>> resources = JobResources(nodes=5, cpus=10, memory=10*1024)
        >>> resources
        JobResources(nodes=5, cpus=10, memory=10240)
        >>> resources.total_cpus
        50
        >>> resources.total_memory
        51200
        >>> resources.total_resources()
        (5, 50, 51200)
        >>> try:
        ...     JobResources(nodes=0, cpus=1, memory=1024)
        ... except Exception as e:
        ...     print(e)
        1 validation error for JobResources
        nodes
        Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
            For further information visit https://errors.pydantic.dev/2.10/v/greater_than
        >>> try:
        ...     JobResources(nodes=2, cpus=4.5, memory=1024)
        ... except Exception as e:
        ...     print(e)
        1 validation error for JobResources
        cpus
        Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=4.5, input_type=float]
            For further information visit https://errors.pydantic.dev/2.10/v/int_from_float
    """

    nodes: PositiveInt
    cpus: PositiveInt
    memory: PositiveInt

    @property
    def total_cpus(self) -> PositiveInt:
        """
        Calculate the total number of CPUs.

        Returns:
            The total number of CPUs represented by this instance.
        """
        return self.nodes * self.cpus

    @property
    def total_memory(self) -> PositiveInt:
        """
        Calculate the total amount of memory.

        Returns:
            The total amount of memory represented by this instance.
        """
        return self.nodes * self.memory

    def total_resources(self) -> tuple[PositiveInt, PositiveInt, PositiveInt]:
        """
        Calculate the total resources.

        Returns:
            A tuple of the nodes, total CPUs, and total memory represented by
            this instance.
        """
        return (self.nodes, self.total_cpus, self.total_memory)


class JobSize(BaseModel):
    """
    A batch submission job size.

    Attributes:
        chains: The number of chains, or equivalent concept, to use.
        blocks: The number of sequential blocks to run per a chain.
        samples: The number of samples to run per a block.
        simulations: The number of simulations to run per a block.

    Raises:
        ValueError: If any of the attributes are less than 1.

    Examples:
        >>> import warnings
        >>> from gempyor.batch import JobSize
        >>> JobSize(blocks=5, chains=10, samples=100, simulations=200)
        JobSize(blocks=5, chains=10, samples=100, simulations=200)
        >>> JobSize(chains=32, simulations=500)
        JobSize(blocks=None, chains=32, samples=None, simulations=500)
        >>> try:
        ...     JobSize(blocks=0, chains=12, simulations=100)
        ... except Exception as e:
        ...     print(e)
        ...
        1 validation error for JobSize
        blocks
        Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
            For further information visit https://errors.pydantic.dev/2.10/v/greater_than
        >>> try:
        ...     JobSize(chains=10, samples=50.5, simulations=200)
        ... except Exception as e:
        ...     print(e)
        ...
        1 validation error for JobSize
        samples
        Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=50.5, input_type=float]
            For further information visit https://errors.pydantic.dev/2.10/v/int_from_float
        >>> try:
        ...     JobSize(samples=100, simulations=50)
        ... except Exception as e:
        ...     print(e)
        ...
        1 validation error for JobSize
        Value error, The number of samples, 100, must be less than or equal to the number of simulations, 50, per a block. [type=value_error, input_value={'samples': 100, 'simulations': 50}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.10/v/value_error
        >>> with warnings.catch_warnings(record=True) as warns:
        ...     JobSize(samples=75, simulations=100)
        ...     for warn in warns:
        ...             print(warn.message)
        ...
        JobSize(blocks=None, chains=None, samples=75, simulations=100)
        The samples to simulations ratio is 75%, which is higher than the recommended limit of 60%.
        >>> JobSize()
        JobSize(blocks=None, chains=None, samples=None, simulations=None)
        >>> size = JobSize(blocks=4, chains=3, samples=10, simulations=25)
        >>> size.samples_per_chain
        40
        >>> size.simulations_per_chain
        100
        >>> size.total_samples
        120
        >>> size.total_simulations
        300
    """

    blocks: PositiveInt | None = None
    chains: PositiveInt | None = None
    samples: PositiveInt | None = None
    simulations: PositiveInt | None = None

    @staticmethod
    def _scale(x: PositiveInt | None, y: PositiveInt | None) -> PositiveInt | None:
        """
        Scale `x` by `y`.

        Args:
            x: The number to scale.
            y: The number to scale by.

        Returns:
            The scaled number or `None` if `x` is `None`.
        """
        return None if x is None else (x if y is None else x * y)

    def _total(self, x: PositiveInt | None) -> PositiveInt | None:
        """
        Calculate the total number of `x` by scaling by the number of chains.

        Args:
            x: The number of `x` to calculate the total of.

        Returns:
            The total number of `x` or `None` if `x` is `None`.
        """
        return self._scale(x, self.chains)

    def _per_chain(self, x: PositiveInt | None) -> PositiveInt | None:
        """
        Calculate the number of `x` per a chain by scaling by the number of blocks.

        Args:
            x: The number of `x` to calculate per a chain.

        Returns:
            The number of `x` per a chain or `None` if `x` is `None`.
        """
        return self._scale(x, self.blocks)

    @computed_field
    @property
    def samples_per_chain(self) -> PositiveInt | None:
        """
        Calculate the number of samples per a chain.

        Multiplies the number of samples by the number of blocks. If blocks is `None`
        then this is the same as the number of samples.

        Returns:
            The number of samples per a chain.
        """
        return self._per_chain(self.samples)

    @computed_field
    @property
    def simulations_per_chain(self) -> PositiveInt | None:
        """
        Calculate the number of simulations per a chain.

        Multiplies the number of simulations by the number of blocks. If blocks is
        `None` then this is the same as the number of simulations.

        Returns:
            The number of simulations per a chain.
        """
        return self._per_chain(self.simulations)

    @computed_field
    @property
    def total_samples(self) -> PositiveInt | None:
        """
        Calculate the total number of samples.

        Multiplies the number of samples by the number of chains. If chains is `None`
        then this is the same as the number of samples.

        Returns:
            The total number of samples.
        """
        return self._total(self.samples_per_chain)

    @computed_field
    @property
    def total_simulations(self) -> PositiveInt | None:
        """
        Calculate the total number of simulations.

        Multiplies the number of simulations by the number of chains. If chains is `None`
        then this is the same as the number of simulations.

        Returns:
            The total number of simulations.
        """
        return self._total(self.simulations_per_chain)

    @model_validator(mode="after")
    def check_samples_is_consistent(self) -> Self:
        if self.samples is not None and self.simulations is not None:
            if self.samples > self.simulations:
                raise ValueError(
                    f"The number of samples, {self.samples}, must be less than or equal to "
                    f"the number of simulations, {self.simulations}, per a block."
                )
            elif (ratio := (self.samples / self.simulations)) > _SAMPLES_SIMULATIONS_RATIO:
                warnings.warn(
                    f"The samples to simulations ratio is {100.*ratio:.0f}%, which is "
                    "higher than the recommended limit of "
                    f"{100.*_SAMPLES_SIMULATIONS_RATIO:.0f}%.",
                    UserWarning,
                )
        return self


def _job_resources_from_size_and_inference(
    job_size: JobSize,
    inference: Literal["emcee", "r"],
    nodes: PositiveInt | None = None,
    cpus: PositiveInt | None = None,
    memory: PositiveInt | None = None,
) -> JobResources:
    """
    Default job resources from a job size and inference method.

    This function is meant to be used by CLI scripts that work with batch environments
    to submit inference/calibration jobs. This particular function is meant meant to be
    a temporary solution to a method that GH-402/GH-432 should implement.

    Args:
        job_size: The job size to infer resources from.
        inference: The inference method being used.
        nodes: The user provided number of nodes to use or `None` to infer from the
            job size and inference method.
        cpus: The user provided number of CPUs to use or `None` to infer from the job
            size and inference method.
        memory: The user provided amount of memory to use or `None` to infer from the
            job size and inference method.

    Returns:
        The inferred job resources.
    """
    if inference == "emcee":
        if nodes is not None and nodes != 1:
            warnings.warn(
                f"EMCEE inference only supports 1 node given {nodes}, overriding."
            )
        return JobResources(
            nodes=1,
            cpus=2 * job_size.chains if cpus is None else cpus,
            memory=(
                2 * 1024 * job_size.simulations_per_chain if memory is None else memory
            ),
        )
    return JobResources(
        nodes=job_size.chains if nodes is None else nodes,
        cpus=2 if cpus is None else cpus,
        memory=2 * 1024 if memory is None else memory,
    )


def _create_inference_command(
    inference: Literal["emcee", "r"], job_size: JobSize, **kwargs
) -> str:
    """
    Create an inference command for a job.

    Args:
        inference: The inference method to use.
        job_size: The job size to infer resources from.
        kwargs: Additional keyword arguments to pass to the template to generate the
            command.

    Returns:
        The inference command.
    """
    template_data = {
        **{"log_output": "/dev/stdout"},
        **job_size.model_dump(),
        **kwargs,
    }
    template = _jinja_environment.get_template(f"{inference}_inference_command.bash.j2")
    return template.render(template_data)


def _inference_is_array_capable(inference: Literal["emcee", "r"]) -> bool:
    """
    Determine if an inference method is capable of running in an array.

    Args:
        inference: The inference method to check.

    Returns:
        Whether the inference method is capable of running in an array.
    """
    return inference == "r"


class JobSubmission(subprocess.CompletedProcess):
    """
    Job submission result.

    This class extends the `subprocess.CompletedProcess` class to include a job ID which
    corresponds to the job submission. This is useful for tracking and managing jobs
    after submission and dependent on the context of the batch system.

    Attributes:
        job_id: The job ID of the submitted job.
        args: The command line arguments of the job submission.
        returncode: The return code of the job submission.
        stdout: The standard output of the job submission.
        stderr: The standard error of the job submission.

    See Also:
        [`subprocess.CompletedProcess`](https://docs.python.org/3/library/subprocess.html#subprocess.CompletedProcess)
    """

    def __init__(
        self,
        job_id: int | None,
        args: Any,
        returncode: int,
        stdout: bytes | str | None = None,
        stderr: bytes | str | None = None,
    ) -> None:
        """
        Create a job submission result.

        Args:
            job_id: The job ID of the submitted job.
            args: The command line arguments of the job submission.
            returncode: The return code of the job submission.
            stdout: The standard output of the job submission.
            stderr: The standard error of the job submission.

        Returns:
            None
        """
        super().__init__(args, returncode, stdout, stderr)
        self.job_id = job_id

    @classmethod
    def from_completed_process(
        cls, job_id: int | None, completed_process: subprocess.CompletedProcess
    ) -> Self:
        """
        Create a job submission result from a completed process.

        This convenience method creates a job submission result from a completed process
        since most job submissions typically are done with `subprocess.run`.

        Args:
            job_id: The job ID of the submitted job.
            completed_process: The completed process to create a job submission from.

        Returns:
            The job submission result.
        """
        return cls(
            job_id=job_id,
            args=completed_process.args,
            returncode=completed_process.returncode,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
        )


class JobResult(BaseModel):
    """
    Representation of a job result.

    Attributes:
        status: The status of the job.
        returncode: The return code of the job.
        wall_time: The wall time of the job.
        memory_efficiency: The memory efficiency of the job.
    """

    status: Literal["pending", "running", "completed", "failed"]
    returncode: int | None = None
    wall_time: timedelta | None = None
    memory_efficiency: (
        Annotated[float, Field(json_schema_extra={"minimum": 0.0, "maximum": 1.0})] | None
    ) = None

    def __eq__(self, other: Any) -> bool:
        """
        Check if two job results are equal.

        Custom implementation of the equality operator to compare two job results to
        make unit testing simpler by doing relative comparisons of the
        `memory_efficiency` attributes.

        Args:
            other: The other job result to compare.

        Returns:
            Whether the two job results are equal.
        """
        if not isinstance(other, JobResult):
            return False
        return (
            self.status == other.status
            and self.returncode == other.returncode
            and self.wall_time == other.wall_time
            and (
                math.isclose(self.memory_efficiency, other.memory_efficiency)
                if (
                    self.memory_efficiency is not None
                    and other.memory_efficiency is not None
                )
                else self.memory_efficiency == other.memory_efficiency
            )
        )


@overload
def _submit_via_subprocess(
    exec: Path,
    coerce_exec: bool,
    exec_method: Literal["run", "popen"],
    options: dict[str, str | Iterable[str]] | None,
    args: Iterable[str] | None,
    job_id_callback: (
        Callable[[subprocess.CompletedProcess | subprocess.Popen], int | None] | None
    ),
    logger: Logger | None,
    dry_run: Literal[True],
) -> None: ...


@overload
def _submit_via_subprocess(
    exec: Path,
    coerce_exec: bool,
    exec_method: Literal["run", "popen"],
    options: dict[str, str | Iterable[str]] | None,
    args: Iterable[str] | None,
    job_id_callback: (
        Callable[[subprocess.CompletedProcess | subprocess.Popen], int | None] | None
    ),
    logger: Logger | None,
    dry_run: Literal[False],
) -> JobSubmission: ...


def _submit_via_subprocess(
    exec: Path,
    coerce_exec: bool,
    exec_method: Literal["run", "popen"],
    options: dict[str, str | Iterable[str]] | None,
    args: Iterable[str] | None,
    job_id_callback: (
        Callable[[subprocess.CompletedProcess | subprocess.Popen], int | None] | None
    ),
    logger: Logger | None,
    dry_run: bool,
) -> JobSubmission | None:
    """
    Submit a job via a subprocess.

    Args:
        exec: The path to the command to execute.
        coerce_exec: Whether to make the command executable if it is not.
        exec_method: The method to use to execute the command, if 'run' then this
            function will use `subprocess.run` and if 'popen' it will use
            `subprocess.Popen`.
        options: Additional options to pass to the command if any.
        args: Additional arguments to pass to the command if any.
        job_id_callback: A callback to extract the job ID from the executed command.
        logger: The logger to use for logging.
        dry_run: Whether to perform a dry run of the submission.

    Returns:
        A job submission result or `None` if a dry run.
    """
    if logger is not None:
        logger.debug("Using script '%s' for local execution", exec.absolute())

    if not exec.exists() or not exec.is_file():
        raise ValueError(
            f"The executable '{exec.absolute()}' either does not exist or is not a file."
        )
    if coerce_exec and not bool((current_perms := exec.stat().st_mode) & S_IXUSR):
        if logger is not None:
            logger.warning(
                "The file '%s' is not executable, making it executable.", exec.absolute()
            )
        new_perms = current_perms | S_IXUSR
        exec.chmod(new_perms)

    cmd_args = [str(exec.absolute())] + _format_cli_options(options)
    if args is not None:
        cmd_args.extend(args)

    if dry_run:
        if logger is not None:
            logger.info(
                "If not dry run would have executed script with: %s",
                " ".join(cmd_args),
            )
        return None

    if logger is not None:
        logger.info("Executing script with: %s", " ".join(cmd_args))
    if exec_method == "popen":
        process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait(timeout=5 * 60)
        stdout, stderr = [comm.decode().strip() for comm in process.communicate()]
    else:
        process = subprocess.run(cmd_args, text=True, capture_output=True)
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

    if logger is not None:
        if process.returncode != 0:
            logger.critical(
                "Received non-zero exit code, %u, from executed script.",
                process.returncode,
            )
            if stdout:
                logger.info("Captured stdout from executed script: %s", stdout)
            else:
                logger.warning("No stdout captured from executed script.")
            if stderr:
                logger.error("Captured stderr from executed script: %s", stderr)
            else:
                logger.warning("No stderr captured from executed script.")
        else:
            if stdout:
                logger.debug("Captured stdout from executed script: %s", stdout)
            if stderr:
                logger.error("Captured stderr from executed script: %s", stderr)

    job_id = None if job_id_callback is None else job_id_callback(process)
    if logger is not None and job_id is not None:
        logger.info("Extracted job ID of: %s", str(job_id))

    if exec_method == "popen":
        return JobSubmission(
            job_id=job_id,
            args=process.args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return JobSubmission.from_completed_process(job_id, process)


class BatchSystem(ABC):
    """
    An abstract base class for batch systems.

    This class contains logic for interacting with batch systems, such as formatting
    job resources and time limits.

    Attributes:
        name: The name of the batch system. Must be unique when registered.
        needs_cluster: Whether the batch system requires a cluster to run jobs, defaults
            to `False`.
        estimatible: Whether the batch system can estimate the resources required for a
            job, defaults to `False`. If `True` then the batch system should implement
            the `status` method.

    Examples:
        >>> from datetime import timedelta
        >>> from gempyor.batch import BatchSystem, JobResources, JobSize
        >>> class MyCustomBatchSystem(BatchSystem):
        ...     name = "my_custom"
        ...     def submit(self, script, options, verbosity, dry_run):
        ...             return None
        >>> batch_system = MyCustomBatchSystem()
        >>> resources = JobResources(nodes=1, cpus=2, memory=1024)
        >>> batch_system.format_nodes(resources)
        '1'
        >>> batch_system.format_cpus(resources)
        '2'
        >>> batch_system.format_memory(resources)
        '1024'
        >>> time_limit = timedelta(days=1, hours=2, minutes=34, seconds=56)
        >>> batch_system.format_time_limit(time_limit)
        '1595'
        >>> batch_system.size_from_jobs_simulations_blocks(2, 10, None, 5)
        JobSize(blocks=2, chains=10, samples=None, simulations=5)
    """

    needs_cluster: bool = False
    estimatible: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @overload
    @abstractmethod
    def submit(
        self,
        script: Path,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: Literal[True] = ...,
    ) -> None: ...

    @overload
    @abstractmethod
    def submit(
        self,
        script: Path,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: Literal[False] = ...,
    ) -> JobSubmission: ...

    @abstractmethod
    def submit(
        self,
        script: Path,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: bool = False,
    ) -> JobSubmission | None:
        """
        Submit a job to the batch system.

        Args:
            script: The path to the script to submit.
            options: Additional options to pass to the batch system, if applicable.
            verbosity: The verbosity level of the submission.
            dry_run: Whether to perform a dry run of the submission, if applicable.

        Returns:
            The job submission result or `None` if a dry run.

        Notes:
            Batch systems which implement this method and do not support dry runs should
            raise a `NotImplementedError` when `dry_run` is `True`.
        """
        raise NotImplementedError

    def submit_command(
        self,
        command: str,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> JobSubmission | None:
        """
        Submit a command to the batch system.

        Args:
            command: The command to submit.
            options: Additional options to pass to the batch system, if applicable.
            verbosity: The verbosity level of the submission.
            dry_run: Whether to perform a dry run of the submission, if applicable.
            **kwargs: Additional keyword arguments to be used by subclasses.

        Returns:
            The job submission result or `None` if a dry run.

        See Also:
            The `submit` method.
        """
        logger = get_script_logger(__name__, verbosity) if verbosity is not None else None
        if logger is not None and platform.system() == "Windows":
            logger.critical(
                "Local batch system does not support command submissions on windows. "
                "If you have experience with scripting on windows and would like to "
                "contribute, please consider opening a pull request."
            )
        with NamedTemporaryFile(mode="w") as temp:
            temp_script = Path(temp.name).absolute()
            temp_script.write_text(
                _jinja_environment.get_template("submit_command.bash.j2").render(
                    {**kwargs, **{"command": command}}
                )
            )
            if dry_run:
                dest = Path.cwd() / temp_script
                shutil.copy(temp_script, Path.cwd() / dest.name)
                if logger is not None:
                    logger.info(
                        "Since dry run copying script to '%s' for inspection.", dest
                    )
            return self.submit(temp_script, options, verbosity, dry_run)

    def status(self, submission: JobSubmission) -> JobResult | None:
        """
        Get the status of a job submission.

        Args:
            submission: The job submission to get the status of.

        Returns:
            The status of the job submission or `None` if the status could not be
            determined or the batch system is not estimatible.
        """
        return None

    def format_nodes(self, job_resources: JobResources) -> str:
        """
        Format the number of nodes for a job.

        The default implementation returns the number of nodes as a string.

        Args:
            job_resources: The job resources to format.

        Returns:
            The formatted number of nodes.
        """
        return str(job_resources.nodes)

    def format_cpus(self, job_resources: JobResources) -> str:
        """
        Format the number of CPUs for a job.

        The default implementation returns the number of CPUs as a string.

        Args:
            job_resources: The job resources to format.

        Returns:
            The formatted number of CPUs.
        """
        return str(job_resources.cpus)

    def format_memory(self, job_resources: JobResources) -> str:
        """
        Format the amount of memory for a job.

        The default implementation returns the amount of memory as a string.

        Args:
            job_resources: The job resources to format.

        Returns:
            The formatted amount of memory.
        """
        return str(job_resources.memory)

    def format_time_limit(self, job_time_limit: timedelta) -> str:
        """
        Format the time limit for a job.

        The default implementation returns the time limit in minutes rounded up.

        Args:
            job_time_limit: The time limit to format.

        Returns:
            The formatted time limit.
        """
        return str(math.ceil(job_time_limit.total_seconds() / 60.0))

    def size_from_jobs_simulations_blocks(
        self,
        blocks: PositiveInt | None,
        chains: PositiveInt | None,
        samples: PositiveInt | None,
        simulations: PositiveInt | None,
    ) -> JobSize:
        """
        Infer a job size from several explicit and implicit parameters.

        Args:
            blocks: An explicit number of blocks per a job.
            chains: An explicit number of chains.
            samples: An explicit number of samples per a block.
            simulations: An explicit number of simulations per a block.

        Returns:
            A job size instance with either the explicit or inferred job sizing.

        See Also:
            `gempyor.batch.JobSize`
        """
        return JobSize(
            blocks=blocks, chains=chains, samples=samples, simulations=simulations
        )

    def options_from_config_and_cli(
        self,
        config: confuse.Configuration,
        cli_options: dict[str, Any],
        verbosity: int | None,
    ) -> dict[str, str | Iterable[str]] | None:
        """
        Generate batch system options from a configuration and CLI options.

        Args:
            config: The configuration options to use.
            cli_options: The CLI options to use.
            verbosity: The verbosity level of the submission.

        Returns:
            The batch system options.
        """
        return None


class LocalBatchSystem(BatchSystem):
    """
    Batch system for running jobs locally.

    This class is a batch system for running jobs locally. It is intended for testing
    and development purposes, not production jobs. Therefore it will override user
    inputs to limit the number of jobs to 1 and the number of blocks x simulations to
    10.

    Attributes:
        name: The name of the batch system which is 'local'.

    Examples:
        >>> import warnings
        >>> from gempyor.batch import LocalBatchSystem
        >>> batch_system = LocalBatchSystem()
        >>> batch_system.size_from_jobs_simulations_blocks(1, 1, 25, 50)
        JobSize(blocks=1, chains=1, samples=25, simulations=50)
        >>> with warnings.catch_warnings(record=True) as warns:
        ...     size = batch_system.size_from_jobs_simulations_blocks(2, 4, 50, 100)
        ...     for warn in warns:
        ...             print(warn.message)
        ...
        Local batch system only supports 1 chain but was given 4, overriding.
        Local batch system only supports 1 block but was given 2, overriding.
        Local batch system only supports 50 total simulations but was given 800, overriding.
        Local batch system only supports 25 total samples but was given 400, overriding.
        >>> size
        JobSize(blocks=1, chains=1, samples=25, simulations=50)
    """

    name = "local"

    def submit(
        self,
        script: Path,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: bool = False,
    ) -> JobSubmission | None:
        """
        Submit a job to the local batch system.

        Args:
            script: The path to the script to submit.
            options: Additional options to pass to the batch system.
            verbosity: The verbosity level of the submission.
            dry_run: Whether to perform a dry run of the submission.

        Returns:
            The job submission result where the `job_id` is the PID of the process or
            `None` if a dry run.
        """
        return _submit_via_subprocess(
            exec=script,
            coerce_exec=True,
            exec_method="popen",
            options=options,
            args=None,
            job_id_callback=lambda proc: proc.pid,
            logger=(
                get_script_logger(__name__, verbosity) if verbosity is not None else None
            ),
            dry_run=dry_run,
        )

    def size_from_jobs_simulations_blocks(
        self,
        blocks: PositiveInt | None,
        chains: PositiveInt | None,
        samples: PositiveInt | None,
        simulations: PositiveInt | None,
    ) -> JobSize:
        """
        Infer a job size from several explicit and implicit parameters.

        Args:
            blocks: An explicit number of blocks per a job.
            chains: An explicit number of chains.
            samples: An explicit number of samples per a block.
            simulations: An explicit number of simulations per a block.

        Returns:
            A job size instance with either the explicit or inferred job sizing.

        See Also:
            `gempyor.batch.JobSize`
        """
        prelim_size = JobSize(
            blocks=blocks, chains=chains, samples=samples, simulations=simulations
        )
        if prelim_size.chains is not None and prelim_size.chains != 1:
            warnings.warn(
                "Local batch system only supports 1 chain but "
                f"was given {prelim_size.chains}, overriding."
            )
        if prelim_size.blocks is not None and prelim_size.blocks != 1:
            warnings.warn(
                "Local batch system only supports 1 block but "
                f"was given {prelim_size.blocks}, overriding."
            )
        if prelim_size.total_samples is not None and prelim_size.total_samples > 25:
            warnings.warn(
                "Local batch system only supports 25 total samples but "
                f"was given {prelim_size.total_samples}, overriding."
            )
        if prelim_size.total_simulations is not None and prelim_size.total_simulations > 50:
            warnings.warn(
                "Local batch system only supports 50 total simulations but "
                f"was given {prelim_size.total_simulations}, overriding."
            )
        return JobSize(
            blocks=None if prelim_size.blocks is None else 1,
            chains=None if prelim_size.chains is None else 1,
            samples=(
                None
                if prelim_size.total_samples is None
                else min(prelim_size.total_samples, 25)
            ),
            simulations=(
                None
                if prelim_size.total_simulations is None
                else min(prelim_size.total_simulations, 50)
            ),
        )


def _slurm_submit_command_cleanup(sbatch_script: Path, cwd: Path) -> None:
    """
    Clean up the sbatch script on exit.

    Internal helper to copy an sbatch submission script to the current working directory
    and remove the original script on exit.

    Args:
        sbatch_script: The path to the sbatch script.
        cwd: The current working directory.

    Returns:
        None
    """
    shutil.copy2(sbatch_script, cwd / sbatch_script.name)
    sbatch_script.unlink(missing_ok=True)


class SlurmBatchSystem(BatchSystem):
    """
    Batch system for running jobs on a Slurm HPC cluster.

    This class is a batch system for running jobs on a Slurm HPC cluster. It provides
    formatting overrides that are specific to Slurm's submission requirements.

    Attributes:
        name: The name of the batch system which is 'slurm'.

    Examples:
        >>> from datetime import timedelta
        >>> from gempyor.batch import SlurmBatchSystem, JobResources
        >>> batch_system = SlurmBatchSystem()
        >>> resources = JobResources(nodes=2, cpus=4, memory=8*1024)
        >>> batch_system.format_memory(resources)
        '8192MB'
        >>> time_limit = timedelta(days=1, hours=2, minutes=34, seconds=56)
        >>> batch_system.format_time_limit(time_limit)
        '26:34:56'
    """

    _sbatch_regex = re.compile(r"submitted batch job (\d+)", flags=re.IGNORECASE)
    _seff_state_regex = re.compile(
        r"state:\s+(pending|running|completed|failed)(\s+\(exit\s+code\s+([0-9]+)\))?",
        flags=re.IGNORECASE,
    )
    _seff_wall_time_regex = re.compile(
        r"job\s+wall\-clock\s+time:\s+([0-9]+):([0-9]+):([0-9]+)", flags=re.IGNORECASE
    )
    _seff_memory_efficiency_regex = re.compile(
        r"memory\s+efficiency:\s([0-9]+\.[0-9]+)%", flags=re.IGNORECASE
    )
    _seff_cpu_efficiency_regex = re.compile(
        r"cpu\s+efficiency:\s([0-9]+\.[0-9]+)%", flags=re.IGNORECASE
    )

    name = "slurm"
    needs_cluster = True
    estimatible = True

    def submit(
        self,
        script: Path,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: bool = False,
    ) -> JobSubmission | None:
        """
        Submit a job to the slurm batch system.

        Args:
            script: The path to the script to submit.
            options: Additional options to pass to the batch system.
            verbosity: The verbosity level of the submission.
            dry_run: Whether to perform a dry run of the submission.

        Returns:
            The job submission result where the `job_id` is the slurm ID of the job or
            `None` if a dry run.
        """
        return _submit_via_subprocess(
            exec=Path(_shutil_which("sbatch")),
            coerce_exec=False,
            exec_method="run",
            options=options,
            args=[str(script.absolute())],
            job_id_callback=lambda proc: int(
                self._sbatch_regex.match(proc.stdout).group(1)
            ),
            logger=(
                get_script_logger(__name__, verbosity) if verbosity is not None else None
            ),
            dry_run=dry_run,
        )

    def submit_command(
        self,
        command: str,
        options: dict[str, str | Iterable[str]] | None = None,
        verbosity: int | None = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> JobSubmission | None:
        """
        Submit a command to the slurm batch system.

        Args:
            command: The command to submit.
            options: Additional options to pass to the batch system, if applicable.
            verbosity: The verbosity level of the submission.
            dry_run: Whether to perform a dry run of the submission, if applicable.
            **kwargs: Additional keyword arguments used to generate the sbatch
                submission script.

        Returns:
            The job submission result or `None` if a dry run.

        Notes:
            If `dry_run` is `True` then the sbatch submission script will be copied to
            the current working directory before program end.
        """
        logger = get_script_logger(__name__, verbosity) if verbosity is not None else None
        options = options or {}
        with NamedTemporaryFile(
            mode="w",
            suffix=".sbatch",
            prefix=None if (job_name := options.get("job_name")) is None else job_name,
        ) as temp_script:
            sbatch_script = Path(temp_script.name).absolute()
            sbatch_script.write_text(
                _jinja_environment.get_template("sbatch_submit_command.bash.j2").render(
                    {**kwargs, **{"command": command}}
                )
            )
            if logger is not None:
                logger.info("Using sbatch script '%s' for submission", sbatch_script)
            if dry_run:
                dest = Path.cwd() / sbatch_script.name
                shutil.copy2(sbatch_script, dest)
                if logger is not None:
                    logger.info("Sbatch script copied to '%s' for inspection", dest)
            return self.submit(
                sbatch_script,
                options,
                verbosity,
                dry_run,
            )

    def status(self, submission: JobSubmission) -> JobResult:
        """
        Get the status of a job submission via `seff`.

        Args:
            submission: The job submission to get the status of.

        Returns:
            The status of the job submission.
        """
        seff = _shutil_which("seff")
        seff_proc = subprocess.run(
            [seff, str(submission.job_id)], text=True, capture_output=True, check=True
        )
        lines = seff_proc.stdout.splitlines()
        job_result_kwargs = {}
        for line in lines:
            line = line.strip().lower()
            if (match := self._seff_state_regex.match(line)) is not None:
                state, _, returncode = match.groups()
                job_result_kwargs["status"] = state.lower()
                job_result_kwargs["returncode"] = int(returncode) if returncode else None
            elif (match := self._seff_wall_time_regex.match(line)) is not None:
                hours, minutes, seconds = match.groups()
                job_result_kwargs["wall_time"] = timedelta(
                    hours=int(hours), minutes=int(minutes), seconds=int(seconds)
                )
            elif (match := self._seff_memory_efficiency_regex.match(line)) is not None:
                job_result_kwargs["memory_efficiency"] = 0.01 * float(match.group(1))
            elif (match := self._seff_cpu_efficiency_regex.match(line)) is not None:
                job_result_kwargs["cpu_efficiency"] = 0.01 * float(match.group(1))
        return JobResult(**job_result_kwargs)

    def format_memory(self, job_resources: JobResources) -> str:
        return f"{job_resources.memory}MB"

    def format_time_limit(self, job_time_limit: timedelta) -> str:
        total_seconds = job_time_limit.total_seconds()
        hours = math.floor(total_seconds / (60.0 * 60.0))
        minutes = math.floor((total_seconds - (60.0 * 60.0 * hours)) / 60.0)
        seconds = math.ceil(total_seconds - (60.0 * minutes) - (60.0 * 60.0 * hours))
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    def options_from_config_and_cli(
        self,
        config: confuse.Configuration,
        cli_options: dict[str, Any],
        verbosity: int | None,
    ) -> dict[str, str | Iterable[str]] | None:
        """
        Generate batch system options from a configuration and CLI options.

        Args:
            config: The configuration options to use.
            cli_options: The CLI options to use.
            verbosity: The verbosity level of the submission.

        Returns:
            The batch system options.
        """
        logger = get_script_logger(__name__, verbosity) if verbosity is not None else None
        options = {}
        if (partition := cli_options.get("extra", {}).get("partition")) is not None:
            options["partition"] = partition
        if (email := cli_options.get("extra", {}).get("email")) is not None:
            options["mail-user"] = email
            options["mail-type"] = "ALL"
        if logger is not None:
            logger.debug("Generated options: %s", options)
        return options


def register_batch_system(batch_system: BatchSystem) -> None:
    """
    Register a batch system with gempyor.

    Add a batch system to the list of registered batch systems with gempyor. This
    function acts as a complement to `get_batch_system` and is intended to be used by
    users who wish to add their own batch system to gempyor.

    Args:
        batch_system: The batch system to register.

    Returns:
        None

    Raises:
        ValueError: If the batch system is already registered, checks the `name`
            attribute to determine this.

    Examples:
        >>> from gempyor.batch import BatchSystem, register_batch_system
        >>> class CustomBatchSystem(BatchSystem):
        ...     name = "custom"
        >>> register_batch_system(CustomBatchSystem()) is None
        True
        >>> try:
        ...     register_batch_system(CustomBatchSystem()) is None
        ... except Exception as e:
        ...     print(e)
        ...
        Batch system 'custom' already registered.
    """
    if any(batch_system.name == bs.name for bs in _batch_systems):
        raise ValueError(f"Batch system '{batch_system.name}' already registered.")
    _batch_systems.append(batch_system)


@overload
def get_batch_system(name: str, raise_on_missing: Literal[True] = ...) -> BatchSystem: ...


def get_batch_system(name: str, raise_on_missing: bool = True) -> BatchSystem | None:
    """
    Get a registered batch system by name.

    Args:
        name: The name of the batch system to get.
        raise_on_missing: Whether to raise an error if the batch system is not found.

    Returns:
        The batch system with the given name or `None` if not found.

    Raises:
        ValueError: If the batch system is not found and `raise_on_missing` is `True`.

    Examples:
        >>> from gempyor.batch import get_batch_system
        >>> get_batch_system("local")
        <gempyor.batch.LocalBatchSystem object at 0x1629f9b10>
        >>> get_batch_system("slurm")
        <gempyor.batch.SlurmBatchSystem object at 0x1629f9ad0>
        >>> try:
        ...     get_batch_system("does not exist")
        ... except Exception as e:
        ...     print(e)
        ...
        Batch system 'does not exist' not found in registered batch systems.
        >>> get_batch_system("does not exist", raise_on_missing=False) is None
        True
    """
    batch_system = next((bs for bs in _batch_systems if bs.name == name), None)
    if batch_system is None and raise_on_missing:
        raise ValueError(f"Batch system '{name}' not found in registered batch systems.")
    return batch_system


def _reset_batch_systems() -> None:
    """
    Reset the batch systems to the default state.

    This function is intended for testing purposes.

    Returns:
        None
    """
    globals()["_batch_systems"] = []
    for batch_system in (SlurmBatchSystem(), LocalBatchSystem()):
        register_batch_system(batch_system)


def write_manifest(
    job_name: str,
    flepi_path: Path,
    project_path: Path,
    destination: Path | None = None,
    **additional_meta: Any,
) -> Path:
    """
    Write job metadata to a manifest file.

    This function produces a manifest metadata file for a batch run. By default the
    json generated by this function will contain:
    * 'cmd': The command line arguments provided to the CLI script invoked.
    * 'job_name': A human readable unique job name.
    * 'data_sha': The git commit of the project git repository, called 'data' for
        legacy reasons.
    * 'flepimop_sha': The git commit of the flepiMoP git repository.
    Further data can be provided via `**additional_meta`, but these values are
    overridden by the defaults described above.

    Args:
        job_name: A user specified or generated from user specified values unique name
            for the job.
        flepi_path: The path to the flepiMoP git repository being used.
        project_path: The path to the project git repository being used.
        destination: Either a path to where the json file should be written or `None` to
            write the json file to 'manifest.json' in the current working directory.
        additional_meta: User specified additional fields added to the manifest json.
            Values with the name 'cmd', 'job_name', 'data_sha', or 'flepimop_sha' will
            be overridden by the default behavior. Must be a json encodable type.

    Returns:
        The path to the written json file.

    Examples:
        >>> import os
        >>> from pathlib import Path
        >>> flepi_path = Path(os.environ["FLEPI_PATH"])
        >>> project_path = flepi_path / "examples" / "tutorials"
        >>> manifest = write_manifest("Foobar", flepi_path, project_path)
        >>> manifest.name
        'manifest.json'
        >>> print(manifest.read_text())
        {
            "cmd": "",
            "job_name": "Foobar",
            "data_sha": "59fe36d13fe34b6c1fb5c92bf8c53b83bd3ba593",
            "flepimop_sha": "2bdfbc74e69bdd0243ef8340dda238f5504f1ad9"
        }
    """
    flepimop_sha = _git_head(flepi_path)
    data_sha = _git_head(project_path)

    manifest = {
        "cmd": " ".join(sys.argv),
        "job_name": job_name,
        "data_sha": data_sha,
        "flepimop_sha": flepimop_sha,
    }
    if additional_meta:
        manifest = {**additional_meta, **manifest}

    destination = Path("manifest.json").absolute() if destination is None else destination
    with destination.open(mode="w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    return destination


def _job_name(name: str | None, timestamp: datetime | None) -> str:
    """
    Generate a unique human readable job name.
    Args:
        name: The config name used as a prefix or `None` for no prefix.
        timestamp: The timestamp used to make the job name unique or `None` to use the
            current UTC timestamp.
    Returns:
        A job name that is unique and intended for use when submitting to slurm.
    Raises:
        ValueError: If `name` does not start with a letter and contains characters other
            than the alphabet, numbers, underscores or dashes.
    Examples:
        >>> from gempyor.batch import _job_name
        >>> _job_name(None, None)
        '20241105T153818'
        >>> _job_name("foobar", None)
        'foobar-20241105T153831'
        >>> from datetime import datetime, timezone
        >>> _job_name(None, datetime(2024, 1, 1, tzinfo=timezone.utc))
        '20240101T000000'
    """
    timestamp = datetime.now(timezone.utc) if timestamp is None else timestamp
    timestamp = timestamp.strftime("%Y%m%dT%H%M%S")
    if name is not None and not _JOB_NAME_REGEX.match(name):
        raise ValueError(f"The given `name`, '{name}', is not a valid safe name.")
    return f"{name}-{timestamp}" if name else timestamp


def _resolve_batch_system_name(name: str | None, local: bool, slurm: bool) -> str | None:
    """
    Resolve the batch system name from the given arguments.

    Args:
        name: The name of the batch system or `None` to infer from `local` and `slurm`.
        local: A flag to use the local batch system.
        slurm: A flag to use the slurm batch system.

    Returns:
        The resolved batch system name lowercased or `None` if no batch system was
        resolved.

    Raises:
        ValueError: If more than one batch system is indicated via boolean flags.
        ValueError: If the given name conflicts with the boolean flags.

    Examples:
        >>> from gempyor.batch import _resolve_batch_system_name
        >>> _resolve_batch_system_name("abc", False, False)
        'abc'
        >>> _resolve_batch_system_name("SLURM", False, False)
        'slurm'
        >>> _resolve_batch_system_name("local", True, False)
        'local'
        >>> _resolve_batch_system_name(None, True, False)
        'local'
        >>> _resolve_batch_system_name(None, False, True)
        'slurm'
        >>> try:
        ...     _resolve_batch_system_name(None, True, True)
        ... except Exception as e:
        ...     print(e)
        There were 2 boolean flags given, expected either 0 or 1.
        >>> try:
        ...     _resolve_batch_system_name("slurm", True, False)
        ... except Exception as e:
        ...     print(e)
        Conflicting batch systems given. The batch system name is 'slurm' and the flags indicate 'local'.
        >>> _resolve_batch_system_name(None, False, False) is None
        True
    """
    name = name.lower() if name is not None else name
    if (boolean_flags := sum((local, slurm))) > 1:
        raise ValueError(
            f"There were {boolean_flags} boolean flags given, expected either 0 or 1."
        )
    if name is not None:
        for flag, flag_name in zip((local, slurm), ("local", "slurm")):
            if flag and name != flag_name:
                raise ValueError(
                    "Conflicting batch systems given. The batch system name "
                    f"is '{name}' and the flags indicate '{flag_name}'."
                )
    if name is None:
        if local:
            name = "local"
        elif slurm:
            name = "slurm"
    return name


def _parse_extra_options(extra: Iterable[str] | None) -> dict[str, str]:
    """
    Parse `--extra` options into a dictionary.

    Args:
        extra: An iterable of extra options to parse if given.

    Returns:
        A dictionary of the parsed extra options.

    Examples:
        >>> from gempyor.batch import _parse_extra_options
        >>> _parse_extra_options(["abc=def", "ghi=jkl"])
        {'abc': 'def', 'ghi': 'jkl'}
        >>> _parse_extra_options([
        ...     "email=bob@example.com",
        ...     "partition=special-cluster",
        ...     "slack=my-alerts-channel",
        ... ])
        {'email': 'bob@example.com', 'partition': 'special-cluster', 'slack': 'my-alerts-channel'}
        >>> _parse_extra_options(None)
        {}
    """
    if extra is None:
        return {}
    return {
        k: v for k, v in (opt.split("=", 1) if "=" in opt else [opt, ""] for opt in extra)
    }


def _submit_scenario_job(
    name: str,
    job_name: str,
    inference: Literal["emcee", "r"],
    job_size: JobSize,
    batch_system: BatchSystem,
    outcome_modifiers_scenario: str,
    seir_modifiers_scenario: str,
    options: dict[str, str | Iterable[str]] | None,
    template_data: dict[str, Any],
    verbosity: int,
    dry_run: bool,
) -> None:
    """
    Submit a job for a scenario.

    Args:
        outcome_modifiers_scenario: The outcome modifiers scenario to use.
        seir_modifiers_scenario: The seir modifiers scenario to use.
        name: The name of the config file used as a prefix for the job name.
        batch_system: The batch system to submit the job to.
        inference_method: The inference method being used.
        config_out: The path to the config file to use.
        job_name: The name of the job to submit.
        job_size: The size of the job to submit.
        job_time_limit: The time limit of the job to submit.
        job_resources: The resources required for the job to submit.
        cluster: The cluster information to use for submitting the job.
        kwargs: Additional options provided to the submit job CLI as keyword arguments.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit/run a job.
        now: The current UTC timestamp.
    """
    # Get logger
    if verbosity is not None:
        logger = get_script_logger(__name__, verbosity)
        if outcome_modifiers_scenario == "None":
            logger.warning(
                "The outcome modifiers scenario is `None`, may lead to "
                "unintended consequences in output file/directory names."
            )
        if seir_modifiers_scenario == "None":
            logger.warning(
                "The seir modifiers scenario is `None`, may lead to "
                "unintended consequences in output file/directory names."
            )

    # Modify the job for the given scenario info
    job_name += f"_{seir_modifiers_scenario}_{outcome_modifiers_scenario}"
    prefix = f"{name}_{seir_modifiers_scenario}_{outcome_modifiers_scenario}"
    if verbosity is not None:
        logger.info(
            "Preparing a job for outcome and seir modifiers scenarios "
            "'%s' and '%s', respectively, with job name '%s'.",
            outcome_modifiers_scenario,
            seir_modifiers_scenario,
            job_name,
        )

    # Template data
    template_data = {
        **template_data,
        **{
            "prefix": prefix,
            "outcome_modifiers_scenario": outcome_modifiers_scenario,
            "seir_modifiers_scenario": seir_modifiers_scenario,
            "job_comment": (
                f"{name} submitted by {template_data.get('user', 'unknown')} at "
                f"{template_data.get('now', 'unknown')} with outcome and seir modifiers "
                f"scenarios '{outcome_modifiers_scenario}' and "
                f"'{seir_modifiers_scenario}', respectively."
            ),
            "job_name": job_name,
        },
    }

    # Get inference command
    inference_command = _create_inference_command(
        inference,
        job_size,
        **{k: v for k, v in template_data.items() if k not in {"inference", "job_size"}},
    )

    # Submit
    batch_system.submit_command(
        inference_command,
        options,
        verbosity,
        dry_run,
        **{
            k: v
            for k, v in template_data.items()
            if k not in {"command", "options", "verbosity", "dry_run"}
        },
    )


@cli.command(
    name="batch-calibrate",
    params=[config_files_argument]
    + list(config_file_options.values())
    + [
        click.Option(
            param_decls=["--flepi-path", "flepi_path"],
            envvar="FLEPI_PATH",
            type=click.Path(exists=True, path_type=Path),
            required=True,
            help="Path to the flepiMoP directory being used.",
        ),
        click.Option(
            param_decls=["--project-path", "project_path"],
            envvar="PROJECT_PATH",
            type=click.Path(exists=True, path_type=Path),
            required=True,
            help="Path to the project directory being used.",
        ),
        click.Option(
            param_decls=["--conda-env", "conda_env"],
            type=str,
            default="flepimop-env",
            help="The conda environment to use for the job.",
        ),
        click.Option(
            param_decls=["--blocks", "blocks"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of sequential blocks to run per a chain.",
        ),
        click.Option(
            param_decls=["--chains", "chains"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of chains or walkers, depending on inference method, to run.",
        ),
        click.Option(
            param_decls=["--samples", "samples"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of samples per a block.",
        ),
        click.Option(
            param_decls=["--simulations", "simulations"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of simulations per a block.",
        ),
        click.Option(
            param_decls=["--time-limit", "time_limit"],
            type=DurationParamType(True, "minutes"),
            default="1hr",
            help=(
                "The time limit for the job. If units "
                "are not specified, minutes are assumed."
            ),
        ),
        click.Option(
            param_decls=["--batch-system", "batch_system"],
            default=None,
            type=str,
            help="The name of the batch system being used.",
        ),
        click.Option(
            param_decls=["--local", "local"],
            default=False,
            is_flag=True,
            help=(
                "Flag to use the local batch system. "
                "Equivalent to `--batch-system local`."
            ),
        ),
        click.Option(
            param_decls=["--slurm", "slurm"],
            default=False,
            is_flag=True,
            help=(
                "Flag to use the slurm batch system. "
                "Equivalent to `--batch-system slurm`."
            ),
        ),
        click.Option(
            param_decls=["--cluster", "cluster"],
            default=None,
            type=str,
            help=(
                "The name of the cluster being used, "
                "only needed if cluster info is required."
            ),
        ),
        click.Option(
            param_decls=["--nodes", "nodes"],
            type=click.IntRange(min=1),
            default=None,
            help="Override for the number of nodes to use.",
        ),
        click.Option(
            param_decls=["--cpus", "cpus"],
            type=click.IntRange(min=1),
            default=None,
            help="Override for the number of CPUs per node to use.",
        ),
        click.Option(
            param_decls=["--memory", "memory"],
            type=MemoryParamType(True, "mb", True),
            default=None,
            help="Override for the amount of memory per node to use in MB.",
        ),
        click.Option(
            param_decls=["--skip-manifest", "skip_manifest"],
            type=bool,
            default=False,
            is_flag=True,
            help="Flag to skip writing a manifest file, useful in dry runs.",
        ),
        click.Option(
            param_decls=["--skip-checkout", "skip_checkout"],
            type=bool,
            default=False,
            is_flag=True,
            help=(
                "Flag to skip checking out a new branch in "
                "the git repository, useful in dry runs."
            ),
        ),
        click.Option(
            param_decls=["--debug", "debug"],
            type=bool,
            default=False,
            is_flag=True,
            help="Flag to enable debugging in batch submission scripts.",
        ),
        click.Option(
            param_decls=["--extra", "extra"],
            type=str,
            default=None,
            multiple=True,
            help=(
                "Extra options to pass to the batch system. Please consult "
                "the batch system documentation for valid options."
            ),
        ),
    ]
    + list(verbosity_options.values()),
)
@click.pass_context
def _click_batch_calibrate(ctx: click.Context = mock_context, **kwargs: Any) -> None:
    """
    Submit a calibration job to a batch system.
    
    This job makes it straightforward to submit a calibration job to a batch system. The
    job will be submitted with the given configuration file and additional options. The
    general steps this tool follows are:
    
    \b
    1) Generate a unique job name from the configuration and timestamp,
    2) Determine the outcome/SEIR modifier scenarios to use,
    3) Determine the batch system to use and required job size/resources/time limit,
    4) Write a 'manifest.json' with job metadata, write the config used to a file, and
       checkout a new branch in the project git repository,
    5) Loop over the outcome/SEIR modifier scenarios and submit a job for each scenario.
    
    To get a better understanding of this tool you can use the `--dry-run` flag which
    will complete all of steps described above except for submitting the jobs. Or if you
    would like to test run the batch scripts without submitting to slurm or other batch 
    systems you can use the `--local` flag which will run the "batch" job locally (only 
    use for small test jobs).
    
    Here is an example of how to use this tool with the `examples/tutorials/` directory:
    
    \b
    ```bash
    $ flepimop batch-calibrate \\
        # The paths and conda environment to use (assuming $FLEPI_PATH is set)
        --flepi-path $FLEPI_PATH \\
        --project-path $FLEPI_PATH/examples/tutorials \\
        --conda-env flepimop-env \\ 
        # The size of the job to run
        --blocks 1 \\
        --chains 50 \\
        --samples 100 \\
        --simulations 500 \\
        # The time limit for the job
        --time-limit 8hr \\
        # The batch system to use, equivalent to `--batch-system slurm`
        --slurm \\
        # Resource options
        --nodes 50 \\
        --cpus 2 \\
        --memory 4GB \\
        # Batch system specific options can be provided via `--extra`
        --extra partition=normal \\
        --extra email=bob@example.edu \\
        # Only run a dry run to see what would be submitted for the config
        --dry-run \\
        -vvv config_sample_2pop_inference.yml
    ```
    """
    # Generic setup
    now = datetime.now(timezone.utc)
    logger = get_script_logger(__name__, kwargs.get("verbosity", 0))
    log_cli_inputs(kwargs)
    cfg = parse_config_files(config, ctx, **kwargs)

    # Job name/run id
    name = cfg["name"].as_str() if cfg["name"].exists() else None
    job_name = _job_name(name, now)
    logger.info("Assigning job name of '%s'", job_name)
    if kwargs.get("run_id") is None:
        kwargs["run_id"] = run_id(now)
    logger.info("Using a run id of '%s'", kwargs.get("run_id"))

    # Parse extra options
    kwargs["extra"] = _parse_extra_options(kwargs.get("extra", None))
    logger.debug("Parsed extra options: %s", kwargs["extra"])

    # Inference method
    if not cfg["inference"].exists():
        logger.critical(
            "No inference section specified in the config "
            "file, likely missing important information."
        )
    inference_method = (
        cfg["inference"]["method"].as_str()
        if cfg["inference"].exists() and cfg["inference"]["method"].exists()
        else "r"
    ).lower()
    logger.info("Using inference method '%s'", inference_method)

    # Outcome/seir modifier scenarios
    outcome_modifiers_scenarios = (
        cfg["outcome_modifiers"]["scenarios"].as_str_seq()
        if cfg["outcome_modifiers"].exists()
        and cfg["outcome_modifiers"]["scenarios"].exists()
        else ["None"]
    )
    logger.info(
        "Using outcome modifier scenarios of '%s'", "', '".join(outcome_modifiers_scenarios)
    )
    seir_modifiers_scenarios = (
        cfg["seir_modifiers"]["scenarios"].as_str_seq()
        if cfg["seir_modifiers"].exists() and cfg["seir_modifiers"]["scenarios"].exists()
        else ["None"]
    )
    logger.info(
        "Using SEIR modifier scenarios of '%s'", "', '".join(seir_modifiers_scenarios)
    )

    # Batch system
    batch_system_name = _resolve_batch_system_name(
        kwargs.get("batch_system", None),
        kwargs.get("local", False),
        kwargs.get("slurm", False),
    )
    logger.debug("Resolved batch system name to '%s'", batch_system_name)
    batch_system = get_batch_system(batch_system_name)
    logger.info("Using batch system '%s'", batch_system.name)

    # Job size
    logger.debug(
        "User provided job size of blocks=%s, chains=%s, samples=%s, simulations=%s",
        kwargs.get("blocks"),
        kwargs.get("chains"),
        kwargs.get("samples"),
        kwargs.get("simulations"),
    )
    job_size = batch_system.size_from_jobs_simulations_blocks(
        kwargs.get("blocks"),
        kwargs.get("chains"),
        kwargs.get("samples"),
        kwargs.get("simulations"),
    )
    logger.info("Using job size of %s", job_size)

    # Job time limit
    job_time_limit = kwargs.get("time_limit")
    logger.info("Using job time limit of %s", job_time_limit)

    # Job resources
    nodes, cpus, memory = (kwargs.get(k) for k in ("nodes", "cpus", "memory"))
    logger.debug(
        "User provided job resources of nodes=%s, cpus=%s, memory=%sMB", nodes, cpus, memory
    )
    if not all((nodes, cpus, memory)):
        raise NotImplementedError(
            "Automatic resource estimation is not yet implemented "
            "and one of nodes, cpus, memory is not given."
        )
    job_resources = _job_resources_from_size_and_inference(
        job_size, inference_method, nodes=nodes, cpus=cpus, memory=memory
    )
    logger.info("Using job resources of %s", job_resources)

    # HPC cluster info
    cluster_name = kwargs.get("cluster")
    logger.debug("User provided cluster name of '%s'", cluster_name)
    cluster = (
        get_cluster_info(cluster_name)
        if (batch_system.needs_cluster or cluster_name is not None)
        else None
    )
    logger.info("Using cluster info of %s", cluster)

    # Manifest
    if not kwargs.get("skip_manifest", False):
        manifest = write_manifest(
            job_name,
            kwargs.get("flepi_path"),
            kwargs.get("project_path"),
        )
        logger.info("Wrote manifest to '%s'", manifest)
    else:
        if kwargs.get("dry_run", False):
            logger.info("Skipping manifest.")
        else:
            logger.warning("Skipping manifest in non-dry run which is not recommended.")

    # Job config
    job_config = Path(f"config_{job_name}.yml").absolute()
    job_config.write_text(_dump_formatted_yaml(cfg))
    if logger is not None:
        logger.info(
            "Dumped the job config for this batch submission to %s", job_config.absolute()
        )

    # Git checkout
    if not kwargs.get("skip_checkout", False):
        _git_checkout(kwargs.get("project_path"), f"run_{job_name}")
    else:
        if kwargs.get("dry_run", False):
            logger.info("Skipping git checkout.")
        else:
            logger.warning("Skipping git checkout in non-dry run which is not recommended.")

    # Construct template data
    general_template_data = {
        **kwargs,
        **{
            "user": getuser(),
            "now": now.strftime("%c"),
            "name": name,
            "job_name": job_name,
            "job_size": job_size.model_dump(),
            "job_time_limit": batch_system.format_time_limit(job_time_limit),
            "job_resources_nodes": batch_system.format_nodes(job_resources),
            "job_resources_cpus": batch_system.format_cpus(job_resources),
            "job_resources_memory": batch_system.format_memory(job_resources),
            "cluster": None if cluster is None else cluster.model_dump(),
            "config": job_config,
            "array_capable": _inference_is_array_capable(inference_method),
        },
    }

    # Submit jobs
    for outcome_modifiers_scenario, seir_modifiers_scenario in product(
        outcome_modifiers_scenarios, seir_modifiers_scenarios
    ):
        _submit_scenario_job(
            name,
            job_name,
            inference_method,
            job_size,
            batch_system,
            outcome_modifiers_scenario,
            seir_modifiers_scenario,
            batch_system.options_from_config_and_cli(
                cfg, kwargs, kwargs.get("verbosity", 0)
            ),
            general_template_data,
            kwargs.get("verbosity", 0),
            kwargs.get("dry_run", False),
        )


_reset_batch_systems()
