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
from datetime import timedelta
import json
from logging import Logger
import math
from pathlib import Path
import re
from stat import S_IXUSR
import subprocess
import sys
from typing import Any, Callable, Literal, overload
import warnings

from pydantic import BaseModel, PositiveInt

from .logging import get_script_logger
from .utils import _format_cli_options, _git_head, _shutil_which


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


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
        chains: The number of chains to use.
        simulations: The number of simulations to run per a block.
        blocks: The number of sequential blocks to run per a job.

    Raises:
        ValueError: If any of the attributes are less than 1.

    Examples:
        >>> from gempyor.batch import JobSize
        >>> size = JobSize(chains=10, simulations=200, blocks=5)
        >>> size
        JobSize(chains=10, simulations=200, blocks=5)
        >>> try:
        ...     JobSize(chains=10, simulations=200, blocks=0)
        ... except Exception as e:
        ...     print(e)
        1 validation error for JobSize
        blocks
        Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
            For further information visit https://errors.pydantic.dev/2.10/v/greater_than
        >>> try:
        ...     JobSize(chains=10, simulations=200.25, blocks=5)
        ... except Exception as e:
        ...     print(e)
        1 validation error for JobSize
        simulations
        Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=200.25, input_type=float]
            For further information visit https://errors.pydantic.dev/2.10/v/int_from_float
    """

    chains: PositiveInt
    simulations: PositiveInt
    blocks: PositiveInt


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
                2 * 1024 * job_size.simulations * job_size.blocks
                if memory is None
                else memory
            ),
        )
    return JobResources(
        nodes=job_size.chains if nodes is None else nodes,
        cpus=2 if cpus is None else cpus,
        memory=2 * 1024 if memory is None else memory,
    )


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
) -> JobSubmission:
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
                "If not dry mode would have executed script with: %s",
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

    Examples:
        >>> from datetime import timedelta
        >>> from gempyor.batch import BatchSystem, JobResources, JobSize
        >>> class MyCustomBatchSystem(BatchSystem):
        ...     name = "my_custom"
        >>> batch_system = MyCustomBatchSystem()
        >>> resources = JobResources(nodes=1, cpus=2, memory=1024)
        >>> batch_system.format_nodes(resources)
        '1'
        >>> batch_system.format_cpus(resources)
        '2'
        >>> batch_system.format_memory(resources)
        '1024
        >>> time_limit = timedelta(days=1, hours=2, minutes=34, seconds=56)
        >>> batch_system.format_time_limit(time_limit)
        '1595'
        >>> batch_system.size_from_jobs_simulations_blocks(2, 10, 5)
        JobSize(chains=2, simulations=10, blocks=5)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @overload
    @abstractmethod
    def submit(
        self,
        script: Path,
        options: dict[str, Any] | None = None,
        verbosity: int | None = None,
        dry_run: Literal[True] = ...,
    ) -> None: ...

    @overload
    @abstractmethod
    def submit(
        self,
        script: Path,
        options: dict[str, Any] | None = None,
        verbosity: int | None = None,
        dry_run: Literal[False] = ...,
    ) -> JobSubmission: ...

    @abstractmethod
    def submit(
        self,
        script: Path,
        options: dict[str, Any] | None = None,
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
        chains: int,
        simulations: int,
        blocks: int,
    ) -> JobSize:
        """
        Infer a job size from several explicit and implicit parameters.

        Args:
            chains: An explicit number of chains.
            simulations: An explicit number of simulations per a block.
            blocks: An explicit number of blocks per a job.
            inference_method: The inference method being used as different methods have
                different restrictions.

        Returns:
            A job size instance with either the explicit or inferred job sizing.
        """
        return JobSize(chains=chains, simulations=simulations, blocks=blocks)


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
        >>> batch_system.size_from_jobs_simulations_blocks(1, 10, 1)
        JobSize(chains=1, simulations=10, blocks=1)
        >>> with warnings.catch_warnings(record=True) as warns:
        ...     size = batch_system.size_from_jobs_simulations_blocks(2, 10, 1)
        ...     for warn in warns:
        ...             print(warn.message)
        ...
        Local batch system only supports 1 job but was given 2, overriding.
        >>> size
        JobSize(chains=1, simulations=10, blocks=1)
        >>> with warnings.catch_warnings(record=True) as warns:
        ...     size = batch_system.size_from_jobs_simulations_blocks(1, 20, 1)
        ...     for warn in warns:
        ...             print(warn.message)
        ...
        Local batch system only supports 10 blocks x simulations but was given 20, overriding.
        >>> size
        JobSize(chains=1, simulations=10, blocks=1)
        >>> with warnings.catch_warnings(record=True) as warns:
        ...     size = batch_system.size_from_jobs_simulations_blocks(4, 10, 2)
        ...     for warn in warns:
        ...             print(warn.message)
        ...
        Local batch system only supports 1 job but was given 4, overriding.
        Local batch system only supports 10 blocks x simulations but was given 20, overriding.
        >>> size
        JobSize(chains=1, simulations=10, blocks=1)
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
        chains: int,
        simulations: int,
        blocks: int,
    ) -> JobSize:
        """
        Infer a job size from several explicit and implicit parameters.

        Args:
            chains: An explicit number of chains.
            simulations: An explicit number of simulations per a block.
            blocks: An explicit number of blocks per a job.
            inference_method: The inference method being used as different methods have
                different restrictions.

        Returns:
            A job size instance with either the explicit or inferred job sizing.
        """
        if chains != 1:
            warnings.warn(
                f"Local batch system only supports 1 chain but was given {chains}, overriding."
            )
        if (blocks_x_simulations := blocks * simulations) > 10:
            warnings.warn(
                "Local batch system only supports 10 blocks x simulations "
                f"but was given {blocks_x_simulations}, overriding."
            )
        return JobSize(chains=1, simulations=min(blocks_x_simulations, 10), blocks=1)


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

    name = "slurm"

    _sbatch_regex = re.compile(r"submitted batch job (\d+)", flags=re.IGNORECASE)

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

    def format_memory(self, job_resources: JobResources) -> str:
        return f"{job_resources.memory}MB"

    def format_time_limit(self, job_time_limit: timedelta) -> str:
        total_seconds = job_time_limit.total_seconds()
        hours = math.floor(total_seconds / (60.0 * 60.0))
        minutes = math.floor((total_seconds - (60.0 * 60.0 * hours)) / 60.0)
        seconds = math.ceil(total_seconds - (60.0 * minutes) - (60.0 * 60.0 * hours))
        return f"{hours}:{minutes:02d}:{seconds:02d}"


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


_reset_batch_systems()
