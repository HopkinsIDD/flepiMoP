__all__ = (
    "BatchSystem",
    "LocalBatchSystem",
    "SlurmBatchSystem",
    "get_batch_system",
    "register_batch_system",
)


from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import timedelta
import logging
import math
import platform
from pathlib import Path
import re
from stat import S_IXUSR
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Literal, overload
import warnings

import confuse
from pydantic import PositiveInt

from .._jinja import _jinja_environment
from ..logging import get_script_logger
from ..utils import _format_cli_options, _shutil_which
from .types import JobResources, JobResult, JobSize, JobSubmission


_batch_systems = []


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
        with NamedTemporaryFile(mode="w", dir=Path.cwd(), delete=False) as temp:
            temp_script = Path(temp.name).absolute()
            temp_script.write_text(
                _jinja_environment.get_template("submit_command.bash.j2").render(
                    {**kwargs, **{"command": command}}
                )
            )
            if logger is not None:
                logger.info(
                    "Submit command script placed at '%s' for inspection.", temp_script
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
            dir=Path.cwd(),
            delete=False,
        ) as temp_script:
            sbatch_script = Path(temp_script.name).absolute()
            sbatch_script.write_text(
                _jinja_environment.get_template("sbatch_submit_command.bash.j2").render(
                    {**kwargs, **{"command": command}}
                )
            )
            if logger is not None:
                logger.info("Using sbatch script '%s' for submission", sbatch_script)
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
    logger: logging.Logger | None,
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
    logger: logging.Logger | None,
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
    logger: logging.Logger | None,
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


_reset_batch_systems()
