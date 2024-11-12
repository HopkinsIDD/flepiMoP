"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including creating 
metadata and job size calculations for example.
"""

__all__ = ["BatchSystem", "JobSize", "JobResources", "JobTimeLimit", "write_manifest"]


from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from getpass import getuser
import json
import math
from pathlib import Path
import re
from shlex import quote
from stat import S_IXUSR
import subprocess
import sys
from typing import Any, Literal, Self

import click

from ._jinja import _render_template_to_file, _render_template_to_temp_file
from .file_paths import run_id
from .info import Cluster, get_cluster_info
from .logging import get_script_logger
from .utils import _format_cli_options, _git_head, _shutil_which, config
from .shared_cli import (
    NONNEGATIVE_DURATION,
    cli,
    config_files_argument,
    config_file_options,
    log_cli_inputs,
    mock_context,
    parse_config_files,
    verbosity_options,
)


_JOB_NAME_REGEX = re.compile(r"^[a-z]{1}([a-z0-9\_\-]+)?$", flags=re.IGNORECASE)


class BatchSystem(Enum):
    """
    Enum representing the various batch systems that flepiMoP can run on.
    """

    AWS = auto()
    LOCAL = auto()
    SLURM = auto()

    @classmethod
    def from_options(
        cls,
        batch_system: Literal["aws", "local", "slurm"] | None,
        aws: bool,
        local: bool,
        slurm: bool,
    ) -> "BatchSystem":
        """
        Resolve the batch system options.

        Args:
            batch_system: The name of the batch system to use if provided explicitly by
                name or `None` to rely on the other flags.
            aws: A flag indicating if the batch system should be AWS.
            local: A flag indicating if the batch system should be local.
            slurm: A flag indicating if the batch system should be slurm.

        Returns:
            The name of the batch system to use given the user options.
        """
        batch_system = batch_system.lower() if batch_system is not None else batch_system
        if (boolean_flags := sum((aws, local, slurm))) > 1:
            raise ValueError(
                f"There were {boolean_flags} boolean flags given, expected either 0 or 1."
            )
        if batch_system is not None:
            for name, flag in zip(("aws", "local", "slurm"), (aws, local, slurm)):
                if flag and batch_system != name:
                    raise ValueError(
                        "Conflicting batch systems given. The batch system name "
                        f"is '{batch_system}' and the flags indicate '{name}'."
                    )
        if batch_system is None:
            if aws:
                batch_system = "aws"
            elif local:
                batch_system = "local"
            else:
                batch_system = "slurm"
        if batch_system == "aws":
            return cls.AWS
        elif batch_system == "local":
            return cls.LOCAL
        return cls.SLURM


@dataclass(frozen=True, slots=True)
class JobSize:
    """
    A batch submission job size.

    Attributes:
        jobs: The number of jobs to use.
        simulations: The number of simulations to run per a block.
        blocks: The number of sequential blocks to run per a job.

    Raises:
        ValueError: If any of the attributes are less than 1.
    """

    jobs: int
    simulations: int
    blocks: int

    def __post_init__(self) -> None:
        for p in self.__slots__:
            if (val := getattr(self, p)) < 1:
                raise ValueError(
                    (
                        f"The '{p}' attribute must be greater than 0, "
                        f"but instead was given '{val}'."
                    )
                )

    @classmethod
    def size_from_jobs_sims_blocks(
        cls,
        jobs: int | None,
        simulations: int | None,
        blocks: int | None,
        inference_method: Literal["emcee"] | None,
    ) -> "JobSize":
        """
        Infer a job size from several explicit and implicit parameters.

        Args:
            jobs: An explicit number of jobs.
            simulations: An explicit number of simulations per a block.
            blocks: An explicit number of blocks per a job.
            inference_method: The inference method being used as different methods have
                different restrictions.

        Returns:
            A job size instance with either the explicit or inferred job sizing.
        """
        if inference_method == "emcee":
            return cls(jobs=jobs, simulations=blocks * simulations, blocks=1)
        return cls(jobs=jobs, simulations=simulations, blocks=blocks)


@dataclass(frozen=True, slots=True)
class JobResources:
    """
    A batch submission job resources request.

    Attributes:
        nodes: The number of nodes to request.
        cpus: The number of CPUs to request per a node.
        memory: The amount of memory to request per a node in MB.

    Raises:
        ValueError: If any of the attributes are less than 1.
    """

    nodes: int
    cpus: int
    memory: int

    def __post_init__(self) -> None:
        for p in self.__slots__:
            if (val := getattr(self, p)) < 1:
                raise ValueError(
                    (
                        f"The '{p}' attribute must be greater than 0, "
                        f"but instead was given '{val}'."
                    )
                )

    @classmethod
    def from_presets(
        cls,
        job_size: JobSize,
        inference_method: Literal["emcee"] | None,
        nodes: int | None = None,
        cpus: int | None = None,
        memory: int | None = None,
    ) -> "JobResources":
        """
        Calculate suggested job resources from presets with optional overrides.

        Args:
            job_size: The size of the job being ran.
            inference_method: The inference method being used for this job.
            nodes: Optional manual override for the number of nodes.
            cpus: Optional manual override for the number of CPUs per node.
            memory: Optional manual override for the amount of memory per node.

        Returns:
            A job resources instances scaled to the job size given.
        """
        if inference_method == "emcee":
            nodes = 1 if nodes is None else nodes
            cpus = 2 * job_size.jobs if cpus is None else cpus
            memory = 2 * 1024 * job_size.simulations if memory is None else memory
        else:
            nodes = job_size.jobs if nodes is None else nodes
            cpus = 2 if cpus is None else cpus
            memory = 2 * 1024 if memory is None else memory
        return cls(nodes=nodes, cpus=cpus, memory=memory)

    @property
    def total_cpus(self) -> int:
        """
        Calculate the total number of CPUs.

        Returns:
            The total number of CPUs represented by this instance.
        """
        return self.nodes * self.cpus

    @property
    def total_memory(self) -> int:
        """
        Calculate the total amount of memory.

        Returns:
            The total amount of memory represented by this instance.
        """
        return self.nodes * self.memory

    def total_resources(self) -> tuple[int, int, int]:
        """
        Calculate the total resources.

        Returns:
            A tuple of the nodes, total CPUs, and total memory represented by
            this instance.
        """
        return (self.nodes, self.total_cpus, self.total_memory)

    def format_nodes(self, batch_system: BatchSystem | None) -> str:
        return str(self.nodes)

    def format_cpus(self, batch_system: BatchSystem | None) -> str:
        return str(self.cpus)

    def format_memory(self, batch_system: BatchSystem | None) -> str:
        if batch_system == BatchSystem.SLURM:
            return f"{self.memory}MB"
        return str(self.memory)


@dataclass(frozen=True, slots=True)
class JobTimeLimit:
    """
    A batch submission job time limit.

    Attributes:
        time_limit: The time limit of the batch job.

    Raises:
        ValueError: If the `time_limit` attribute is not positive.
    """

    time_limit: timedelta

    def __post_init__(self) -> None:
        if (total_seconds := self.time_limit.total_seconds()) <= 0.0:
            raise ValueError(
                f"The `time_limit` attribute has {math.floor(total_seconds):,} "
                "seconds, which is less than or equal to 0."
            )

    def __str__(self) -> str:
        return self.format()

    def __hash__(self) -> int:
        return hash(self.time_limit)

    def __eq__(self, other: Self | timedelta) -> bool:
        if isinstance(other, JobTimeLimit):
            return self.time_limit == other.time_limit
        if isinstance(other, timedelta):
            return self.time_limit == other
        raise TypeError(
            "'==' not supported between instances of "
            f"'JobTimeLimit' and '{type(other).__name__}'."
        )

    def __lt__(self, other: Self | timedelta) -> bool:
        if isinstance(other, JobTimeLimit):
            return self.time_limit < other.time_limit
        if isinstance(other, timedelta):
            return self.time_limit < other
        raise TypeError(
            "'<' not supported between instances of "
            f"'JobTimeLimit' and '{type(other).__name__}'."
        )

    def __le__(self, other: Self | timedelta) -> bool:
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other: Self | timedelta) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Self | timedelta) -> bool:
        return self.__eq__(other) or self.__gt__(other)

    def format(self, batch_system: BatchSystem | None = None) -> str:
        """
        Format the job time limit as a string appropriate for a given batch system.

        Args:
            batch_system: The batch system the format should be formatted for.

        Returns:
            The time limit formatted for the batch system.

        Examples:
            >>> from gempyor.batch import BatchSystem
            >>> from datetime import timedelta
            >>> job_time_limit = JobTimeLimit(
            ...     time_limit=timedelta(days=1, hours=2, minutes=34, seconds=5)
            ... )
            >>> job_time_limit.format()
            '1595'
            >>> job_time_limit.format(batch_system=BatchSystem.SLURM)
            '26:34:05'
        """
        if batch_system == BatchSystem.SLURM:
            total_seconds = self.time_limit.total_seconds()
            hours = math.floor(total_seconds / (60.0 * 60.0))
            minutes = math.floor((total_seconds - (60.0 * 60.0 * hours)) / 60.0)
            seconds = math.ceil(total_seconds - (60.0 * minutes) - (60.0 * 60.0 * hours))
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        limit_in_mins = math.ceil(self.time_limit.total_seconds() / 60.0)
        return str(limit_in_mins)

    @classmethod
    def from_per_simulation_time(
        cls, job_size: JobSize, time_per_simulation: timedelta, initial_time: timedelta
    ) -> "JobTimeLimit":
        """
        Construct a job time limit that scales with job size.

        Args:
            job_size: The job size to scale the time limit with.
            time_per_simulation: The time per a simulation.
            initial_time: Time required to setup per a job.

        Returns:
            A job time limit that is scaled to match `job_size`.

        Raises:
            ValueError: If `time_per_simulation` is non-positive.
            ValueError: If `initial_time` is non-positive.
        """
        if (total_seconds := time_per_simulation.total_seconds()) <= 0.0:
            raise ValueError(
                f"The `time_per_simulation` is '{math.floor(total_seconds):,}' "
                "seconds, which is less than or equal to 0."
            )
        if (total_seconds := initial_time.total_seconds()) <= 0.0:
            raise ValueError(
                f"The `initial_time` is '{math.floor(total_seconds):,}' "
                "seconds, which is less than or equal to 0."
            )
        time_limit = (
            job_size.blocks * job_size.simulations * time_per_simulation
        ) + initial_time
        return cls(time_limit=time_limit)


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
        >>> from pathlib import Path
        >>> flepi_path = Path("~/Desktop/GitHub/HopkinsIDD/flepiMoP").expanduser()
        >>> project_path = Path("~/Desktop/GitHub/HopkinsIDD/flepimop_sample").expanduser()
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


def _local(script: Path, verbosity: int | None, dry_run: bool) -> None:
    """
    Execute a local script.

    Args:
        script: The script to run.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually execute the script given.
    """
    if verbosity is not None:
        logger = get_script_logger(__name__, verbosity)
        logger.debug("Using script '%s' for local execution", script.absolute())

    if not script.exists() or not script.is_file():
        raise ValueError(
            f"The script '{script.absolute()}' either does not exist or is not a file."
        )
    if not bool((current_perms := script.stat().st_mode) & S_IXUSR):
        if verbosity is not None:
            logger.warning(
                "The script '%s' is not executable, making it executable.",
                script.absolute(),
            )
        new_perms = current_perms | S_IXUSR
        script.chmod(new_perms)
    cmd_args = [str(script.absolute())]

    if dry_run:
        if verbosity is not None:
            logger.info(
                "If not dry mode would have executed script with: %s",
                " ".join(cmd_args),
            )
        return

    if verbosity is not None:
        logger.info("Executing script with: %s", " ".join(cmd_args))
    process = subprocess.run(cmd_args, check=True, capture_output=True)
    stdout = process.stdout.decode().strip()
    stderr = process.stderr.decode().strip()
    if verbosity is not None:
        if stdout:
            logger.debug("Captured stdout from executed script: %s", stdout)
        if stderr:
            logger.error("Captured stderr from executed script: %s", stderr)


def _local_template(
    template: str,
    script: Path | None,
    template_data: dict[str, Any],
    verbosity: int | None,
    dry_run: bool,
) -> Path:
    """
    Execute a local script generated from a template.

    Args:
        template: The name of the template to use for rendering the executable script.
        script: Either the path of where to save the rendered executable script to or
            `None` for a tmp file.
        template_data: Data accessible to the template when rendering.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit a job to slurm.

    Returns:
        The rendered executable script, either a tmp file if `script` is None otherwise
        `script` is returned.
    """
    if script is None:
        script = _render_template_to_temp_file(template, template_data, suffix=".bash")
        new_perms = script.stat().st_mode | S_IXUSR
        script.chmod(new_perms)
    else:
        _render_template_to_file(template, template_data, script)
    _local(script, verbosity, dry_run)
    return script


def _sbatch(
    script: Path,
    environment_variables: dict[str, Any] | Literal["all", "nil", "none"],
    options: dict[str, Any],
    verbosity: int | None,
    dry_run: bool,
) -> None:
    """
    Submit a job to slurm via the `sbatch` command.

    Args:
        script: The batch file to submit to slurm.
        environment_variables: Environment variables to pass to the job via the
            '--export' option. Keys correspond to the variable name and the value is
            the variable value. All values are coerced to a string and then escaped.
            Or can be a literal for one of the '--export' option's special values.
        options: Options to pass when calling sbatch. Keys correspond to the option
            name and the value is the option value. Options can be provided as either
            the long or short name, but this function will not be able to determine that
            these are duplicates so only using long names is recommended.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit a job to slurm.

    Raises:
        ValueError: If 'export' is given as a key in `options` instead of using
            `environment_variables`.

    Examples:
        >>> from pathlib import Path
        >>> _sbatch(
        ...     Path("my_batch_script.sbatch"),
        ...     {"VAR1": 1, "VAR2": "true"},
        ...     {"J": "My job name", "output": Path("out.log")},
        ...     3,
        ...     True,
        ... )
        2024-10-31 09:26:48,361:DEBUG:gempyor._slurm> Using batch script '/my_batch_script.sbatch' to submit to slurm.
        2024-10-31 09:26:48,363:INFO:gempyor._slurm> If not dry mode would have submitted to slurm with: /usr/bin/sbatch --export=VAR1=1,VAR2=true -J='My job name' --output=out.log /my_batch_script.sbatch.

    See Also:
        [`sbatch`'s documentation](https://slurm.schedmd.com/sbatch.html)
    """
    if "export" in options:
        raise ValueError(
            "Found 'export' in `options`, please use `environment_variables` instead."
        )

    if verbosity is not None:
        logger = get_script_logger(__name__, verbosity)
        logger.debug("Using batch script '%s' to submit to slurm", script.absolute())

    if isinstance(environment_variables, dict):
        env_vars = [f"{k}={quote(str(v))}" for k, v in environment_variables.items()]
        export = "--export=" + ",".join(env_vars)
    else:
        export = f"--export={environment_variables.upper()}"
    options = _format_cli_options(options)
    sbatch_cmd = _shutil_which("sbatch")
    if len(export) > 9:
        cmd_args = [sbatch_cmd, export] + options + [str(script.absolute())]
    else:
        cmd_args = [sbatch_cmd] + options + [str(script.absolute())]

    if dry_run:
        if verbosity is not None:
            logger.info(
                "If not dry mode would have submitted to slurm with: %s",
                " ".join(cmd_args),
            )
        return

    if verbosity is not None:
        logger.info("Submitting to slurm with: %s", " ".join(cmd_args))
    process = subprocess.run(cmd_args, check=True, capture_output=True)
    stdout = process.stdout.decode().strip()
    stderr = process.stderr.decode().strip()
    if verbosity is not None:
        if stdout:
            logger.debug("Captured stdout from sbatch submission: %s", stdout)
        if stderr:
            logger.error("Captured stderr from sbatch submission: %s", stderr)


def _sbatch_template(
    template: str,
    script: Path | None,
    template_data: dict[str, Any],
    environment_variables: dict[str, Any] | Literal["all", "nil", "none"],
    options: dict[str, Any],
    verbosity: int | None,
    dry_run: bool,
) -> Path:
    """
    Submit a job from a template to slurm via the `sbatch` command.

    Args:
        template: The name of the template to use for rendering the sbatch script.
        script: Either the path of where to save the rendered sbatch script to or `None`
            for a tmp file.
        template_data: Data accessible to the template when rendering.
        environment_variables: Environment variables to pass to the job via the
            '--export' option. Keys correspond to the variable name and the value is
            the variable value. All values are coerced to a string and then escaped.
            Or can be a literal for one of the '--export' option's special values.
        options: Options to pass when calling sbatch. Keys correspond to the option
            name and the value is the option value. Options can be provided as either
            the long or short name, but this function will not be able to determine that
            these are duplicates so only using long names is recommended.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit a job to slurm.

    Returns:
        The rendered sbatch script, either a tmp file if `script` is None otherwise
        `script` is returned.

    Raises:
        ValueError: If 'options' is found in `template_data` and `options` is not empty.
    """
    template_data["interpreter"] = template_data.get("interpreter", "bash")
    if "options" in template_data and options:
        raise ValueError(
            "Found 'options' in `template_data` but `options` is not empty, can only one."
        )
    template_data["options"] = _format_cli_options(template_data.get("options", options))
    if script is None:
        script = _render_template_to_temp_file(template, template_data, suffix=".sbatch")
    else:
        _render_template_to_file(template, template_data, script)
    _sbatch(script, environment_variables, {}, verbosity, dry_run)
    return script


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


@cli.command(
    name="submit",
    params=[config_files_argument]
    + list(config_file_options.values())
    + [
        click.Option(
            param_decls=["--simulations", "simulations"],
            default=None,
            type=click.IntRange(min=1),
            help="The number of simulations per a job.",
        ),
        click.Option(
            param_decls=["--blocks", "blocks"],
            default=None,
            type=click.IntRange(min=1),
            help="The number of sequential blocks to run per a job.",
        ),
        click.Option(
            param_decls=["--batch-system", "batch_system"],
            default=None,
            type=click.Choice(("aws", "local", "slurm"), case_sensitive=False),
            help="The name of the batch system being used.",
        ),
        click.Option(
            param_decls=["--aws", "aws"],
            default=False,
            type=bool,
            is_flag=True,
            help="A flag indicating this is being run on AWS.",
        ),
        click.Option(
            param_decls=["--local", "local"],
            default=False,
            type=bool,
            is_flag=True,
            help="A flag indicating this is being run local.",
        ),
        click.Option(
            param_decls=["--slurm", "slurm"],
            default=False,
            type=bool,
            is_flag=True,
            help="A flag indicating this is being run on slurm.",
        ),
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
            param_decls=["--partition", "partition"],
            type=str,
            default=None,
            help="The slurm partition to submit the job to.",
        ),
        click.Option(
            param_decls=["--simulation-time", "simulation_time"],
            type=NONNEGATIVE_DURATION,
            default="3min",
            help="The time limit per a simulation.",
        ),
        click.Option(
            param_decls=["--initial-time", "initial_time"],
            type=NONNEGATIVE_DURATION,
            default="20min",
            help="The initialization time limit.",
        ),
        click.Option(
            param_decls=["--cluster", "cluster"],
            type=str,
            default=None,
            help=(
                "The name of the cluster this job is being launched on. "
                "Only applicable to slurm submissions."
            ),
        ),
        click.Option(
            param_decls=["--conda-env", "conda_env"],
            type=str,
            default="flepimop-env",
            help="The name of the conda environment being used for this batch run.",
        ),
        click.Option(
            param_decls=["--debug", "debug"],
            type=bool,
            default=False,
            is_flag=True,
            help="Flag to enable debugging in batch submission scripts.",
        ),
        click.Option(
            param_decls=["--config-out", "config_out"],
            type=click.Path(path_type=Path),
            default=None,
            help=(
                "The location to dump the final parsed config file. "
                "If not provided a temporary file will be used."
            ),
        ),
        click.Option(
            param_decls=["--id", "--run-id", "run_id"],
            envvar="FLEPI_RUN_INDEX",
            type=str,
            default=None,
            help="Unique identifier for this run.",
        ),
        click.Option(
            param_decls=["--prefix", "prefix"],
            envvar="FLEPI_PREFIX",
            type=str,
            default=None,
            help="Unique prefix for this run.",
        ),
        click.Option(
            param_decls=["--email", "email"],
            type=str,
            default=None,
            help="Optionally an email that can be notified on job begin and end.",
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
            type=click.IntRange(min=1),
            default=None,
            help="Override for the amount of memory per node to use in MB.",
        ),
    ]
    + list(verbosity_options.values()),
)
@click.pass_context
def _click_submit(ctx: click.Context = mock_context, **kwargs) -> None:
    """Submit batch jobs"""
    # Generic setup
    now = datetime.now(timezone.utc)
    logger = get_script_logger(__name__, kwargs["verbosity"])
    log_cli_inputs(kwargs)
    cfg = parse_config_files(config, ctx, **kwargs)

    # Temporary limitation
    if (
        not cfg["inference"].exists()
        or not cfg["inference"]["method"].exists()
        or cfg["inference"]["method"].as_str() != "emcee"
    ):
        raise NotImplementedError(
            "The `flepimop submit` CLI only supports EMCEE inference jobs."
        )
    inference_method = cfg["inference"]["method"].as_str()
    inference_method = (
        inference_method if inference_method is None else inference_method.lower()
    )

    # Job name/run id
    name = cfg["name"].get(str) if cfg["name"].exists() else None
    job_name = _job_name(name, now)
    logger.info("Assigning job name of '%s'", job_name)
    if kwargs["run_id"] is None:
        kwargs["run_id"] = run_id(now)
    logger.info("Using a run id of '%s'", kwargs["run_id"])

    # Batch system
    batch_system = BatchSystem.from_options(
        kwargs["batch_system"], kwargs["aws"], kwargs["local"], kwargs["slurm"]
    )
    if batch_system != BatchSystem.SLURM:
        # Temporary limitation
        raise NotImplementedError(
            "The `flepimop submit` CLI only supports batch submission to slurm."
        )
    logger.info("Constructing a job to submit to %s", batch_system)
    if batch_system != BatchSystem.SLURM and kwargs["email"] is not None:
        logger.warning(
            "The email option, given '%s', is only used when "
            "the batch system is slurm, but is instead %s.",
            kwargs["email"],
            batch_system,
        )

    # Job size
    job_size = JobSize.size_from_jobs_sims_blocks(
        kwargs["jobs"],
        kwargs["simulations"],
        kwargs["blocks"],
        inference_method,
    )
    logger.info("Preparing a job with size %s", job_size)
    if inference_method == "emcee" and job_size.blocks != 1:
        logger.warning(
            "When using EMCEE for inference the job size blocks is ignored, given %u.",
            job_size.blocks,
        )

    # Job time limit
    job_time_limit = JobTimeLimit.from_per_simulation_time(
        job_size, kwargs["simulation_time"], kwargs["initial_time"]
    )
    logger.info("Setting a total job time limit of %s minutes", job_time_limit.format())

    # Job resources
    job_resources = JobResources.from_presets(
        job_size,
        inference_method,
        nodes=kwargs["nodes"],
        cpus=kwargs["cpus"],
        memory=kwargs["memory"],
    )

    # Cluster info
    cluster: Cluster | None = None
    if batch_system == BatchSystem.SLURM:
        if kwargs["cluster"] is None:
            raise ValueError("When submitting a batch job to slurm a cluster is required.")
        cluster = get_cluster_info(kwargs["cluster"])
    logger.info("Utilizing info for the '%s' cluster to construct this job", cluster.name)
    logger.debug(
        "The full settings for cluster '%s' are %s", cluster.name, cluster.model_dump()
    )

    # Restart/continuation location
    # TODO: Implement this

    # Manifest
    manifest = write_manifest(job_name, kwargs["flepi_path"], kwargs["project_path"])
    logger.info("Writing manifest metadata to '%s'", manifest.absolute())

    # Config out
    if (config_out := kwargs["config_out"]) is None:
        config_out = kwargs["project_path"] / f"config_{job_name}.yml"
    with config_out.open(mode="w") as f:
        f.write(cfg.dump())
    logger.info(
        "Dumped the final config for this batch submission to %s", config_out.absolute()
    )

    # Construct the sbatch call
    template_data = {
        "conda_env": kwargs["conda_env"],
        "config_out": config_out.absolute(),
        "cluster": cluster if cluster is None else cluster.model_dump(),
        "debug": kwargs["debug"],
        "flepi_path": kwargs["flepi_path"].absolute(),
        "inference_method": "" if inference_method is None else inference_method,
        "job_name": job_name,
        "jobs": job_size.jobs,
        "nslots": job_size.simulations,  # aka nwalkers
        "prefix": kwargs["prefix"],
        "project_path": kwargs["project_path"].absolute(),
        "run_id": kwargs["run_id"],
        "simulations": job_size.simulations,
    }
    options = {
        "chdir": kwargs["project_path"].absolute(),
        "comment": f"Generated on {now:%c %Z} and submitted by {getuser()}.",
        "cpus-per-task": job_resources.format_cpus(batch_system),
        "job-name": job_name,
        "mem": job_resources.format_memory(batch_system),
        "nodes": job_resources.format_nodes(batch_system),
        "ntasks-per-node": 1,
        "time": job_time_limit.format(batch_system),
    }
    if kwargs["partition"] is not None:
        options["partition"] = kwargs["partition"]
    if kwargs["email"] is not None:
        options["mail-type"] = "BEGIN,END"
        options["mail-user"] = kwargs["email"]
    _sbatch_template(
        "inference.sbatch.j2",
        None,
        template_data,
        "all",
        options,
        kwargs["verbosity"],
        kwargs["dry_run"],
    )
