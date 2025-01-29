"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including creating 
metadata and job size calculations for example.
"""

__all__ = [
    "BatchSystem",
    "SlurmBatchSystem",
    "LocalBatchSystem",
    "register_batch_system",
    "get_batch_system",
    "JobSize",
    "JobResources",
    "write_manifest",
]


from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal, overload
import warnings

from .utils import _git_head


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


_batch_systems = []


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


class BatchSystem(ABC):
    """
    An abstract base class for batch systems.

    This class contains logic for interacting with batch systems, such as formatting
    job resources and time limits.

    Attributes:
        name: The name of the batch system. Must be unique when registered.
    """

    @property
    @abstractmethod
    def name(self) -> str:
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
        return str(self.memory)

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
        jobs: int,
        simulations: int,
        blocks: int,
    ) -> JobSize:
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
        return JobSize(jobs=jobs, simulations=simulations, blocks=blocks)


class SlurmBatchSystem(BatchSystem):
    name = "slurm"

    def format_memory(self, job_resources: JobResources) -> str:
        return f"{job_resources.memory}MB"

    def format_time_limit(self, job_time_limit: timedelta) -> str:
        total_seconds = job_time_limit.total_seconds()
        hours = math.floor(total_seconds / (60.0 * 60.0))
        minutes = math.floor((total_seconds - (60.0 * 60.0 * hours)) / 60.0)
        seconds = math.ceil(total_seconds - (60.0 * minutes) - (60.0 * 60.0 * hours))
        return f"{hours}:{minutes:02d}:{seconds:02d}"


class LocalBatchSystem(BatchSystem):
    name = "local"

    def size_from_jobs_simulations_blocks(
        self,
        jobs: int,
        simulations: int,
        blocks: int,
    ) -> JobSize:
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
        if jobs != 1:
            warnings.warn(
                f"Local batch system only supports 1 job but was given {jobs}, overriding."
            )
        if (blocks_x_simulations := blocks * simulations) > 10:
            warnings.warn(
                "Local batch system only supports 10 blocks x simulations "
                f"but was given {blocks_x_simulations}, overriding."
            )
        return JobSize(jobs=1, simulations=min(blocks_x_simulations, 10), blocks=1)


def register_batch_system(batch_system: BatchSystem) -> None:
    """
    Register a batch system with gempyor.

    Args:
        batch_system: The batch system to register.

    Returns:
        None

    Raises:
        ValueError: If the batch system is already registered, checks the `name`
            attribute to determine this.
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
