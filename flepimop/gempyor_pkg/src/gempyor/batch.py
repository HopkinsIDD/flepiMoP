"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including creating 
metadata and job size calculations for example.
"""

__all__ = ["BatchSystem", "JobSize", "JobTimeLimit"]


from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
import math
from typing import Literal, Self


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
        inference_method = (
            inference_method if inference_method is None else inference_method.lower()
        )
        if inference_method == "emcee":
            return cls(jobs=jobs, simulations=blocks * simulations, blocks=1)
        return cls(jobs=jobs, simulations=simulations, blocks=blocks)


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

    def format(self, batch_system: Literal["aws", "local", "slurm"] | None = None) -> str:
        """
        Format the job time limit as a string appropriate for a given batch system.

        Args:
            batch_system: The batch system the format should be formatted for.

        Returns:
            The time limit formatted for the batch system.

        Examples:
            >>> from datetime import timedelta
            >>> job_time_limit = JobTimeLimit(
            ...     time_limit=timedelta(days=1, hours=2, minutes=34, seconds=5)
            ... )
            >>> job_time_limit.format()
            '1595'
            >>> job_time_limit.format(batch_system="slurm")
            '26:34:05'
        """
        if batch_system == "slurm":
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

        Examples:
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
