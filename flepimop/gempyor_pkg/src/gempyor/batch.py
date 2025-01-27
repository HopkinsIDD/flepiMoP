"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including creating 
metadata and job size calculations for example.
"""

__all__ = ["JobSize", "JobTimeLimit"]


from dataclasses import dataclass
from datetime import timedelta
import math
from typing import Literal, Self


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
        iterations_per_slot: int | None,
        slots: int | None,
        subpops: int | None,
        batch_system: Literal["aws", "local", "slurm"],
    ) -> "JobSize":
        """
        Infer a job size from several explicit and implicit parameters.

        Args:
            jobs: An explicit number of jobs.
            simulations: An explicit number of simulations per a block.
            blocks: An explicit number of blocks per a job.
            iterations_per_slot: A total number of iterations per a job, which is
                simulations times blocks. Required if `simulations` or `blocks` is
                not given.
            slots: An implicit number of slots to use for the job. Required if `jobs`
                is not given.
            subpops: The number of subpopulations being considered in this job. Affects
                the inferred simulations per a job on AWS. Required if `simulations`
                and `blocks` are not given.
            batch_size: The system the job is being sized for. Affects the inferred
                simulations per a job.

        Returns:
            A job size instance with either the explicit or inferred job sizing.

        Examples:
            >>> JobSize.size_from_jobs_sims_blocks(1, 2, 3, None, None, None, "local")
            JobSize(jobs=1, simulations=2, blocks=3)
            >>> JobSize.size_from_jobs_sims_blocks(
            ...     None, None, None, 100, 10, 25, "local"
            ... )
            JobSize(jobs=10, simulations=100, blocks=1)
            >>> JobSize.size_from_jobs_sims_blocks(None, None, 4, 100, 10, 25, "local")
            JobSize(jobs=10, simulations=25, blocks=4)

        Raises:
            ValueError: If `iterations_per_slot` is `None` and either `simulations` or
                `blocks` is `None`.
            ValueError: If `jobs` and `slots` are both `None`.
            ValueError: If `simulations`, `blocks`, and `subpops` are all `None`.
        """
        if iterations_per_slot is None and (simulations is None or blocks is None):
            raise ValueError(
                (
                    "If simulations and blocks are not all explicitly "
                    "provided then an iterations per slot must be given."
                )
            )

        jobs = slots if jobs is None else jobs
        if jobs is None:
            raise ValueError(
                "If jobs is not explicitly provided, it must be given via slots."
            )

        if simulations is None:
            if blocks is None:
                if subpops is None:
                    raise ValueError(
                        (
                            "If simulations and blocks are not explicitly "
                            "provided, then a subpops must be given."
                        )
                    )
                if batch_system == "aws":
                    simulations = 5 * math.ceil(max(60 - math.sqrt(subpops), 10) / 5)
                else:
                    simulations = iterations_per_slot
            else:
                simulations = math.ceil(iterations_per_slot / blocks)

        if blocks is None:
            blocks = math.ceil(iterations_per_slot / simulations)

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


def _resolve_batch_system(
    batch_system: Literal["aws", "local", "slurm"] | None,
    aws: bool,
    local: bool,
    slurm: bool,
) -> Literal["aws", "local", "slurm"]:
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
    return batch_system
