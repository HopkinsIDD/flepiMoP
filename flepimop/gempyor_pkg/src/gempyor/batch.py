"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including creating 
metadata and job size calculations for example.
"""

__all__ = ["JobSize"]


from dataclasses import dataclass
import math
from typing import Literal


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
