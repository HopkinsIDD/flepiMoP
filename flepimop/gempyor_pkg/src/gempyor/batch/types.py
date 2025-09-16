__all__ = ("EstimationSettings", "JobResources", "JobResult", "JobSize", "JobSubmission")


from datetime import timedelta
from math import isclose
from subprocess import CompletedProcess
import sys
from typing import Annotated, Any, Literal
import warnings

from pydantic import BaseModel, Field, PositiveInt, computed_field, model_validator

from ..constants import _SAMPLES_SIMULATIONS_RATIO


if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


class EstimationSettings(BaseModel):
    runs: PositiveInt
    interval: Annotated[float, Field(gt=0.0, lt=1.0)]
    vary: set[Literal["blocks", "chains", "simulations"]] = Field(min_length=1)
    factors: set[str] = Field(min_length=1)
    measurements: set[Literal["cpu", "memory", "time"]] = Field(min_length=1)
    scale_upper: Annotated[float, Field(gt=0.0)]
    scale_lower: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def check_factors(self) -> Self:
        if invalid_factors := self.factors - (
            JobSize.model_fields.keys() | JobSize.model_computed_fields.keys()
        ):
            raise ValueError(f"Factors must be derived from JobSize: {invalid_factors}.")
        return self

    @model_validator(mode="after")
    def check_scales(self) -> Self:
        if self.scale_upper >= self.scale_lower or isclose(
            self.scale_upper, self.scale_lower
        ):
            raise ValueError(
                f"The lower scale, {self.scale_lower}, must be greater "
                f"than the upper scale, {self.scale_upper}."
            )
        return self


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
        >>> JobResources(nodes=0, cpus=1, memory=1024)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for JobResources
        nodes
          Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
            For further information visit https://errors.pydantic.dev/2.11/v/greater_than
        >>> JobResources(nodes=2, cpus=4.5, memory=1024)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for JobResources
        cpus
          Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=4.5, input_type=float]
            For further information visit https://errors.pydantic.dev/2.11/v/int_from_float
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
                isclose(self.memory_efficiency, other.memory_efficiency)
                if (
                    self.memory_efficiency is not None
                    and other.memory_efficiency is not None
                )
                else self.memory_efficiency == other.memory_efficiency
            )
        )


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
        JobSize(blocks=5, chains=10, samples=100, simulations=200, samples_per_chain=500, simulations_per_chain=1000, total_samples=5000, total_simulations=10000)
        >>> JobSize(chains=32, simulations=500)
        JobSize(blocks=None, chains=32, samples=None, simulations=500, samples_per_chain=None, simulations_per_chain=500, total_samples=None, total_simulations=16000)
        >>> JobSize(blocks=0, chains=12, simulations=100)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for JobSize
        blocks
          Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
            For further information visit https://errors.pydantic.dev/2.11/v/greater_than
        >>> JobSize(chains=10, samples=50.5, simulations=200)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for JobSize
        samples
          Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=50.5, input_type=float]
            For further information visit https://errors.pydantic.dev/2.11/v/int_from_float
        >>> JobSize(samples=100, simulations=50)
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for JobSize
          Value error, The number of samples, 100, must be less than or equal to the number of simulations, 50, per a block. [type=value_error, input_value={'samples': 100, 'simulations': 50}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error
        >>> with warnings.catch_warnings(record=True) as warns:
        ...     JobSize(samples=75, simulations=100)
        ...     for warn in warns:
        ...             print(warn.message)
        JobSize(blocks=None, chains=None, samples=75, simulations=100, samples_per_chain=75, simulations_per_chain=100, total_samples=75, total_simulations=100)
        The samples to simulations ratio is 75%, which is higher than the recommended limit of 60%.
        >>> JobSize()
        JobSize(blocks=None, chains=None, samples=None, simulations=None, samples_per_chain=None, simulations_per_chain=None, total_samples=None, total_simulations=None)
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


class JobSubmission(CompletedProcess):
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
        cls, job_id: int | None, completed_process: CompletedProcess
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
