__all__ = ()


from typing import Literal
import warnings

from pydantic import PositiveInt

from .._jinja import _jinja_environment
from .types import JobResources, JobSize


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
