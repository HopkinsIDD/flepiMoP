__all__ = ()


from collections.abc import Iterable
from datetime import timedelta
from typing import Any, Literal, overload

from ..logging import get_script_logger
from ._inference import _create_inference_command
from .systems import BatchSystem
from .types import JobResources, JobSize, JobSubmission


@overload
def _submit_scenario_job(
    name: str,
    job_name: str,
    inference: Literal["emcee", "r"],
    job_size: JobSize,
    batch_system: BatchSystem,
    outcome_modifiers_scenario: str,
    seir_modifiers_scenario: str,
    options: None,
    template_data: dict[str, Any],
    verbosity: int,
    dry_run: Literal[False],
) -> JobSubmission: ...


@overload
def _submit_scenario_job(
    name: str,
    job_name: str,
    inference: Literal["emcee", "r"],
    job_size: JobSize,
    batch_system: BatchSystem,
    outcome_modifiers_scenario: str,
    seir_modifiers_scenario: str,
    options: None,
    template_data: dict[str, Any],
    verbosity: int,
    dry_run: Literal[True],
) -> JobSubmission: ...


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
) -> JobSubmission | None:
    """
    Submit a job for a scenario.

    Args:
        name: The name of the config file used as a prefix for the job name.
        job_name: The name of the job to submit.
        inference: The inference method to use.
        job_size: The size of the job to submit.
        batch_system: The batch system to submit the job to.
        outcome_modifiers_scenario: The outcome modifiers scenario to use.
        seir_modifiers_scenario: The SEIR modifiers scenario to use.
        options: The options to use for the job submission.
        template_data: The template data to use for the job submission.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit/run a job.

    Returns:
        A JobSubmission object if `dry_run` is `False`, otherwise `None`.
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
                "The SEIR modifiers scenario is `None`, may lead to "
                "unintended consequences in output file/directory names."
            )

    # Modify the job for the given scenario info
    job_name += f"_{seir_modifiers_scenario}_{outcome_modifiers_scenario}"
    prefix = f"{name}_{seir_modifiers_scenario}_{outcome_modifiers_scenario}"
    if verbosity is not None:
        logger.info(
            "Preparing a job for outcome and SEIR modifiers scenarios "
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
                f"{template_data.get('now', 'unknown')} with outcome and SEIR "
                f"modifiers scenarios '{outcome_modifiers_scenario}' and "
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

    # Submit inference job
    submission = batch_system.submit_command(
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

    # Optionally submit sync job
    if template_data.get("sync") is not None:
        sync_time_limit = timedelta(hours=1)
        sync_resources = JobResources(
            nodes=1,
            cpus=2,
            memory=1024,
        )
        sync_template_data = template_data | {
            "array_capable": False,
            "job_name": f"{template_data['job_name']}_sync",
            "job_time_limit": batch_system.format_time_limit(sync_time_limit),
            "job_comment": f"Sync {template_data['job_comment']}",
            "job_resources_nodes": batch_system.format_nodes(sync_resources),
            "job_resources_cpus": batch_system.format_cpus(sync_resources),
            "job_resources_memory": batch_system.format_memory(sync_resources),
        }
        batch_system.submit_command(
            f"flepimop sync --protocol {template_data['sync']} {template_data['config']}",
            options,
            verbosity,
            dry_run,
            **(
                {
                    k: v
                    for k, v in sync_template_data.items()
                    if k not in {"command", "options", "verbosity", "dry_run"}
                }
                | {"job_dependency": submission.job_id}
            ),
        )

    return submission
