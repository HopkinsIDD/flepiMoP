__all__ = ()


from collections.abc import Iterable
from typing import Any, Literal

from ..logging import get_script_logger
from ._inference import _create_inference_command
from .systems import BatchSystem
from .types import JobSize, JobSubmission


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
        outcome_modifiers_scenario: The outcome modifiers scenario to use.
        seir_modifiers_scenario: The SEIR modifiers scenario to use.
        name: The name of the config file used as a prefix for the job name.
        batch_system: The batch system to submit the job to.
        inference_method: The inference method being used.
        config_out: The path to the config file to use.
        job_name: The name of the job to submit.
        job_size: The size of the job to submit.
        job_time_limit: The time limit of the job to submit.
        job_resources: The resources required for the job to submit.
        cluster: The cluster information to use for submitting the job.
        kwargs: Additional options provided to the submit job CLI as keyword arguments.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit/run a job.
        now: The current UTC timestamp.
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

    if template_data.get("sync") is not None:
        batch_system.submit_command(
            f"flepimop sync --protocol {template_data['sync']} {template_data['config']}",
            options,
            verbosity,
            dry_run,
            **(
                {
                    k: v
                    for k, v in template_data.items()
                    if k not in {"command", "options", "verbosity", "dry_run"}
                }
                | {"job_dependency": submission.job_id}
            ),
        )

    return submission
