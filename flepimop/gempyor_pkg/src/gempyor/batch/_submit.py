"""Internal helpers for submitting scenario jobs to a batch system."""

__all__ = ()


from collections.abc import Iterable
from datetime import timedelta
from typing import Any

from ..logging import get_script_logger
from ._inference import _create_inference_command
from .systems import BatchSystem
from .types import JobResources, JobSize, JobSubmission


def _submit_scenario_job(
    job_size: JobSize,
    batch_system: BatchSystem,
    options: dict[str, str | Iterable[str]] | None,
    template_data: dict[str, Any],
) -> JobSubmission | None:
    """
    Submit a job for a scenario.

    Args:
        job_size: The size of the job to submit.
        batch_system: The batch system to submit the job to.
        options: The options to use for the job submission.
        template_data: The template data to use for the job submission.

    Returns:
        A JobSubmission object if `dry_run` is `False`, otherwise `None`.
    """

    # Get logger
    logger = get_script_logger(__name__, template_data["verbosity"])
    if template_data["outcome_modifiers_scenario"] == "None":
        logger.warning(
            "The outcome modifiers scenario is `None`, may lead to "
            "unintended consequences in output file/directory names."
        )
    if template_data["seir_modifiers_scenario"] == "None":
        logger.warning(
            "The SEIR modifiers scenario is `None`, may lead to "
            "unintended consequences in output file/directory names."
        )

    # Modify the job for the given scenario info
    suffix = (
        f"{template_data['seir_modifiers_scenario']}_"
        f"{template_data['outcome_modifiers_scenario']}"
    )
    job_name = f"{template_data['job_name']}_{suffix}"
    prefix = f"{template_data['name']}_{suffix}"
    if template_data["verbosity"] is not None:
        logger.info(
            "Preparing a job for outcome and SEIR modifiers scenarios "
            "'%s' and '%s', respectively, with job name '%s'.",
            template_data["outcome_modifiers_scenario"],
            template_data["seir_modifiers_scenario"],
            job_name,
        )

    # Template data
    template_data = {
        **template_data,
        **{
            "prefix": prefix,
            "outcome_modifiers_scenario": template_data["outcome_modifiers_scenario"],
            "seir_modifiers_scenario": template_data["seir_modifiers_scenario"],
            "job_comment": (
                f"{template_data['name']} submitted by "
                f"{template_data.get('user', 'unknown')} at "
                f"{template_data.get('now', 'unknown')} with outcome and SEIR "
                f"modifiers scenarios '{template_data['outcome_modifiers_scenario']}' "
                f"and '{template_data['seir_modifiers_scenario']}', respectively."
            ),
            "job_name": job_name,
        },
    }

    # Get inference command
    command = _create_inference_command(
        template_data["inference_method"],
        job_size,
        **{
            k: v
            for k, v in template_data.items()
            if k not in {"inference", "inference_method", "job_size"}
        },
    )

    # Submit inference job
    submission = batch_system.submit_command(
        command,
        options,
        template_data["verbosity"],
        template_data["dry_run"],
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
            "job_name": f"sync_{template_data['job_name']}",
            "job_time_limit": batch_system.format_time_limit(sync_time_limit),
            "job_comment": f"Sync {template_data['job_comment']}",
            "job_resources_nodes": batch_system.format_nodes(sync_resources),
            "job_resources_cpus": batch_system.format_cpus(sync_resources),
            "job_resources_memory": batch_system.format_memory(sync_resources),
        }
        project_path = str(template_data.get("project_path"))
        if not project_path.endswith("/"):
            project_path += "/"
        command = [
            "flepimop sync \\",
            f"  --protocol {template_data['sync_protocol']} \\",
            f"  --target '+ {template_data['job_name']}' \\",
            "  --mkpath \\",
            f"  --fprefix 's {template_data['job_name']}' \\",
            f"  {template_data['config']}",
            "",
            "flepimop sync \\",
            f"  --protocol {template_data['sync_protocol']} \\",
            f"  --target '+ {template_data['job_name']}' \\",
            "  --mkpath \\",
            f"  --fprefix '+ {template_data['job_name']}*.pdf' \\",
            f"  --source {project_path} \\",
            f"  {template_data['config']}",
            "",
            "flepimop sync \\",
            f"  --protocol {template_data['sync_protocol']} \\",
            f"  --target '+ {template_data['job_name']}' \\",
            "  --mkpath \\",
            f"  --fprefix '+ {template_data['job_name']}*.h5' \\",
            f"  --source {project_path} \\",
            f"  {template_data['config']}",
            "",
        ]
        if not template_data.get("skip_manifest", False):
            command += [
                "flepimop sync \\",
                f"  --protocol {template_data['sync_protocol']} \\",
                f"  --target '+ {template_data['job_name']}' \\",
                "  --mkpath \\",
                "  --source manifest.json \\",
                f"  {template_data['config']}",
                "",
            ]
        batch_system.submit_command(
            "\n".join(command),
            options,
            template_data["verbosity"],
            template_data["dry_run"],
            **(
                {
                    k: v
                    for k, v in sync_template_data.items()
                    if k not in {"command", "options", "verbosity", "dry_run"}
                }
                | {"job_dependency": submission.job_id if submission else None}
            ),
        )

    return submission
