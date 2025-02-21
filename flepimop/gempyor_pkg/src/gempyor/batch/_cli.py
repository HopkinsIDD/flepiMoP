__all__ = ()


from datetime import datetime, timezone
from getpass import getuser
from itertools import product
from pathlib import Path
from typing import Any

import click

from .._click import DurationParamType, MemoryParamType
from ..file_paths import run_id
from ..info import get_cluster_info
from ..logging import get_script_logger
from ..shared_cli import (
    cli,
    config_files_argument,
    config_file_options,
    log_cli_inputs,
    mock_context,
    parse_config_files,
    verbosity_options,
)
from ..utils import _dump_formatted_yaml, _git_checkout, config
from ._estimate import _estimate_job_resources
from ._helpers import _job_name, _parse_extra_options
from ._inference import _inference_is_array_capable, _job_resources_from_size_and_inference
from ._submit import _submit_scenario_job
from .manifest import write_manifest
from .systems import _resolve_batch_system_name, get_batch_system


@cli.command(
    name="batch-calibrate",
    params=[config_files_argument]
    + list(config_file_options.values())
    + [
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
            param_decls=["--conda-env", "conda_env"],
            type=str,
            default="flepimop-env",
            help="The conda environment to use for the job.",
        ),
        click.Option(
            param_decls=["--blocks", "blocks"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of sequential blocks to run per a chain.",
        ),
        click.Option(
            param_decls=["--chains", "chains"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of chains or walkers, depending on inference method, to run.",
        ),
        click.Option(
            param_decls=["--samples", "samples"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of samples per a block.",
        ),
        click.Option(
            param_decls=["--simulations", "simulations"],
            required=True,
            type=click.IntRange(min=1),
            help="The number of simulations per a block.",
        ),
        click.Option(
            param_decls=["--time-limit", "time_limit"],
            type=DurationParamType(True, "minutes"),
            default="1hr",
            help=(
                "The time limit for the job. If units "
                "are not specified, minutes are assumed."
            ),
        ),
        click.Option(
            param_decls=["--batch-system", "batch_system"],
            default=None,
            type=str,
            help="The name of the batch system being used.",
        ),
        click.Option(
            param_decls=["--local", "local"],
            default=False,
            is_flag=True,
            help=(
                "Flag to use the local batch system. "
                "Equivalent to `--batch-system local`."
            ),
        ),
        click.Option(
            param_decls=["--slurm", "slurm"],
            default=False,
            is_flag=True,
            help=(
                "Flag to use the slurm batch system. "
                "Equivalent to `--batch-system slurm`."
            ),
        ),
        click.Option(
            param_decls=["--cluster", "cluster"],
            default=None,
            type=str,
            help=(
                "The name of the cluster being used, "
                "only needed if cluster info is required."
            ),
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
            type=MemoryParamType(True, "mb", True),
            default=None,
            help="Override for the amount of memory per node to use in MB.",
        ),
        click.Option(
            param_decls=["--estimate", "estimate"],
            type=bool,
            default=False,
            is_flag=True,
            help=(
                "Should this be submitted as an estimation job? If this flag is given "
                "then several jobs will be submitted with smaller sizes to estimate "
                "the time and resources needed for the full job. A time limit and "
                "memory requirement must still be given, but act as upper bounds on "
                "estimation jobs."
            ),
        ),
        click.Option(
            param_decls=["--estimate-runs", "estimate_runs"],
            type=click.IntRange(min=6),
            default=10,
            help=(
                "The number of estimation runs to perform. Must be at least 6 due to "
                "the estimation method, but more runs will provide a better estimate."
            ),
        ),
        click.Option(
            param_decls=["--estimate-interval", "estimate_interval"],
            type=click.FloatRange(min=0.0, max=1.0),
            default=0.9,
            help=(
                "The size of the prediction interval to use for estimating the "
                "required resources. Must be between 0 and 1."
            ),
        ),
        click.Option(
            param_decls=["--skip-manifest", "skip_manifest"],
            type=bool,
            default=False,
            is_flag=True,
            help="Flag to skip writing a manifest file, useful in dry runs.",
        ),
        click.Option(
            param_decls=["--skip-checkout", "skip_checkout"],
            type=bool,
            default=False,
            is_flag=True,
            help=(
                "Flag to skip checking out a new branch in "
                "the git repository, useful in dry runs."
            ),
        ),
        click.Option(
            param_decls=["--debug", "debug"],
            type=bool,
            default=False,
            is_flag=True,
            help="Flag to enable debugging in batch submission scripts.",
        ),
        click.Option(
            param_decls=["--extra", "extra"],
            type=str,
            default=None,
            multiple=True,
            help=(
                "Extra options to pass to the batch system. Please consult "
                "the batch system documentation for valid options."
            ),
        ),
    ]
    + list(verbosity_options.values()),
)
@click.pass_context
def _click_batch_calibrate(ctx: click.Context = mock_context, **kwargs: Any) -> None:
    """
    Submit a calibration job to a batch system.
    
    This job makes it straightforward to submit a calibration job to a batch system. The
    job will be submitted with the given configuration file and additional options. The
    general steps this tool follows are:
    
    \b
    1) Generate a unique job name from the configuration and timestamp,
    2) Determine the outcome/SEIR modifier scenarios to use,
    3) Determine the batch system to use and required job size/resources/time limit,
    4) Write a 'manifest.json' with job metadata, write the config used to a file, and
       checkout a new branch in the project git repository,
    5) Loop over the outcome/SEIR modifier scenarios and submit a job for each scenario.
    
    To get a better understanding of this tool you can use the `--dry-run` flag which
    will complete all of steps described above except for submitting the jobs. Or if you
    would like to test run the batch scripts without submitting to slurm or other batch 
    systems you can use the `--local` flag which will run the "batch" job locally (only 
    use for small test jobs).
    
    Here is an example of how to use this tool with the `examples/tutorials/` directory:
    
    \b
    ```bash
    $ flepimop batch-calibrate \\
        # The paths and conda environment to use
        --flepi-path $FLEPI_PATH \\
        --project-path $FLEPI_PATH/examples/tutorials \\
        --conda-env flepimop-env \\ 
        # The size of the job to run
        --blocks 1 \\
        --chains 50 \\
        --samples 100 \\
        --simulations 500 \\
        # The time limit for the job
        --time-limit 8hr \\
        # The batch system to use, equivalent to `--batch-system slurm`
        --slurm \\
        # Resource options
        --nodes 50 \\
        --cpus 2 \\
        --memory 4GB \\
        # Batch system specific options can be provided via `--extra`
        --extra partition=normal \\
        --extra email=bob@example.edu \\
        # Only run a dry run to see what would be submitted for the config
        --dry-run \\
        -vvv config_sample_2pop_inference.yml
    ```
    """
    # Generic setup
    now = datetime.now(timezone.utc)
    logger = get_script_logger(__name__, kwargs.get("verbosity", 0))
    log_cli_inputs(kwargs)
    cfg = parse_config_files(config, ctx, **kwargs)

    # Job name/run id
    name = cfg["name"].as_str() if cfg["name"].exists() else None
    job_name = _job_name(name, now)
    logger.info("Assigning job name of '%s'", job_name)
    if kwargs.get("run_id") is None:
        kwargs["run_id"] = run_id(now)
    logger.info("Using a run id of '%s'", kwargs.get("run_id"))

    # Parse extra options
    kwargs["extra"] = _parse_extra_options(kwargs.get("extra", None))
    logger.debug("Parsed extra options: %s", kwargs["extra"])

    # Inference method
    if not cfg["inference"].exists():
        logger.critical(
            "No inference section specified in the config "
            "file, likely missing important information."
        )
    inference_method = (
        cfg["inference"]["method"].as_str()
        if cfg["inference"].exists() and cfg["inference"]["method"].exists()
        else "r"
    ).lower()
    logger.info("Using inference method '%s'", inference_method)

    # Outcome/SEIR modifier scenarios
    outcome_modifiers_scenarios = (
        cfg["outcome_modifiers"]["scenarios"].as_str_seq()
        if cfg["outcome_modifiers"].exists()
        and cfg["outcome_modifiers"]["scenarios"].exists()
        else ["None"]
    )
    logger.info(
        "Using outcome modifier scenarios of '%s'", "', '".join(outcome_modifiers_scenarios)
    )
    seir_modifiers_scenarios = (
        cfg["seir_modifiers"]["scenarios"].as_str_seq()
        if cfg["seir_modifiers"].exists() and cfg["seir_modifiers"]["scenarios"].exists()
        else ["None"]
    )
    logger.info(
        "Using SEIR modifier scenarios of '%s'", "', '".join(seir_modifiers_scenarios)
    )

    # Batch system
    batch_system_name = _resolve_batch_system_name(
        kwargs.get("batch_system", None),
        kwargs.get("local", False),
        kwargs.get("slurm", False),
    )
    logger.debug("Resolved batch system name to '%s'", batch_system_name)
    batch_system = get_batch_system(batch_system_name)
    logger.info("Using batch system '%s'", batch_system.name)

    # Job size
    logger.debug(
        "User provided job size of blocks=%s, chains=%s, samples=%s, simulations=%s",
        kwargs.get("blocks"),
        kwargs.get("chains"),
        kwargs.get("samples"),
        kwargs.get("simulations"),
    )
    job_size = batch_system.size_from_jobs_simulations_blocks(
        kwargs.get("blocks"),
        kwargs.get("chains"),
        kwargs.get("samples"),
        kwargs.get("simulations"),
    )
    logger.info("Using job size of %s", job_size)

    # Job time limit
    job_time_limit = kwargs.get("time_limit")
    logger.info("Using job time limit of %s", job_time_limit)

    # Job resources
    nodes, cpus, memory = (kwargs.get(k) for k in ("nodes", "cpus", "memory"))
    logger.debug(
        "User provided job resources of nodes=%s, cpus=%s, memory=%sMB", nodes, cpus, memory
    )
    if not all((nodes, cpus, memory)):
        raise NotImplementedError(
            "Automatic resource estimation is not yet implemented "
            "and one of nodes, cpus, memory is not given."
        )
    job_resources = _job_resources_from_size_and_inference(
        job_size, inference_method, nodes=nodes, cpus=cpus, memory=memory
    )
    logger.info("Using job resources of %s", job_resources)

    # HPC cluster info
    cluster_name = kwargs.get("cluster")
    logger.debug("User provided cluster name of '%s'", cluster_name)
    cluster = (
        get_cluster_info(cluster_name)
        if (batch_system.needs_cluster or cluster_name is not None)
        else None
    )
    logger.info("Using cluster info of %s", cluster)

    # Job config
    job_config = Path(f"config_{job_name}.yml").absolute()
    job_config.write_text(_dump_formatted_yaml(cfg))
    if logger is not None:
        logger.info(
            "Dumped the job config for this batch submission to %s", job_config.absolute()
        )

    # Construct template data
    general_template_data = {
        **kwargs,
        **{
            "user": getuser(),
            "now": now.strftime("%c %Z"),
            "name": name,
            "job_name": job_name,
            "job_size": job_size.model_dump(),
            "job_time_limit": batch_system.format_time_limit(job_time_limit),
            "job_resources_nodes": batch_system.format_nodes(job_resources),
            "job_resources_cpus": batch_system.format_cpus(job_resources),
            "job_resources_memory": batch_system.format_memory(job_resources),
            "cluster": None if cluster is None else cluster.model_dump(),
            "config": job_config,
            "array_capable": _inference_is_array_capable(inference_method),
        },
    }

    # Switch to estimation
    if kwargs.get("estimate", False):
        return _estimate_job_resources(
            name,
            job_name,
            inference_method,
            job_size,
            job_resources,
            job_time_limit,
            batch_system,
            outcome_modifiers_scenarios,
            seir_modifiers_scenarios,
            batch_system.options_from_config_and_cli(
                cfg, kwargs, kwargs.get("verbosity", 0)
            ),
            general_template_data,
            kwargs.get("estimate_runs", 10),
            kwargs.get("estimate_interval", 0.9),
            kwargs.get("verbosity", 0),
            kwargs.get("dry_run", False),
        )

    # Manifest
    if not kwargs.get("skip_manifest", False):
        manifest = write_manifest(
            job_name,
            kwargs.get("flepi_path"),
            kwargs.get("project_path"),
        )
        logger.info("Wrote manifest to '%s'", manifest)
    else:
        if kwargs.get("dry_run", False):
            logger.info("Skipping manifest.")
        else:
            logger.warning("Skipping manifest in non-dry run which is not recommended.")

    # Git checkout
    if not kwargs.get("skip_checkout", False):
        _git_checkout(kwargs.get("project_path"), f"run_{job_name}")
    else:
        if kwargs.get("dry_run", False):
            logger.info("Skipping git checkout.")
        else:
            logger.warning("Skipping git checkout in non-dry run which is not recommended.")

    # Submit jobs
    for outcome_modifiers_scenario, seir_modifiers_scenario in product(
        outcome_modifiers_scenarios, seir_modifiers_scenarios
    ):
        _submit_scenario_job(
            name,
            job_name,
            inference_method,
            job_size,
            batch_system,
            outcome_modifiers_scenario,
            seir_modifiers_scenario,
            batch_system.options_from_config_and_cli(
                cfg, kwargs, kwargs.get("verbosity", 0)
            ),
            general_template_data,
            kwargs.get("verbosity", 0),
            kwargs.get("dry_run", False),
        )
