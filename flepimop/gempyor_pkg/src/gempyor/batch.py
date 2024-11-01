"""
Functionality for creating and submitting batch jobs.

This module provides functionality for required for batch jobs, including creating 
metadata and job size calculations for example.
"""

__all__ = ["JobSize", "write_manifest"]


from dataclasses import dataclass
import json
import math
from pathlib import Path
from shlex import quote
import subprocess
import sys
from typing import Any, Literal

from ._jinja import _render_template_to_file, _render_template_to_temp_file
from .logging import get_script_logger
from .utils import _format_cli_options, _git_head, _shutil_which


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
        >>> from pathlib import Path
        >>> flepi_path = Path("~/Desktop/GitHub/HopkinsIDD/flepiMoP").expanduser()
        >>> project_path = Path("~/Desktop/GitHub/HopkinsIDD/flepimop_sample").expanduser()
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


def _sbatch(
    script: Path,
    environment_variables: dict[str, Any] | Literal["all", "nil", "none"],
    options: dict[str, Any],
    verbosity: int | None,
    dry_run: bool,
) -> None:
    """
    Submit a job to slurm via the `sbatch` command.

    Args:
        script: The batch file to submit to slurm.
        environment_variables: Environment variables to pass to the job via the
            '--export' option. Keys correspond to the variable name and the value is
            the variable value. All values are coerced to a string and then escaped.
            Or can be a literal for one of the '--export' option's special values.
        options: Options to pass when calling sbatch. Keys correspond to the option
            name and the value is the option value. Options can be provided as either
            the long or short name, but this function will not be able to determine that
            these are duplicates so only using long names is recommended.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit a job to slurm.

    Raises:
        ValueError: If 'export' is given as a key in `options` instead of using
            `environment_variables`.

    Examples:
        >>> from pathlib import Path
        >>> _sbatch(
        ...     Path("my_batch_script.sbatch"),
        ...     {"VAR1": 1, "VAR2": "true"},
        ...     {"J": "My job name", "output": Path("out.log")},
        ...     3,
        ...     True,
        ... )
        2024-10-31 09:26:48,361:DEBUG:gempyor._slurm> Using batch script '/my_batch_script.sbatch' to submit to slurm.
        2024-10-31 09:26:48,363:INFO:gempyor._slurm> If not dry mode would have submitted to slurm with: /usr/bin/sbatch --export=VAR1=1,VAR2=true -J='My job name' --output=out.log /my_batch_script.sbatch.

    See Also:
        [`sbatch`'s documentation](https://slurm.schedmd.com/sbatch.html)
    """
    if "export" in options:
        raise ValueError(
            "Found 'export' in `options`, please use `environment_variables` instead."
        )

    if verbosity is not None:
        logger = get_script_logger(__name__, verbosity)
        logger.debug("Using batch script '%s' to submit to slurm", script.absolute())

    if isinstance(environment_variables, dict):
        env_vars = [f"{k}={quote(str(v))}" for k, v in environment_variables.items()]
        export = "--export=" + ",".join(env_vars)
    else:
        export = f"--export={environment_variables.upper()}"
    options = _format_cli_options(options)
    sbatch_cmd = _shutil_which("sbatch")
    if len(export) > 9:
        cmd_args = [sbatch_cmd, export] + options + [str(script.absolute())]
    else:
        cmd_args = [sbatch_cmd] + options + [str(script.absolute())]

    if dry_run:
        if verbosity is not None:
            logger.info(
                "If not dry mode would have submitted to slurm with: %s",
                " ".join(cmd_args),
            )
        return

    if verbosity is not None:
        logger.info("Submitting to slurm with: %s", " ".join(cmd_args))
    process = subprocess.run(cmd_args, check=True, capture_output=True)
    if verbosity is not None:
        if process.stdout:
            logger.debug(
                "Captured stdout from sbatch submission: %s", process.stdout.decode()
            )
        if process.stderr:
            logger.error(
                "Captured stderr from sbatch submission: %s", process.stderr.decode()
            )


def _sbatch_template(
    template: str,
    script: Path | None,
    template_data: dict[str, Any],
    environment_variables: dict[str, Any] | Literal["all", "nil", "none"],
    options: dict[str, Any],
    verbosity: int | None,
    dry_run: bool,
) -> Path:
    """
    Submit a job from a template to slurm via the `sbatch` command.

    Args:
        template: The name of the template to use for rendering the sbatch script.
        script: Either the path of where to save the rendered sbatch script to or `None`
            for a tmp file.
        template_data: Data accessible to the template when rendering.
        environment_variables: Environment variables to pass to the job via the
            '--export' option. Keys correspond to the variable name and the value is
            the variable value. All values are coerced to a string and then escaped.
            Or can be a literal for one of the '--export' option's special values.
        options: Options to pass when calling sbatch. Keys correspond to the option
            name and the value is the option value. Options can be provided as either
            the long or short name, but this function will not be able to determine that
            these are duplicates so only using long names is recommended.
        verbosity: A integer verbosity level to enable logging or `None` for no logging.
        dry_run: A boolean indicating if this is a dry run or not, if set to `True` this
            function will not actually submit a job to slurm.

    Returns:
        The rendered sbatch script, either a tmp file if `script` is None otherwise
        `script` is returned.

    Raises:
        ValueError: If 'options' is found in `template_data` and `options` is not empty.
    """
    template_data["interpreter"] = template_data.get("interpreter", "bash")
    if "options" in template_data and options:
        raise ValueError(
            "Found 'options' in `template_data` but `options` is not empty, can only one."
        )
    template_data["options"] = _format_cli_options(template_data.get("options", options))
    if script is None:
        script = _render_template_to_temp_file(template, template_data, suffix=".sbatch")
    else:
        _render_template_to_file(template, template_data, script)
    _sbatch(script, environment_variables, {}, verbosity, dry_run)
    return script
