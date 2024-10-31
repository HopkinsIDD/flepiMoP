"""
Internal utilities for interacting with slurm.
"""

__all__ = []


from pathlib import Path
from shlex import quote
import subprocess
from typing import Any, Literal

from .logging import get_script_logger
from .utils import _shutil_which


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
    options = [
        f"{'-' if len(k) == 1 else '--'}{k}={quote(str(v))}" for k, v in options.items()
    ]
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
