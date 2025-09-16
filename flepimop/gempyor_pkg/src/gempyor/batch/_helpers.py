__all__ = ()


from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal

from ..constants import _JOB_NAME_REGEX


def _format_resource_bounds(bounds: dict[Literal["cpu", "memory", "time"], float]) -> str:
    """
    Format resource bounds for logging.

    Args:
        bounds: The resource bounds to format.

    Returns:
        The formatted resource bounds.
    """
    fmts = {
        "cpu": "{:.2f}",
        "memory": "{:.2f}MB",
        "time": "{:.2f}s",
    }
    return ", ".join(fmts[k].format(v) for k, v in bounds.items())


def _job_name(name: str | None, timestamp: datetime | None) -> str:
    """
    Generate a unique human readable job name.
    Args:
        name: The config name used as a prefix or `None` for no prefix.
        timestamp: The timestamp used to make the job name unique or `None` to use the
            current UTC timestamp.
    Returns:
        A job name that is unique and intended for use when submitting to slurm.
    Raises:
        ValueError: If `name` does not start with a letter and contains characters other
            than the alphabet, numbers, underscores or dashes.
    Examples:
        >>> from datetime import datetime, timezone
        >>> from gempyor.batch._helpers import _job_name
        >>> _job_name(None, datetime(2024, 1, 1, tzinfo=timezone.utc))
        '20240101T000000'
    """
    timestamp = datetime.now(timezone.utc) if timestamp is None else timestamp
    timestamp = timestamp.strftime("%Y%m%dT%H%M%S")
    if name is not None and not _JOB_NAME_REGEX.match(name):
        raise ValueError(f"The given `name`, '{name}', is not a valid safe name.")
    return f"{name}-{timestamp}" if name else timestamp


def _parse_extra_options(extra: Iterable[str] | None) -> dict[str, str]:
    """
    Parse `--extra` options into a dictionary.

    Args:
        extra: An iterable of extra options to parse if given.

    Returns:
        A dictionary of the parsed extra options.

    Examples:
        >>> from gempyor.batch._helpers import _parse_extra_options
        >>> _parse_extra_options(["abc=def", "ghi=jkl"])
        {'abc': 'def', 'ghi': 'jkl'}
        >>> _parse_extra_options([
        ...     "email=bob@example.com",
        ...     "partition=special-cluster",
        ...     "slack=my-alerts-channel",
        ... ])
        {'email': 'bob@example.com', 'partition': 'special-cluster', 'slack': 'my-alerts-channel'}
        >>> _parse_extra_options(None)
        {}
    """
    if extra is None:
        return {}
    return {
        k: v for k, v in (opt.split("=", 1) if "=" in opt else [opt, ""] for opt in extra)
    }
