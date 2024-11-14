__all__ = ["Cluster", "Module", "PathExport", "get_cluster_info"]


import os
from pathlib import Path
import re
from socket import getfqdn
from typing import Pattern, TypeVar

from pydantic import BaseModel
import yaml


class Module(BaseModel):
    name: str
    version: str | None = None


class PathExport(BaseModel):
    path: Path
    prepend: bool = True
    error_if_missing: bool = False


class Cluster(BaseModel):
    name: str
    modules: list[Module] = []
    path_exports: list[PathExport] = []


T = TypeVar("T", bound=BaseModel)


_CLUSTER_FQDN_REGEXES: tuple[tuple[str, Pattern], ...] = (
    ("longleaf", re.compile(r"^longleaf\-login[0-9]+\.its\.unc\.edu$")),
    ("rockfish", re.compile(r"^login[0-9]+\.cm\.cluster$")),
)


def _get_info(
    category: str, name: str, model: type[T], flepi_path: os.PathLike | None
) -> T:
    """
    Get and parse an information yaml file.

    This function is a light wrapper around reading and parsing yaml files located in
    `$FLEPI_PATH/info`.

    Args:
        category: The category of info to get, corresponds to a subdirectory in
            `$FLEPI_PATH/info`.
        name: The name of the info to get, corresponds to the name of a yaml file and
            is usually a human readable short name.
        model: The pydantic class to parse the info file with, determines the return
            type.
        flepi_path: Either a path like determine the directory to look for the info
            directory in or `None` to use the `FLEPI_PATH` environment variable.

    Returns:
        An instance of `model` with the contained info found and parsed.
    """
    flepi_path = Path(os.getenv("FLEPI_PATH") if flepi_path is None else flepi_path)
    info = flepi_path / "info" / category / f"{name}.yml"
    if not info.exists() or not info.is_file():
        raise ValueError(
            f"Was expecting an information yaml at {info.absolute()}, "
            "but either does not exist or is not a file."
        )
    return model.model_validate(yaml.safe_load(info.read_text()))


def get_cluster_info(name: str | None, flepi_path: os.PathLike | None = None) -> Cluster:
    """
    Get cluster specific info.

    Args:
        name: The name of the cluster to pull information for. Currently only 'longleaf'
            and 'rockfish' are supported or `None` to infer from the FQDN.
        flepi_path: Either a path like determine the directory to look for the info
            directory in or `None` to use the `FLEPI_PATH` environment variable.

    Returns
        An object containing the information about the `name` cluster.
    """
    name = _infer_cluster_from_fqdn() if name is None else name
    return _get_info("cluster", name, Cluster, flepi_path)


def _infer_cluster_from_fqdn() -> str:
    """
    Infer the cluster name from the FQDN.

    Returns:
        The name of the cluster inferred from the FQDN.

    Raises:
        ValueError: If the value of `socket.getfqdn()` does not match an expected regex.
    """
    fqdn = getfqdn()
    for cluster, regex in _CLUSTER_FQDN_REGEXES:
        if regex.match(fqdn):
            return cluster
    raise ValueError(f"The fqdn, '{fqdn}', does not match any of the expected clusters.")
