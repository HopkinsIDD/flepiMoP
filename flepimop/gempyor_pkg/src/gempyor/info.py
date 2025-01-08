"""
Retrieving static information from developer managed yaml files.

Currently, it includes utilities for handling cluster-specific information, but it can 
be extended to other categories as needed.

Classes:
    Module: Represents a software module with a name and optional version.
    PathExport: Represents a path export with a path, prepend flag, and error handling.
    Cluster: Represents a cluster with a name, list of modules, and list of path 
        exports.

Functions:
    get_cluster_info: Retrieves cluster-specific information.

Examples:
    >>> from pprint import pprint
    >>> from gempyor.info import get_cluster_info
    >>> cluster_info = get_cluster_info("longleaf")
    >>> cluster_info.name
    'longleaf'
    >>> pprint(cluster_info.modules)
    [Module(name='gcc', version='9.1.0'),
    Module(name='anaconda', version='2023.03'),
    Module(name='git', version=None),
    Module(name='aws', version=None)]
"""

__all__ = ["Cluster", "Module", "PathExport", "get_cluster_info"]


import os
from pathlib import Path
import re
from socket import getfqdn
from typing import Pattern, TypeVar

from pydantic import BaseModel
import yaml


class Module(BaseModel):
    """
    A model representing a module to load.

    Attributes:
        name: The name of the module to load.
        version: The specific version of the module to load if there is one.

    See Also:
        [Lmod](https://lmod.readthedocs.io/en/latest/)
    """

    name: str
    version: str | None = None


class PathExport(BaseModel):
    """
    A model representing the export path configuration.

    Attributes:
        path: The file system path of the path to add to the `$PATH` environment
            variable.
        prepend: A flag indicating whether to prepend additional information to the
            `$PATH` environment variable.
        error_if_missing: A flag indicating whether to raise an error if the path is
            missing.
    """

    path: Path
    prepend: bool = True
    error_if_missing: bool = False


class Cluster(BaseModel):
    """
    A model representing a cluster configuration.

    Attributes:
        name: The name of the cluster.
        modules: A list of modules associated with the cluster.
        path_exports: A list of path exports for the cluster.
    """

    name: str
    modules: list[Module] = []
    path_exports: list[PathExport] = []


_BASE_MODEL_TYPE = TypeVar("T", bound=BaseModel)


_CLUSTER_FQDN_REGEXES: tuple[tuple[str, Pattern], ...] = (
    ("longleaf", re.compile(r"^longleaf\-login[0-9]+\.its\.unc\.edu$")),
    ("rockfish", re.compile(r"^login[0-9]+\.cm\.cluster$")),
)


def _get_info(
    category: str, name: str, model: type[_BASE_MODEL_TYPE], flepi_path: os.PathLike | None
) -> _BASE_MODEL_TYPE:
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

    Returns:
        An object containing the information about the `name` cluster.

    Examples:
        >>> from gempyor.info import get_cluster_info
        >>> cluster_info = get_cluster_info("longleaf")
        >>> cluster_info.name
        'longleaf'
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
