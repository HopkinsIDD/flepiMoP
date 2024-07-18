import os
from pathlib import Path
import sys
from typing import Any, Literal

from .io import DirectoryIODriver, resolve_paths

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


__all__ = ["ModelOutput"]


class ModelOutput:
    """
    Interface for interacting with a model output directory.

    TODO: Describe the output directory structure here at a high level.

    Attributes:
        directory: A `Path` of the directory being described.
    """

    def __init__(
        self,
        directory: str | bytes | os.PathLike | Path,
        directory_drivers: dict[
            Literal["hnpi", "hosp", "hpar", "init", "llik", "snpi", "spar"],
            DirectoryIODriver,
        ] = {},
        default_directory_driver: DirectoryIODriver = DirectoryIODriver(),
    ) -> None:
        """
        Initializes an instance based on a generic set of parameters.

        Args:
            directory: The directory to center this instance around.
            directory_drivers: A mapping describing which `DirectoryIODriver` to use for
                each output type.
            default_directory_driver: The default driver to use if not specified in the
                `directory_drivers` argument.
        """
        self.directory = resolve_paths(directory)
        if extra_keys := set(directory_drivers.keys()) - {
            "hnpi",
            "hosp",
            "hpar",
            "init",
            "llik",
            "snpi",
            "spar",
        }:
            raise ValueError(
                (
                    "Unexpected keys were given in the 'directory_drivers' "
                    f"argument: {', '.join(extra_keys)}."
                )
            )
        self._directory_drivers = directory_drivers
        self._default_directory_driver = default_directory_driver

    @classmethod
    def from_existing_directory(
        cls, directory: str | bytes | os.PathLike | Path
    ) -> Self:
        """
        Create an instance from an existing directory and infer the appropriate driver.

        Args:
            directory: An existing directory to center this instance around.

        Raises:
            NotImplementedError: This functionality is not implemented yet and this
                documentation serves only as a spec for the moment.
        """
        directory = resolve_paths(directory)
        if not directory.is_dir():
            raise ValueError(
                f"The 'directory' argument given, '{directory}', is not a directory."
            )
        raise NotImplementedError
