"""
Interact with a standardized `model_output/` directory.

This module provides an interface for working with a `model_output/` directory in a 
standardized way. The main entry point is the `ModelOutput` class which abstracts away 
specific details about directory layout and file format/type from end users.  
"""

__all__ = ["ModelOutput"]


import os
from pathlib import Path
import re
import sys
from typing import Any, Literal

from .io import DirectoryIODriver, resolve_paths

if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


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
        name: None | str = None,
        seir_modifier: None | str = None,
        outcome_modifier: None | str = None,
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
            setup_name:
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
        if wrong_value_types := [
            k
            for k, v in directory_drivers.items()
            if not isinstance(v, DirectoryIODriver)
        ]:
            raise ValueError(
                (
                    "The following keys have values that are not an instance of "
                    "DirectoryIODriver in the 'directory_drivers' argument: "
                    f"{', '.join(wrong_value_types)}."
                )
            )
        self._directory_drivers = directory_drivers
        self._default_directory_driver = default_directory_driver

    @classmethod
    def from_existing_directory(
        cls,
        directory: str | bytes | os.PathLike | Path,
        name: None | str = None,
        seir_modifier: None | str = None,
        outcome_modifier: None | str = None,
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

    @staticmethod
    def _parse_directory(
        directory: str | bytes | os.PathLike | Path,
        name: None | str,
        seir_modifier: None | str,
        outcome_modifier: None | str,
    ) -> tuple[Path, str, str, str]:
        """
        Extract the relevant information from a directory with supplemental help.

        Args:
            directory: The directory to attempt parsing relevant info from.
            name: The scenario name being considered, or `None` to be inferred from
                `directory`.
            seir_modifier: The SEIR scenario modifier being considered, or `None` to be
                inferred from `directory`.
            outcome_modifier: The outcome scenario modifier being considered, or `None`
                to be inferred from `directory`.

        Returns:
            The parsed `directory`, `name`, `seir_modifier`, and `outcome_modifier` as a
            tuple of appropriate types.
        """
        directory = resolve_paths(directory)
        pattern = (
            rf"^(?i).*/({name if name else ".*"})\_"
            rf"{seir_modifier if seir_modifier else ".*"}\_"
            rf"{outcome_modifier if outcome_modifier else ".*"}$"
        )
        m = re.match(pattern, str(directory))
        raise NotImplementedError

    def rglob(
        self,
        pattern: str,
        output_type: Literal["hnpi", "hosp", "hpar", "init", "llik", "snpi", "spar"],
    ) -> list[Path]:
        """
        Glob the given pattern recursively for a given output type.

        This method will query against the file system directly, which defeats the
        purpose of abstracting away file details. This method is meant for use in only
        specific use cases or internally.

        Args:
            pattern: A string pattern to use when searching.
            output_type: The output type to search under.

        Returns:
            Paths that match the given `pattern` and `output_type`.

        See also:
            [`pathlib.Path.rglob`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.rglob)

        Raises:
            NotImplementedError: This functionality is not implemented yet and this
                documentation serves only as a spec for the moment.
        """
        raise NotImplementedError

    def write_output_type(
        self,
        obj: Any,
        output_type: Literal["hnpi", "hosp", "hpar", "init", "llik", "snpi", "spar"],
        inference: Literal[None, "chimeric", "global"],
        inference_step: Literal[None, "final", "intermediate"],
    ) -> None:
        """
        Write an object to an output type using the appropriate driver.

        This method largely delegates to the `DirectoryIODriver` specified for the given
        output type in either `directory_drivers` or `default_directory_driver` given
        in initialization. While this method accepts an `obj` of any type, the directory
        driver may not, please refer to the appropriate documentation for details.

        Args:
            obj: The object to write.
            output_type: The output type to write.
            inference: The type of inference parameter this object belongs to, if `None`
                it is assumed this is a 'simulation' mode run.
            inference_step: The inference step this object belongs to, if `None` it is
                assumed this is a 'simulation' mode run.

        Returns:
            None

        Raises:
            NotImplementedError: This functionality is not implemented yet and this
                documentation serves only as a spec for the moment.
        """
        raise NotImplementedError
