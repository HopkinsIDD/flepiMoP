"""Interface definitions for initial condition objects."""

__all__: tuple[str, ...] = ()

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, overload

import confuse
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, field_validator

from ..compartments import Compartments
from ..subpopulation_structure import SubpopulationStructure
from ._utils import check_population


class InitialConditionsABC(ABC, BaseModel):
    """
    Abstract base class for initial conditions.

    This class defines the interface for initial conditions implementations, which
    should provide a method to generate initial conditions for a simulation based on
    the provided configuration.

    Attributes:
        method: The method name used for generating initial conditions, used as a type
            discriminator when instantiating initial conditions from a configuration.
        path_prefix: The path prefix for the geodata and mobility files, defaults to
            the current working directory.
    """

    method: str
    path_prefix: Path

    @overload
    @classmethod
    def _validate_path_prefix(cls, value: Path | str | None) -> Path: ...

    @overload
    @classmethod
    def _validate_path_prefix(cls, value: Any) -> Any: ...

    @field_validator("path_prefix", mode="before")
    @classmethod
    def _validate_path_prefix(cls, value: Path | str | None | Any) -> Path | Any:
        """
        Validate and convert the path prefix to a Path object.

        Args:
            value: The path prefix to validate.

        Returns:
            A Path object representing the path prefix if `value` is `None`, a string,
            or a Path otherwise returns the value unchanged.
        """
        if value is None:
            return Path.cwd()
        if isinstance(value, str | Path):
            return Path(value).resolve()
        return value

    @classmethod
    def from_confuse_config(
        cls, config: confuse.Subview, path_prefix: Path | str | None = None
    ) -> "InitialConditionsABC":
        """
        Create a `SubpopulationStructure` instance from a confuse configuration view.

        Args:
            config: A configuration view containing the subpopulation
                configuration.
            path_prefix: The path prefix for the geodata and mobility files or `None` to
                use the current working directory.

        Returns:
            An instance of `SubpopulationStructure`.
        """
        return cls.model_validate(dict(config.get()) | {"path_prefix": path_prefix})

    @abstractmethod
    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
    ) -> npt.NDArray[np.float64]:
        """
        Produce an array of initial conditions from the configuration.

        This method should be implemented by subclasses to provide specific initial
        conditions based on the configuration.

        Args:
            sim_id: An integer id for the simulation being ran.
            compartments: The compartments object.
            subpopulation_structure: The subpopulation structure object.

        Returns:
            A numpy array of initial conditions for the simulation.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses to provide specific "
            "initial conditions."
        )

    def get_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
    ) -> npt.NDArray[np.float64]:
        """
        Get initial conditions for the simulation.

        This method is a wrapper around `create_initial_conditions` to maintain
        compatibility with existing code and provide a check on the initial
        conditions against the subpopulation structure.

        Args:
            sim_id: An integer id for the simulation being ran.
            compartments: The compartments object.
            subpopulation_structure: The subpopulation structure object.

        Returns:
            A numpy array of initial conditions for the simulation.
        """
        y0 = self.create_initial_conditions(sim_id, compartments, subpopulation_structure)
        check_population(
            y0,
            subpopulation_structure.subpop_names,
            subpopulation_structure.subpop_pop,
            ignore_population_checks=getattr(self, "ignore_population_checks", False),
        )
        return y0
