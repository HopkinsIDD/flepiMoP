"""Interface definitions for initial condition objects."""

__all__: tuple[str, ...] = ()

from abc import ABC, abstractmethod
from typing import Any

import confuse
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from ..compartments import Compartments
from ..model_meta import ModelMeta
from ..parameters import Parameters
from ..subpopulation_structure import SubpopulationStructure
from ..time_setup import TimeSetup
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
        ignore_population_checks: If `True`, population checks will be ignored when
            validating initial conditions against the subpopulation structure.
            Defaults to `False`.
        meta: Either an instance of `gempyor.model_meta.ModelMeta` or `None` for
            interacting with the filesystem.
        time_setup: Meta information about the time range of the simulation as
            represented by `gempyor.time_setup.TimeSetup` or `None` if not available.

    """

    method: str
    ignore_population_checks: bool = False
    meta: ModelMeta | None = None
    time_setup: TimeSetup | None = None

    @classmethod
    def from_confuse_config(
        cls, config: confuse.Subview, **kwargs: Any
    ) -> "InitialConditionsABC":
        """
        Create a `SubpopulationStructure` instance from a confuse configuration view.

        Args:
            config: A configuration view containing the subpopulation
                configuration.
            **kwargs: Additional keyword arguments to pass to the model validation.

        Returns:
            An instance of `SubpopulationStructure` created from a confuse
            configuration view.
        """
        try:
            conf = dict(config.get())
        except confuse.NotFoundError:
            conf = {"method": "Default"}
        return cls.model_validate(conf | kwargs)

    @abstractmethod
    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
        parameters: Parameters,
        p_draw: npt.NDArray[np.float64],
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
        parameters: Parameters,
        p_draw: npt.NDArray[np.float64],
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
            parameters: The parameters object containing a representation of simulation
                parameters.
            p_draw: A numpy array of floats representing the parameter draws for the
                simulation.

        Returns:
            A numpy array of initial conditions for the simulation.
        """
        y0 = self.create_initial_conditions(
            sim_id, compartments, subpopulation_structure, parameters, p_draw
        )
        check_population(
            y0,
            subpopulation_structure.subpop_names,
            subpopulation_structure.subpop_pop,
            ignore_population_checks=self.ignore_population_checks,
        )
        return y0
