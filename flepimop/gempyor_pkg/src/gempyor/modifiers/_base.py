"""ABC for modifiers in the gempyor package."""

__all__: tuple[str, ...] = ()

import numpy as np
import pandas as pd
import typing
from abc import ABC, abstractmethod

from pydantic import BaseModel


class ModifierABC(ABC, BaseModel):
    """
    Base class for modifiers in the gempyor package.

    This class serves as an abstract base class (ABC) for all modifiers, providing
    a common interface and structure for modifier implementations.
    """

    method: str

    def __init__(self, *, name):
        self.name = name

    @abstractmethod
    def apply(
        self,
        parameter: np.ndarray,
        modification: typing.Union[pd.DataFrame, float],
        method: str = "product",
    ) -> np.ndarray:
        pass
