"""ABC for modifiers in the gempyor package."""

__all__: tuple[str, ...] = ()

from abc import ABC

from pydantic import BaseModel


class ModifierABC(ABC, BaseModel):
    """
    Base class for modifiers in the gempyor package.

    This class serves as an abstract base class (ABC) for all modifiers, providing
    a common interface and structure for modifier implementations.
    """

    method: str
