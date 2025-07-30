""""""

__all__: tuple[str, ...] = ()

from typing import Literal
from pydantic import BaseModel


class StackedModifier(BaseModel):
    """A container for stacking multiple modifiers to be applied later to specific scenarios."""

    method: Literal["stacked"] = "stacked"
    name: str
    scenarios: list[str]
    modifiers: list["Modifier"]