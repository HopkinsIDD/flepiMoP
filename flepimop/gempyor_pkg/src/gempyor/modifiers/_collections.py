""""""

__all__: tuple[str, ...] = ()

from typing import Annotated, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator

from ._periodic_modifier import PeriodicModifier
from ._stacked_modifier import StackedModifier


Modifier = Annotated[
    Union[
        PeriodicModifier,
        StackedModifier,
    ],
    Field(discriminator="method"),
]


class ModifiersCollection(BaseModel):
    """A collection of modifiers for a given set of scenarios."""

    scenarios: list[str]
    modifiers: list[Modifier]
    stacked_scenarios_map: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("modifiers", mode="before")
    @classmethod
    def _extract_name(cls, value: Any) -> Any:
        if isinstance(value, dict) and len(keys := value.keys()) == 1:
            return list(value.values())[0] | {"name": list(keys)[0]}
        return value
    
    @model_validator(mode="after")
    def _rewrite_stacked_modifiers(self) -> "ModifiersCollection":
        """Re-write stacked modifiers to ModifiersCollection to maintain backwards compatibility."""
        unpacked_modifiers = []
        stacked_scenarios_map = self.stacked_scenarios_map.copy()

        for modifier in self.modifiers:
            if isinstance(modifier, StackedModifier):
                stacked_scenarios_map[modifier.name] = modifier.scenarios 
                unpacked_modifiers.extend(modifier.modifiers)

            else:
                unpacked_modifiers.append(modifier)
        
        self.modifiers = unpacked_modifiers
        self.stacked_scenarios_map = stacked_scenarios_map
        
        return self
    




