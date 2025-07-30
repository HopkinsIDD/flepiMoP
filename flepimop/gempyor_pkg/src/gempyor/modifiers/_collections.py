""""""

__all__: tuple[str, ...] = ()

import numpy as np
import collections
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

    # Added these to enable .appy() delegation
    sum_parameters: list[str] = []
    reduction_product_parameters: list[str] = []

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

                for inner_modifier in modifier.modifiers:
                    inner_modifier.scenarios = modifier.scenarios
                    unpacked_modifiers.append(inner_modifier)
            else:
                unpacked_modifiers.append(modifier)

        self.modifiers = unpacked_modifiers
        self.stacked_scenarios_map = stacked_scenarios_map
        return self

    def apply(
        self, parameters: dict[str, np.ndarray], scenario: str, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Filters modifiers for the specified scenario and then
        delegates calculations to modifier's own .apply() methods.

        Args:
            parameters: A dictionary of parameter names and their ndarray values.
            scenario: The specific scenario to apply modifiers for.
            rng: A random number generator to sample from modifier distribution

        Returns:
            A dictionary with the modified parameter values.
        """

        modified_parameters = parameters.copy()

        # Filter for modifiers relevant to given scenario
        applicable_modifiers = [
            m for m in self.modifiers if not m.scenarios or scenario in m.scenarios
        ]

        # Group modifiers by the parameter they affect
        modifiers_by_param = collections.defaultdict(list)
        for m in applicable_modifiers:
            modifiers_by_param[m.parameter].append(m)

        # For each parameter, chain the .apply() calls from its parent modifier
        for param, mods in modifiers_by_param.items():
            current_value = modified_parameters[param]
            for modifier in mods:
                if modifier.parameter in self.sum_parameters:
                    apply_method = "sum"
                elif modifier.parameter in self.reduction_product_parameters:
                    apply_method = "reduction_product"
                else:
                    apply_method = "product"

                # Get the modification value by sampling the distribution and delegate application
                modification_value = modifier.value.sample(size=1, rng=rng)[
                    0
                ]  # TODO: change to new callable () after rebasing to head of dev

                current_value = modifier.apply(
                    parameter=current_value,
                    modification=modification_value,
                    method=apply_method,
                )
            modified_parameters[param] = current_value
        return modified_parameters
