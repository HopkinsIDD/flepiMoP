"""Management for initial conditions plugins."""

__all__: tuple[str, ...] = ()


from pathlib import Path
from typing import Any, Literal, get_args, get_origin
import warnings

import confuse

from ..warnings import ConfigurationWarning
from ._base import InitialConditionsABC


_initial_conditions_plugins: dict[str, InitialConditionsABC] = {}


def register_initial_conditions_plugin(
    plugin: type[InitialConditionsABC], force: bool = False
) -> None:
    """
    Register an initial conditions plugin class.

    Args:
        plugin: The initial conditions class to register. It should subclass
            `InitialConditionsABC` and have an attribute 'method' that is a `Literal`
            identifying the plugin.
        force: Whether this plugin should override previously registered plugins in the
            case of conflicting 'method' names.

    Raises:
        ValueError: If `plugin` is not an instance of the `InitialConditionsABC` type.
        ValueError: If `plugin` does not have a 'method' attribute.
        ValueError: If the `plugin`'s 'method' attribute is not a `Literal`.
        ValueError: If there is a plugin already registered with one of the values from
            the 'method' attribute and `force` is `False`.

    """
    # if not isinstance(plugin, type[InitialConditionsABC]):
    if not issubclass(plugin, InitialConditionsABC):
        raise ValueError(
            "Initial conditions plugins must subclass "
            f"`InitialConditionsABC`, instead was given {plugin}."
        )
    if (method := plugin.model_fields.get("method")) is None:
        raise ValueError(
            "An initial conditions plugin must have a 'method' attribute, "
            f"instead the plugin given, {plugin}, has attributes: "
            f"{', '.join(plugin.model_fields.keys())}."
        )
    if (origin := get_origin(method.annotation)) is not Literal:
        raise ValueError(
            f"The initial conditions plugin, {plugin}, has an 'method' attribute "
            f"but it is not a Literal type instead it is {origin}."
        )
    for arg in get_args(method.annotation):
        if not force and arg in _initial_conditions_plugins:
            raise ValueError(
                "There is already an initial conditions plugin "
                f"with the method of '{arg}' registered."
            )
        _initial_conditions_plugins[arg] = plugin


def initial_conditions_from_plugin(
    config: confuse.ConfigView, path_prefix: Path | str | None = None, **kwargs: Any
) -> InitialConditionsABC:
    """
    Create an initial conditions object from the registered plugins.

    Args:
        config: The relevant confuse configuration view for the initial conditions.
        path_prefix: The path prefix.
        **kwargs: Additional keyword arguments provided to the `from_confuse_config`
            class method.

    Returns:
        An instance of an initial conditions class, subclassing `InitialConditionsABC`,
        corresponding to the 'method' found in the given configuration.

    Raises:
        ValueError: If `config` does not contain a 'method' key.
        ValueError: If the 'method' key from `config` is not found in the registered
            plugin options.

    """
    if not config["method"].exists():
        warnings.warn(
            "Initial conditions plugin 'method' was not specified, assuming 'Default'.",
            ConfigurationWarning,
        )
        method = "Default"
    else:
        method = config["method"].as_str()
    if (initial_conditions_class := _initial_conditions_plugins.get(method)) is None:
        raise ValueError(
            "There is no initial conditions plugin matching "
            f"'method' name of '{method}'. Instead the available "
            f"options are: {', '.join(sorted(_initial_conditions_plugins.keys()))}."
        )
    return initial_conditions_class.from_confuse_config(
        config, path_prefix=path_prefix, **kwargs
    )
