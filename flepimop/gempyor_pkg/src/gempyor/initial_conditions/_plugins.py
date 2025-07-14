"""Management for initial conditions plugins."""

__all__: tuple[str, ...] = ()


import importlib
from pathlib import Path
import sys
from typing import Any, Final, Literal, get_args, get_origin
import warnings

import confuse

from ..warnings import ConfigurationWarning
from ._base import InitialConditionsABC


_STANDARD_INITIAL_CONDITION_PLUGINS: Final[set[str]] = {
    "Default",
    "SetInitialConditions",
    "SetInitialConditionsFolderDraw",
    "FromFile",
    "InitialConditionsFolderDraw",
}

_custom_initial_conditions_plugins: list[Path] = []

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
    global _initial_conditions_plugins
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
    global _custom_initial_conditions_plugins, _initial_conditions_plugins
    if not config["method"].exists():
        warnings.warn(
            "Initial conditions plugin 'method' was not specified, assuming 'Default'.",
            ConfigurationWarning,
        )
        method = "Default"
    else:
        method = config["method"].as_str()
    if config["module"].exists():
        module_path = Path(config["module"].as_str())
        before_len = len(_initial_conditions_plugins)
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path.stem] = module
        spec.loader.exec_module(module)
        if len(_initial_conditions_plugins) == before_len:
            raise ValueError(
                "No initial conditions plugin was registered from the module "
                f"'{module_path}'. Was `register_initial_conditions_plugin` "
                "called by the custom plugin?"
            )
        _custom_initial_conditions_plugins.append(module_path)
    if (initial_conditions_class := _initial_conditions_plugins.get(method)) is None:
        raise ValueError(
            "There is no initial conditions plugin matching "
            f"'method' name of '{method}'. Instead the available "
            f"options are: {', '.join(sorted(_initial_conditions_plugins.keys()))}."
        )
    return initial_conditions_class.from_confuse_config(
        config, path_prefix=path_prefix, **kwargs
    )


def _get_custom_initial_conditions_plugins() -> list[Path]:
    """
    Get the list of custom initial conditions plugins.

    Returns:
        A list of `Path` objects representing the custom initial conditions plugins.

    """
    global _custom_initial_conditions_plugins
    return _custom_initial_conditions_plugins


def _reset_initial_conditions_plugins() -> None:
    """
    Reset the registered initial conditions plugins.

    This is useful for testing purposes to ensure a clean state.

    Examples:
        >>> from pprint import pprint
        >>> from typing import Literal
        >>> from gempyor.initial_conditions import (
        ...     DefaultInitialConditions,
        ...     register_initial_conditions_plugin,
        ... )
        >>> from gempyor.initial_conditions._plugins import (
        ...     _initial_conditions_plugins,
        ...     _reset_initial_conditions_plugins,
        ... )
        >>> pprint(list(_initial_conditions_plugins.keys()))
        ['Default',
         'SetInitialConditions',
         'SetInitialConditionsFolderDraw',
         'FromFile',
         'InitialConditionsFolderDraw']
        >>> class DefaultTwoInitialConditions(DefaultInitialConditions):
        ...     method: Literal["DefaultTwo"] = "DefaultTwo"
        >>> register_initial_conditions_plugin(DefaultTwoInitialConditions)
        >>> pprint(list(_initial_conditions_plugins.keys()))
        ['Default',
         'SetInitialConditions',
         'SetInitialConditionsFolderDraw',
         'FromFile',
         'InitialConditionsFolderDraw',
         'DefaultTwo']
        >>> _reset_initial_conditions_plugins() is None
        True
        >>> pprint(list(_initial_conditions_plugins.keys()))
        ['Default',
         'SetInitialConditions',
         'SetInitialConditionsFolderDraw',
         'FromFile',
         'InitialConditionsFolderDraw']
    """
    global _custom_initial_conditions_plugins, _initial_conditions_plugins
    for plugin in (
        set(_initial_conditions_plugins.keys()) - _STANDARD_INITIAL_CONDITION_PLUGINS
    ):
        del _initial_conditions_plugins[plugin]
    _custom_initial_conditions_plugins = []
