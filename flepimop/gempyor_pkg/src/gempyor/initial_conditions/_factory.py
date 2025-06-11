"""Create initial conditions from configuration or plugin with a factory function."""

__all__: tuple[str, ...] = ()


import confuse

from ..utils import search_and_import_plugins_class
from ._initial_conditions import InitialConditions


def initial_conditions_factory(
    config: confuse.ConfigView, path_prefix: str = "."
) -> InitialConditions:
    """
    Create an initial conditions object from config or plugin.

    Args:
        config: The configuration object containing initial conditions settings.
        path_prefix: The prefix path for file paths in the configuration.

    Returns:
        An instance of the `InitialConditions` class created from configuration or a
        plugin subclass.
    """
    if config is not None and "method" in config.keys():
        if config["method"].as_str() == "plugin":
            klass = search_and_import_plugins_class(
                plugin_file_path=config["plugin_file_path"].as_str(),
                class_name="InitialConditions",
                config=config,
                path_prefix=path_prefix,
            )
            return klass
    return InitialConditions(config, path_prefix=path_prefix)
