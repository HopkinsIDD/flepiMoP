"""Internal utilities for multiprocessing in gempyor."""

from pathlib import Path
import multiprocessing
import importlib
import sys


def _multiprocessing_pool_initializer(custom_initial_conditions: list[Path]) -> None:
    """
    Initialize the multiprocessing pool.

    This function is called when a new worker process is started.
    It can be used to set up any necessary state for the worker.
    """
    if multiprocessing.get_start_method() in {"forkserver", "spawn"}:
        for module_path in custom_initial_conditions:
            module_name = module_path.stem
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)


def _pool(
    processes: int | None = None, custom_plugins: list[Path] | None = None
) -> multiprocessing.Pool:
    """
    Create a multiprocessing pool with the specified number of processes.

    Args:
        processes: The number of worker processes to use. If None, the number of
            processes will be set to the number of CPU cores available, determined by
            `os.process_cpu_count()`. See `multiprocessing.pool.Pool` for more details.
        custom_plugins: A list of custom plugins to be loaded in the worker processes.

    Returns:
        A multiprocessing Pool object.
    """
    additional_kwargs = (
        {}
        if custom_plugins is None
        else {
            "initializer": _multiprocessing_pool_initializer,
            "initargs": (custom_plugins,),
        }
    )
    return multiprocessing.Pool(processes=processes, **additional_kwargs)
