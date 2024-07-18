from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any


__all__ = ["resolve_paths", "DirectoryIODriver", "infer_directory_driver_from_suffix"]


def resolve_paths(
    paths: str | bytes | os.PathLike | Path | list[str | bytes | os.PathLike | Path],
    resolve: bool = True,
) -> Path | list[Path]:
    """Resolve and convert path(s) into a Path object.

    Args:
        paths: An object or list of objects to convert to a `Path` or list of `Path`s.
        absolute: If `True` the `paths` given will be converted to absolute paths if
            they are relative.

    Returns:
        Returns a list of `Path`s if a list is given, otherwise just a `Path`.

    Examples:
        >>> import os
        >>> from pathlib import Path
        >>> resolve_paths("/abc/def/ghi")
        PosixPath('/abc/def/ghi')
        >>> resolve_paths(b"/jkl/mno")
        PosixPath('/jkl/mno')
        >>> resolve_paths(["/path/one", b"/path/two", Path("/path/three")])
        [PosixPath('/path/one'), PosixPath('/path/two'), PosixPath('/path/three')]
        >>> os.chdir("/bin")
        >>> resolve_paths("ls")
        PosixPath('/bin/ls')
        >>> resolve_paths("ls", resolve=False)
        PosixPath('ls')
    """
    if isinstance(paths, list):
        return [resolve_paths(p, resolve=resolve) for p in paths]
    # At this point 'paths' is a misnomer, refers to a singular path
    paths = paths.decode() if isinstance(paths, bytes) else paths
    paths = Path(paths)
    paths = paths.resolve() if resolve else paths
    return paths


class DirectoryIODriver(ABC):
    """Represents a driver for interacting with a directory.

    This abstract base class provides a template for directory drivers that can be used
    by file IO classes that want to be abstracted away from the exact details of the
    file type/format.
    """

    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def read_file(self, filename: str | bytes | os.PathLike | Path) -> Any:
        """Read a file via a directory driver.

        Args:
            filename: The name of the file to read from.

        Returns:
            The object saved to the given `filename`.
        """
        pass

    @abstractmethod
    def write_file(self, obj: Any, filename: str | bytes | os.PathLike | Path) -> None:
        """Write a file via a directory driver.

        Args:
            obj: The object to save.
            filename: The name of the file to save to.

        Returns:
            None
        """
        pass


def infer_directory_driver_from_suffix(file_suffixes: set[str]) -> DirectoryIODriver:
    """
    Find an appropriate subclass of `DirectoryIODriver` based on the file suffixes.

    TODO: Describe the heuristics used by this function in order of selection

    Args:
        file_suffixes: A set of file suffixes found in a given directory.

    Raises:
        ValueError: If an appropriate subclass cannot be found for the given
            `file_suffixes`.
        NotImplementedError: This functionality is not implemented yet and this
            documentation serves only as a spec for the moment.
    """
    raise NotImplementedError
