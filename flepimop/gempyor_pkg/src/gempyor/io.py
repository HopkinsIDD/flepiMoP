import os
from pathlib import Path


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
