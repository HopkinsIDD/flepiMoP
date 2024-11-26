"""
Internal Jinja2 template rendering utilities.

This module contains utilities for finding and rendering Jinja2 templates. This module
is tightly coupled to the organization of the package and not intended for external use.
"""

# Exports
__all__ = []


# Imports
from os.path import dirname
from pathlib import Path
from tempfile import mkstemp
from typing import Any

from jinja2 import Environment, FileSystemLoader, PackageLoader, Template


# Globals
try:
    _jinja_environment = Environment(
        loader=FileSystemLoader(dirname(__file__).replace("\\", "/") + "/templates")
    )
except ValueError:
    _jinja_environment = Environment(loader=PackageLoader("gempyor"))


# Functions
def _get_template(name: str) -> Template:
    """
    Get a jinja template by name.

    Args:
        name: The name of the template to pull.

    Returns:
        A jinja template object corresponding to `name`.

    Examples:
        >>> _get_template("test_template.j2")
        <Template 'test_template.j2'>
    """
    return _jinja_environment.get_template(name)


def _render_template(name: str, data: dict[str, Any]) -> str:
    """
    Render a jinja template by name.

    Args:
        name: The name of the template to pull.
        data: The data to pass to the template when rendering.

    Returns:
        The rendered template as a string.

    Examples:
        >>> _render_template("test_template.j2", {"name": "Bob"})
        'Hello Bob!'
    """
    return _get_template(name).render(data)


def _render_template_to_file(name: str, data: dict[str, Any], file: Path) -> None:
    """
    Render a jinja template and save to a file.

    Args:
        name: The name of the template to pull.
        data: The data to pass to the template when rendering.
        file: The file to save the rendered template to.

    Examples:
        >>> from pathlib import Path
        >>> file = Path("hi.txt")
        >>> _render_template_to_file("test_template.j2", {"name": "Jane"}, file)
        >>> file.read_text()
        'Hello Jane!'
    """
    with file.open(mode="w", encoding="utf-8") as f:
        f.write(_render_template(name, data))


def _render_template_to_temp_file(
    name: str,
    data: dict[str, Any],
    suffix: str | None = None,
    prefix: str | None = None,
) -> Path:
    """
    Render a jinja template and save to a temporary file.

    Args:
        name: The name of the template to pull.
        data: The data to pass to the template when rendering.
        suffix: The suffix of the temporary file, such as an extension. Passed on to
            `tempfile.mkstemp`.
        prefix: The prefix of the temporary file. Passed on to `tempfile.mkstemp`.

    Returns:
        The file containing the rendered template as a `Path` object.

    See Also:
        [`tempfile.mkstemp`](https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp)

    Examples:
        >>> file = _render_template_to_temp_file(
        ...     "test_template.j2", {"name": "John"}, suffix=".txt", prefix="foo_"
        ... )
        >>> file
        PosixPath('/var/folders/2z/h3pc0p7s3ng1tvxrgsw5kr680000gp/T/foo_ocaomg4k.txt')
        >>> file.read_text()
        'Hello John!'
    """
    _, tmp = mkstemp(suffix=suffix, prefix=prefix, text=True)
    tmp = Path(tmp)
    _render_template_to_file(name, data, tmp)
    return tmp
