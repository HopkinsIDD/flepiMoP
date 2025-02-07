from pathlib import Path
from typing import Any

import pytest

from gempyor._jinja import _render_template_to_file


@pytest.mark.parametrize(
    ("name", "data", "file", "output"),
    (
        ("test_template.j2", {"name": "John"}, Path("foo.txt"), "Hello John!"),
        ("test_template.j2", {"name": "Jake"}, Path("bar.txt"), "Hello Jake!"),
        ("test_template.j2", {"name": "Jane"}, Path("fizz.md"), "Hello Jane!"),
    ),
)
def test__render_template_to_file_renders_correctly(
    tmp_path: Path, name: str, data: dict[str, Any], file: Path, output: str
) -> None:
    if file.is_absolute():
        raise ValueError("The `file` argument must be relative for unit testing.")
    file = tmp_path / file
    _render_template_to_file(name, data, file)
    assert file.read_text() == output
