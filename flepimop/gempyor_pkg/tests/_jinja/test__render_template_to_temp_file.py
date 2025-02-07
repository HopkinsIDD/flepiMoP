from typing import Any

import pytest

from gempyor._jinja import _render_template_to_temp_file


@pytest.mark.parametrize(
    ("name", "data", "suffix", "prefix", "output"),
    (
        ("test_template.j2", {"name": "John"}, None, None, "Hello John!"),
        ("test_template.j2", {"name": "Jake"}, ".txt", None, "Hello Jake!"),
        ("test_template.j2", {"name": "Jane"}, None, "abc_", "Hello Jane!"),
        ("test_template.j2", {"name": "Job"}, ".dat", "def_", "Hello Job!"),
    ),
)
def test__render_template_to_temp_file_renders_correctly(
    name: str, data: dict[str, Any], suffix: str | None, prefix: str | None, output: str
) -> None:
    temp_file = _render_template_to_temp_file(name, data, suffix=suffix, prefix=prefix)
    assert temp_file.exists()
    if suffix:
        assert temp_file.name.endswith(suffix)
    if prefix:
        assert temp_file.name.startswith(prefix)
    assert temp_file.read_text(encoding="utf-8") == output
