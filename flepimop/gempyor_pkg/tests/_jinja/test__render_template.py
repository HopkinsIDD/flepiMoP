from typing import Any

import pytest

from gempyor._jinja import _render_template


@pytest.mark.parametrize(
    ("name", "data", "output"),
    (
        ("test_template.j2", {"name": "John"}, "Hello John!"),
        ("test_template.j2", {"name": "Jake"}, "Hello Jake!"),
        ("test_template.j2", {"name": "Jane"}, "Hello Jane!"),
    ),
)
def test__render_template_renders_correctly(
    name: str, data: dict[str, Any], output: str
) -> None:
    assert _render_template(name, data) == output
