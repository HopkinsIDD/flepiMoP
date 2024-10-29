from typing import Any

from jinja2 import Template
from jinja2.exceptions import TemplateNotFound
import pytest

from gempyor._jinja import _get_template


@pytest.mark.parametrize(
    "name", (("template_does_not_exist.j2"), ("template_does_not_exist_again.j2"))
)
def test__get_template_template_not_found_error(name: str) -> None:
    with pytest.raises(TemplateNotFound, match=name):
        _get_template(name)


@pytest.mark.parametrize(
    ("name", "data", "output"),
    (
        ("test_template.j2", {"name": "John"}, "Hello John!"),
        ("test_template.j2", {"name": "Jake"}, "Hello Jake!"),
        ("test_template.j2", {"name": "Jane"}, "Hello Jane!"),
    ),
)
def test__get_template_renders_correctly(
    name: str, data: dict[str, Any], output: str
) -> None:
    template = _get_template(name)
    assert isinstance(template, Template)
    rendered_template = template.render(**data)
    assert rendered_template == output
