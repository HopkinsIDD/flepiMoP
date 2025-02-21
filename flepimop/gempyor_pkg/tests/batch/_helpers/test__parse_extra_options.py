import pytest

from gempyor.batch._helpers import _parse_extra_options


@pytest.mark.parametrize(
    ("extra", "expected"),
    (
        (None, {}),
        ([], {}),
        (["abc=def"], {"abc": "def"}),
        (["abc=def", "ghi=jkl"], {"abc": "def", "ghi": "jkl"}),
        (["abc=def=ghi"], {"abc": "def=ghi"}),
        (["abc=def", "abc=jkl"], {"abc": "jkl"}),
        (["abc"], {"abc": ""}),
    ),
)
def test_exact_results_for_select_inputs(
    extra: list[str] | None, expected: dict[str, str]
) -> None:
    assert _parse_extra_options(extra) == expected
