from typing import Any

import confuse
import pytest

from gempyor.utils import _dump_formatted_yaml


@pytest.mark.parametrize(
    ("data", "expected"),
    (
        ({"key": "value"}, "key: value\n"),
        (
            {"name": "Test Config", "compartments": {"infection_stage": ["S", "I", "R"]}},
            """name: "Test Config"
compartments:
    infection_stage: [S, I, R]
""",
        ),
        (
            {
                "seir": {
                    "parameters": {
                        "beta": {"value": 3.4},
                        "gamma": {"value": 5.6},
                    },
                    "transitions": {
                        "source": ["S"],
                        "destination": ["E"],
                        "rate": ["beta * gamma"],
                        "proportional_to": [["S"], ["I"]],
                        "proportion_exponent": [1, 1],
                    },
                }
            },
            """seir:
    parameters:
        beta:
            value: 3.4
        gamma:
            value: 5.6
    transitions:
        source: [S]
        destination: [E]
        rate: ["beta * gamma"]
        proportional_to: [[S], [I]]
        proportion_exponent: [1, 1]
""",
        ),
    ),
)
def test_exact_output_for_select_values(data: dict[str, Any], expected: str) -> None:
    cfg = confuse.Configuration("test", __name__)
    cfg.set(data)
    assert _dump_formatted_yaml(cfg) == expected
