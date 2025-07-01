import pandas as pd
import numpy as np
import pytest
import confuse

from gempyor.NPI import helpers


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([[1, 2], [3, 4]], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([], []),
        ([[], [1, 2]], [1, 2]),
    ],
)
def test_flatten_list_of_lists(input_list, expected_output):
    assert helpers.flatten_list_of_lists(input_list) == expected_output


@pytest.mark.parametrize(
    "input_list, expected_output",
    [([1, 2, 3], [[1, 2, 3]]), ([[1, 2], [3]], [[1, 2], [3]]), ([], [])],
)
def test_make_list_of_list(input_list, expected_output):
    assert helpers.make_list_of_list(input_list) == expected_output


class TestReduceParameter:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base_param = np.ones((2, 2)) * 10

    @pytest.mark.parametrize(
        "method, modification, expected_result",
        [
            ("product", 2.0, np.ones((2, 2)) * 20),
            ("sum", 2.0, np.ones((2, 2)) * 12),
            ("reduction_product", 0.2, np.ones((2, 2)) * 8),
        ],
    )
    def test_reduce_with_float(self, method, modification, expected_result):
        result = helpers.reduce_parameter(self.base_param, modification, method=method)
        np.testing.assert_array_equal(result, expected_result)

    def test_reduce_with_dataframe(self):
        param_array = np.ones((2, 2))
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        index_labels = ["subpop_A", "subpop_B"]
        modification_df = pd.DataFrame(3.0, index=index_labels, columns=dates)
        result = helpers.reduce_parameter(param_array, modification_df, method="product")
        expected = np.ones((2, 2)) * 3.0
        np.testing.assert_array_equal(result, expected)

    def test_reduce_with_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            helpers.reduce_parameter(self.base_param, 0.5, method="unknown_method")


@pytest.mark.parametrize(
    "test_name, mock_config, affected_subpops, expected_grouped, expected_ungrouped",
    [
        ("no groups defined", {}, ["A", "B", "C"], [], ["A", "B", "C"]),
        (
            "grouping all subpops",
            {"subpop_groups": "all"},
            ["A", "B", "C"],
            [["A", "B", "C"]],
            [],
        ),
        (
            "specific subpop groups",
            {"subpop_groups": [["A", "C"], ["D"]]},
            ["A", "B", "C", "D", "E"],
            [["A", "C"], ["D"]],
            ["B", "E"],
        ),
        (
            "groups with extra subpops",
            {"subpop_groups": [["A", "F"], ["G", "B"]]},
            ["A", "B", "C"],
            [["A"], ["B"]],
            ["C"],
        ),
        (
            "groups with empty lists",
            {"subpop_groups": [["A", "C"], [], ["D"]]},
            ["A", "B", "C", "D", "E"],
            [["A", "C"], ["D"]],
            ["B", "E"],
        ),
    ],
)
def test_get_spatial_groups(
    test_name, mock_config, affected_subpops, expected_grouped, expected_ungrouped
):
    config = confuse.Configuration("TestApp", __name__)
    config.set(mock_config)
    result = helpers.get_spatial_groups(config, sorted(affected_subpops))
    assert result["grouped"] == expected_grouped
    assert result["ungrouped"] == expected_ungrouped
