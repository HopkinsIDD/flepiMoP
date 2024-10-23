from dataclasses import dataclass
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import confuse
import pandas as pd
from pandas.testing import assert_frame_equal
import pyarrow.parquet as pq
import pytest

from gempyor.compartments import Compartments, NestedListOfStr
from gempyor.testing import create_confuse_configview_from_dict


@dataclass(frozen=True)
class MockCompartmentsInput:
    """
    A self contained class containing mock inputs for the `Compartments` class.

    Attributes:
        seir: A dictionary representation of an seir config section.
        compartments: A dictionary representation fo a compartments config section.
        compartments_file: A pathlike object of a parquet file containing the
            compartment names.
        transitions_file: A pathlike object of a parquet file containing the compartment
            transitions.
    """

    seir: dict[str, Any] | None
    compartments: dict[str, Any] | None
    compartments_file: PathLike | None
    transitions_file: PathLike | None
    transitions: pd.DataFrame | None

    def seir_subview(self) -> confuse.Subview | None:
        """
        Create a Subview representation of an seir config section.

        Returns:
            The seir config section represented as a confuse subview as gempyor expects
            internally or `None` if the dict representation is `None`.
        """
        return (
            None
            if self.seir is None
            else create_confuse_configview_from_dict(self.seir, "seir")
        )

    def compartments_subview(self) -> confuse.Subview | None:
        """
        Create a Subview representation of a compartments config section.

        Returns:
            The compartments config section represented as a confuse subview as gempyor
            expects internally or `None` if the dict representation is `None`.
        """
        return (
            None
            if self.compartments is None
            else create_confuse_configview_from_dict(self.compartments, "compartments")
        )

    def compartments_instance(self) -> Compartments:
        """
        Create a Compartments instance from the inputs represented by this class.

        Returns:
            An instance of `Compartments` represented by the inputs corresponding to
            this mock inputs class.
        """
        return Compartments(
            seir_config=self.seir_subview(),
            compartments_config=self.compartments_subview(),
            compartments_file=self.compartments_file,
            transitions_file=self.transitions_file,
        )

    def compartments_dataframe(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame representing the compartments of this mock input.

        Returns:
            A pandas DataFrame with the names of the stages as the columns and a column
            called 'name' for the unique name.
        """
        if self.compartments:
            df = pd.DataFrame(
                data=list(product(*self.compartments.values())),
                columns=self.compartments.keys(),
            )
            df["name"] = df.agg("_".join, axis=1)
            return df
        raise NotImplementedError


def empty_inputs_factory(tmp_path: Path) -> MockCompartmentsInput:
    return MockCompartmentsInput(
        seir=None,
        compartments=None,
        compartments_file=None,
        transitions_file=None,
        transitions=None,
    )


def sir_from_config_inputs_factory(tmp_path: Path) -> MockCompartmentsInput:
    return MockCompartmentsInput(
        seir={
            "integration": {
                "method": "rk4",
                "dt": 1.0,
            },
            "parameters": {
                "beta": {
                    "value": 0.1,
                },
                "gamma": {
                    "value": 0.2,
                },
            },
            "transitions": [
                {
                    "source": ["S"],
                    "destination": ["I"],
                    "rate": ["beta"],
                    "proportional_to": [["S"], ["I"]],
                    "proportion_exponent": [[1], [1]],
                },
                {
                    "source": ["I"],
                    "destination": ["R"],
                    "rate": ["gamma"],
                    "proportional_to": [["I"]],
                    "proportion_exponent": [[1]],
                },
            ],
        },
        compartments={
            "infection_stage": ["S", "I", "R"],
        },
        compartments_file=None,
        transitions_file=None,
        transitions=pd.DataFrame.from_records(
            data=[
                {
                    "source": ["S"],
                    "destination": ["I"],
                    "rate": ["beta"],
                    "proportional_to": [[["S"]], [["I"]]],
                    "proportion_exponent": [["1"], ["1"]],
                },
                {
                    "source": ["I"],
                    "destination": ["R"],
                    "rate": ["gamma"],
                    "proportional_to": [[["I"]]],
                    "proportion_exponent": [["1"]],
                },
            ]
        ),
    )


def ris_from_config_inputs_factory(tmp_path: Path) -> MockCompartmentsInput:
    return MockCompartmentsInput(
        seir={
            "integration": {
                "method": "rk4",
                "dt": 1.0,
            },
            "parameters": {
                "beta": {
                    "value": 0.1,
                },
                "gamma": {
                    "value": 0.2,
                },
            },
            "transitions": [
                {
                    "source": ["I"],
                    "destination": ["R"],
                    "rate": ["gamma"],
                    "proportional_to": [["I"]],
                    "proportion_exponent": [[1]],
                },
                {
                    "source": ["S"],
                    "destination": ["I"],
                    "rate": ["beta"],
                    "proportional_to": [["S"], ["I"]],
                    "proportion_exponent": [[1], [1]],
                },
            ],
        },
        compartments={
            "infection_stage": ["R", "I", "S"],
        },
        compartments_file=None,
        transitions_file=None,
        transitions=pd.DataFrame.from_records(
            data=[
                {
                    "source": ["I"],
                    "destination": ["R"],
                    "rate": ["gamma"],
                    "proportional_to": [[["I"]]],
                    "proportion_exponent": [["1"]],
                },
                {
                    "source": ["S"],
                    "destination": ["I"],
                    "rate": ["beta"],
                    "proportional_to": [[["S"]], [["I"]]],
                    "proportion_exponent": [["1"], ["1"]],
                },
            ]
        ),
    )


def seir_from_config_inputs_factory(tmp_path: Path) -> MockCompartmentsInput:
    return MockCompartmentsInput(
        seir={
            "integration": {
                "method": "rk4",
                "dt": 1,
            },
            "parameters": {
                "sigma": 0.25,
                "gamma": 0.2,
                "R0": 2.5,
            },
            "transitions": [
                {
                    "source": ["S"],
                    "destination": ["E"],
                    "rate": ["R0 * gamma"],
                    "proportional_to": [["S"], ["I"]],
                    "proportion_exponent": ["1", "1"],
                },
                {
                    "source": ["E"],
                    "destination": ["I"],
                    "rate": ["sigma"],
                    "proportional_to": ["E"],
                    "proportion_exponent": ["1"],
                },
                {
                    "source": ["I"],
                    "destination": ["R"],
                    "rate": ["gamma"],
                    "proportional_to": ["I"],
                    "proportion_exponent": ["1"],
                },
            ],
        },
        compartments={
            "infection_stage": ["S", "E", "I", "R"],
        },
        compartments_file=None,
        transitions_file=None,
        transitions=pd.DataFrame.from_records(
            data=[
                {
                    "source": ["S"],
                    "destination": ["E"],
                    "rate": ["R0 * gamma"],
                    "proportional_to": [[["S"]], [["I"]]],
                    "proportion_exponent": [["1"], ["1"]],
                },
                {
                    "source": ["E"],
                    "destination": ["I"],
                    "rate": ["sigma"],
                    "proportional_to": [[["E"]]],
                    "proportion_exponent": [["1"]],
                },
                {
                    "source": ["I"],
                    "destination": ["R"],
                    "rate": ["gamma"],
                    "proportional_to": [[["I"]]],
                    "proportion_exponent": [["1"]],
                },
            ]
        ),
    )


valid_input_factories = (
    (sir_from_config_inputs_factory),
    (ris_from_config_inputs_factory),
    (seir_from_config_inputs_factory),
)


class TestCompartments:
    def test_config_or_file_not_set_value_error(self, tmp_path: Path) -> None:
        mock_inputs = empty_inputs_factory(tmp_path)
        with pytest.raises(
            ValueError,
            match=r"^Compartments object not set\, no config or file provided$",
        ):
            mock_inputs.compartments_instance()

    @pytest.mark.parametrize("factory", valid_input_factories)
    def test_instance_attributes_and_simpler_methods(
        self, tmp_path: Path, factory: Callable[[Path], MockCompartmentsInput]
    ) -> None:
        mock_inputs = factory(tmp_path)
        compartments = mock_inputs.compartments_instance()

        assert compartments.times_set == 1

        assert isinstance(compartments.compartments, pd.DataFrame)
        assert_frame_equal(
            compartments.compartments, mock_inputs.compartments_dataframe()
        )

        assert isinstance(compartments.transitions, pd.DataFrame)

        if mock_inputs.transitions is not None:
            assert_frame_equal(compartments.transitions, mock_inputs.transitions)

        assert compartments.check_transition_element(None, None) == True
        assert compartments.check_transition_elements(None, None) == True

        assert compartments.get_ncomp() == len(mock_inputs.compartments_dataframe())

        df = compartments.get_compartments_explicitDF()
        assert id(df) != id(compartments.compartments)
        assert all(c.startswith("mc_") for c in df.columns)

    @pytest.mark.parametrize(
        ("factory_one", "factory_two", "expected_result"),
        (
            (sir_from_config_inputs_factory, sir_from_config_inputs_factory, True),
            (sir_from_config_inputs_factory, ris_from_config_inputs_factory, False),
            (sir_from_config_inputs_factory, seir_from_config_inputs_factory, None),
            (ris_from_config_inputs_factory, seir_from_config_inputs_factory, None),
            (seir_from_config_inputs_factory, seir_from_config_inputs_factory, True),
        ),
    )
    def test_compartments_equality(
        self,
        tmp_path: Path,
        factory_one: Callable[[Path], MockCompartmentsInput],
        factory_two: Callable[[Path], MockCompartmentsInput],
        expected_result: bool | None,
    ) -> None:
        mock_inputs_one = factory_one(tmp_path)
        mock_inputs_two = factory_two(tmp_path)
        compartments_one = mock_inputs_one.compartments_instance()
        compartments_two = mock_inputs_two.compartments_instance()

        if isinstance(expected_result, bool):
            assert (compartments_one == compartments_two) == expected_result
        else:
            # Comparing compartments is an unsafe operation
            with pytest.raises(
                ValueError,
                match=(
                    r"^Can only compare identically-labeled \(both "
                    r"index and columns\) DataFrame objects$"
                ),
            ):
                compartments_one == compartments_two

    @pytest.mark.parametrize(
        ("factory", "comp_dict", "error_info"),
        (
            (sir_from_config_inputs_factory, {"name": "Z"}, "no information"),
            (
                seir_from_config_inputs_factory,
                {"infection_stage": "A"},
                "foobar fizzbuzz",
            ),
        ),
    )
    def test_get_comp_idx_ambiguous_filter_value_error(
        self,
        tmp_path: Path,
        factory: Callable[[Path], MockCompartmentsInput],
        comp_dict: dict[str, NestedListOfStr],
        error_info: str,
    ) -> None:
        mock_inputs = factory(tmp_path)
        compartments = mock_inputs.compartments_instance()

        with pytest.raises(
            ValueError,
            match=(
                "(?s)^The provided dictionary does not allow to isolate a compartment: "
                f"{comp_dict} isolate .* from options {compartments.compartments}. The "
                f"get_comp_idx function was called by'{error_info}'.$"
            ),
        ):
            compartments.get_comp_idx(comp_dict, error_info=error_info)

    @pytest.mark.parametrize(
        ("factory", "comp_dict"),
        (
            (sir_from_config_inputs_factory, {"name": "S"}),
            (seir_from_config_inputs_factory, {"infection_stage": "E"}),
        ),
    )
    def test_get_comp_idx_output_validation(
        self,
        tmp_path: Path,
        factory: Callable[[Path], MockCompartmentsInput],
        comp_dict: dict[str, NestedListOfStr],
    ) -> None:
        mock_inputs = factory(tmp_path)
        compartments = mock_inputs.compartments_instance()

        idx = compartments.get_comp_idx(comp_dict)

        comp_dict = {
            k: (v if isinstance(v, list) else [v]) for k, v in comp_dict.items()
        }
        df = mock_inputs.compartments_dataframe()
        expected_idx = df.index[
            df[comp_dict.keys()].isin(comp_dict).all(axis="columns")
        ][0]

        assert idx == expected_idx

    @pytest.mark.parametrize("factory", valid_input_factories)
    @pytest.mark.parametrize(
        ("compartments_file", "transitions_file"),
        (
            ("compartments.parquet", "transitions.parquet"),
            ("compartments.csv", "transitions.csv"),
        ),
    )
    @pytest.mark.parametrize("write_parquet", (True, False))
    def test_to_file_output_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        factory: Callable[[Path], MockCompartmentsInput],
        compartments_file: PathLike,
        transitions_file: PathLike,
        write_parquet: bool,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mock_inputs = factory(tmp_path)
        compartments = mock_inputs.compartments_instance()

        compartments_path = Path(compartments_file).absolute()
        transitions_path = Path(transitions_file).absolute()
        assert not compartments_path.exists()
        assert not transitions_path.exists()

        assert (
            compartments.toFile(
                compartments_file=compartments_file,
                transitions_file=transitions_file,
                write_parquet=write_parquet,
            )
            is None
        )

        assert compartments_path.exists()
        assert transitions_path.exists()

        reader = lambda x: (
            pq.read_table(x).to_pandas() if write_parquet else pd.read_csv(x)
        )
        compartments_df = reader(compartments_path)
        transitions_df = reader(transitions_path)
        assert isinstance(compartments_df, pd.DataFrame)
        assert isinstance(transitions_df, pd.DataFrame)
