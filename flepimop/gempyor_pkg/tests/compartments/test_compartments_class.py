from dataclasses import dataclass
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import confuse
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from gempyor.compartments import Compartments
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
