from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import confuse
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
            else create_confuse_configview_from_dict("seir", self.seir)
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
            else create_confuse_configview_from_dict("compartments", self.compartments)
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


def empty_inputs_factory(tmp_path: Path) -> MockCompartmentsInput:
    return MockCompartmentsInput(
        seir=None,
        compartments=None,
        compartments_file=None,
        transitions_file=None,
    )


class TestCompartments:
    def test_config_or_file_not_set_value_error(self, tmp_path: Path) -> None:
        mock_inputs = empty_inputs_factory(tmp_path)
        with pytest.raises(
            ValueError,
            match=r"^Compartments object not set\, no config or file provided$",
        ):
            mock_inputs.compartments_instance()
