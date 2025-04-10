"""Unit tests for the `gempyor.subpopulation_structure.SubpopulationStructure` class."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import confuse
import pandas as pd
import pytest
import numpy as np
import scipy.sparse

from gempyor.subpopulation_structure import (
    SUBPOP_NAMES_KEY,
    SUBPOP_POP_KEY,
    SubpopulationStructure,
)
from gempyor.testing import create_confuse_configview_from_dict


@dataclass(frozen=True)
class MockSubpopulationStructureInput:
    """
    Dataclass for creating a `SubpopulationStructure` instance for testing purposes.

    Attributes:
        setup_name: Name of the setup.
        subpop_config: A dictionary containing the subpopulation configuration.
        path_prefix: The path prefix for the geodata and mobility files.
    """

    setup_name: str
    subpop_config: dict[str, Any]
    path_prefix: Path
    geodata: pd.DataFrame
    mobility: pd.DataFrame | None = None

    def create_confuse_subview(self) -> confuse.Subview:
        """
        Create a confuse subview from the subpopulation configuration dictionary.

        Returns:
            A confuse subview containing the subpopulation configuration.
        """
        return create_confuse_configview_from_dict(self.subpop_config, name="subpop")

    def create_subpopulation_structure_instance(self) -> SubpopulationStructure:
        """
        Create a `SubpopulationStructure` instance.

        Returns:
            A `SubpopulationStructure` instance corresponding to the input data
            represented by this class.
        """
        return SubpopulationStructure(
            setup_name=self.setup_name,
            subpop_config=self.create_confuse_subview(),
            path_prefix=self.path_prefix,
        )


def geodata_only_test_factory(
    tmp_path: Path, records: list[dict[str, Any]]
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata only test.

    Returns:
        A `MockSubpopulationStructureInput` instance with the given geodata.
    """
    geodata = pd.DataFrame.from_records(records)
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
        },
        path_prefix=tmp_path,
        geodata=geodata,
    )


def subpop_pop_key_not_in_geodata_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that does not contain the subpopulation population key.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that does not
        contain the subpopulation population key.
    """
    return geodata_only_test_factory(
        tmp_path,
        [
            {
                "subpop": "USA",
                "people": 100,
            },
            {
                "subpop": "Canada",
                "people": 0,
            },
        ],
    )


def subpop_names_key_not_in_geodata_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that does not contain the subpopulation names key.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that does not
        contain the subpopulation names key.
    """
    return geodata_only_test_factory(
        tmp_path,
        [
            {
                "country": "USA",
                "population": 100,
            },
            {
                "country": "Canada",
                "population": 50,
            },
        ],
    )


def subpop_with_zero_population_in_geodata_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains subpopulations with zero population.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        subpopulations with zero population.
    """
    return geodata_only_test_factory(
        tmp_path,
        [
            {
                "subpop": "USA",
                "population": 100,
            },
            {
                "subpop": "Canada",
                "population": 0,
            },
        ],
    )


def duplicate_subpops_in_geodata_factory(tmp_path: Path) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains duplicate subpopulation names.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        duplicate subpopulation names.
    """
    return geodata_only_test_factory(
        tmp_path,
        [
            {
                "subpop": "USA",
                "population": 100,
            },
            {
                "subpop": "Canada",
                "population": 50,
            },
            {
                "subpop": "USA",
                "population": 75,
            },
        ],
    )


def valid_2pop_geodata_only_test_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations.
    """
    return geodata_only_test_factory(
        tmp_path,
        [
            {
                "subpop": "USA",
                "population": 100,
            },
            {
                "subpop": "Canada",
                "population": 50,
            },
        ],
    )


def valid_2pop_with_txt_mobility_test_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    geodata = pd.DataFrame(
        {
            "subpop": ["USA", "Canada"],
            "population": [100, 50],
        }
    )
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    mobility = np.array([[0, 1], [2, 0]])
    with (tmp_path / "mobility.txt").open("w") as f:
        np.savetxt(f, mobility)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
            "mobility": "mobility.txt",
        },
        path_prefix=tmp_path,
        geodata=geodata,
        mobility=mobility,
    )


def valid_2pop_with_csv_mobility_test_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    geodata = pd.DataFrame(
        {
            "subpop": ["USA", "Canada"],
            "population": [100, 50],
        }
    )
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    mobility = pd.DataFrame(
        {
            "ori": ["USA", "Canada"],
            "dest": ["Canada", "USA"],
            "amount": [2, 1],
        }
    )
    with (tmp_path / "mobility.csv").open("w") as f:
        mobility.to_csv(f, index=False)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        path_prefix=tmp_path,
        geodata=geodata,
        mobility=mobility,
    )


def valid_2pop_with_npz_mobility_test_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    geodata = pd.DataFrame(
        {
            "subpop": ["USA", "Canada"],
            "population": [100, 50],
        }
    )
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    mobility = scipy.sparse.csr_matrix([[0, 1], [2, 0]])
    scipy.sparse.save_npz(tmp_path / "mobility.npz", mobility)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
            "mobility": "mobility.npz",
        },
        path_prefix=tmp_path,
        geodata=geodata,
        mobility=mobility,
    )


def mobility_greater_than_population_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata and mobility where mobility is greater than population.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix where mobility is greater than
        population.
    """
    geodata = pd.DataFrame(
        {
            "subpop": ["USA", "Canada"],
            "population": [100, 50],
        }
    )
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    mobility = pd.DataFrame(
        {
            "ori": ["USA", "Canada"],
            "dest": ["Canada", "USA"],
            "amount": [101, 1],
        }
    )
    with (tmp_path / "mobility.csv").open("w") as f:
        mobility.to_csv(f, index=False)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        path_prefix=tmp_path,
        geodata=geodata,
        mobility=mobility,
    )


def mobility_greater_than_two_populations_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata and mobility where mobility is greater than population.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix where mobility is greater than
        population.
    """
    geodata = pd.DataFrame(
        {
            "subpop": ["USA", "Canada"],
            "population": [100, 50],
        }
    )
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    mobility = pd.DataFrame(
        {
            "ori": ["USA", "Canada"],
            "dest": ["Canada", "USA"],
            "amount": [101, 51],
        }
    )
    with (tmp_path / "mobility.csv").open("w") as f:
        mobility.to_csv(f, index=False)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        path_prefix=tmp_path,
        geodata=geodata,
        mobility=mobility,
    )


def mobility_greater_than_three_populations_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata and mobility where mobility is greater than population.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix where mobility is greater than
        population.
    """
    geodata = pd.DataFrame(
        {
            "subpop": ["USA", "Canada", "Mexico"],
            "population": [100, 50, 25],
        }
    )
    with (tmp_path / "geodata.csv").open("w") as f:
        geodata.to_csv(f, index=False)
    mobility = pd.DataFrame(
        {
            "ori": ["USA", "USA", "Canada"],
            "dest": ["Canada", "Mexico", "USA"],
            "amount": [60, 60, 2],
        }
    )
    with (tmp_path / "mobility.csv").open("w") as f:
        mobility.to_csv(f, index=False)
    return MockSubpopulationStructureInput(
        setup_name="test",
        subpop_config={
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        path_prefix=tmp_path,
        geodata=geodata,
        mobility=mobility,
    )


VALID_FACTORIES: Final = [
    valid_2pop_geodata_only_test_factory,
    valid_2pop_with_txt_mobility_test_factory,
    valid_2pop_with_csv_mobility_test_factory,
    valid_2pop_with_npz_mobility_test_factory,
]


def test_geodata_missing_subpop_pop_key_raises_value_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when the subpopulation population key is missing."""
    mock_input = subpop_pop_key_not_in_geodata_factory(tmp_path)
    assert SUBPOP_POP_KEY not in mock_input.geodata.columns
    with pytest.raises(
        ValueError,
        match=(
            f"^The '{SUBPOP_POP_KEY}' column was not found in the "
            f"geodata file '.*{mock_input.subpop_config['geodata']}'.$"
        ),
    ):
        mock_input.create_subpopulation_structure_instance()


def test_geodata_missing_subpop_names_key_raises_value_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when the subpopulation names key is missing."""
    mock_input = subpop_names_key_not_in_geodata_factory(tmp_path)
    assert SUBPOP_NAMES_KEY not in mock_input.geodata.columns
    with pytest.raises(
        ValueError,
        match=(
            f"^The '{SUBPOP_NAMES_KEY}' column was not found in the "
            f"geodata file '.*{mock_input.subpop_config['geodata']}'.$"
        ),
    ):
        mock_input.create_subpopulation_structure_instance()


def test_geodata_subpop_with_zero_population_raises_value_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when a subpopulation has zero population."""
    mock_input = subpop_with_zero_population_in_geodata_factory(tmp_path)
    assert (mock_input.geodata["population"] == 0).any()
    with pytest.raises(
        ValueError,
        match="^There are [0-9]+ subpops with zero population.$",
    ):
        mock_input.create_subpopulation_structure_instance()


def test_geodata_duplicate_subpop_names_raises_value_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when there are duplicate subpopulation names."""
    mock_input = duplicate_subpops_in_geodata_factory(tmp_path)
    assert mock_input.geodata["subpop"].nunique() < len(mock_input.geodata)
    with pytest.raises(
        ValueError,
        match=(
            "^The following subpopulation names are duplicated in the "
            f"geodata file '.*{mock_input.subpop_config['geodata']}': .*$"
        ),
    ):
        mock_input.create_subpopulation_structure_instance()


@pytest.mark.parametrize("factory", VALID_FACTORIES)
def test_subpopulation_structure_instance_attributes(
    caplog: pytest.LogCaptureFixture,
    recwarn: pytest.WarningsRecorder,
    tmp_path: Path,
    factory: Callable[[Path], MockSubpopulationStructureInput],
) -> None:
    """Test that the `SubpopulationStructure` instance has the correct attributes."""
    mock_input = factory(tmp_path)
    subpop_struct = mock_input.create_subpopulation_structure_instance()
    assert subpop_struct.setup_name == mock_input.setup_name
    assert subpop_struct.nsubpops == len(mock_input.geodata)
    assert (subpop_struct.subpop_pop == mock_input.geodata["population"].to_numpy()).all()
    assert subpop_struct.subpop_names == mock_input.geodata["subpop"].tolist()
    assert subpop_struct.data.equals(mock_input.geodata)
    assert scipy.sparse.issparse(subpop_struct.mobility)
    if mock_input.mobility is None:
        assert subpop_struct.mobility.nnz == 0
    if mock_input.subpop_config.get("mobility", "").endswith(".txt"):
        warn = recwarn.pop(PendingDeprecationWarning)
        assert str(warn.message) == (
            "Mobility files as matrices are not recommended. "
            "Please switch to long form csv files."
        )
    assert len(caplog.records) == int(mock_input.mobility is None)


@pytest.mark.parametrize(
    ("factory", "subpopulations"),
    [
        (mobility_greater_than_population_factory, {"USA"}),
        (mobility_greater_than_two_populations_factory, {"Canada", "USA"}),
        (mobility_greater_than_three_populations_factory, {"USA"}),
    ],
)
def test_mobility_greater_than_population_raises_value_error(
    tmp_path: Path,
    factory: Callable[[Path], MockSubpopulationStructureInput],
    subpopulations: set[str],
) -> None:
    """Test that a ValueError is raised when mobility is greater than population."""
    mock_input = factory(tmp_path)
    match_regex = "^The following subpopulations have mobility exceeding their population:"
    for i in range(len(subpopulations)):
        match_regex += f"(?:{',' if i > 0 else ''} (?:{'|'.join(subpopulations)}))"
    match_regex += ".$"
    with pytest.raises(ValueError, match=match_regex):
        mock_input.create_subpopulation_structure_instance()
