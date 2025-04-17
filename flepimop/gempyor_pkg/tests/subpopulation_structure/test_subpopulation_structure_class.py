"""Unit tests for the `gempyor.subpopulation_structure.SubpopulationStructure` class."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Final, Literal

import confuse
import pandas as pd
from pydantic import ValidationError
import pytest
import numpy as np
import numpy.typing as npt
import scipy.sparse

from gempyor.subpopulation_structure import SubpopulationStructure
from gempyor.testing import create_confuse_configview_from_dict


@dataclass(frozen=True)
class MockSubpopulationStructureInput:
    """
    Dataclass for creating a `SubpopulationStructure` instance for testing purposes.

    Attributes:
        subpop_config: A dictionary containing the subpopulation configuration.
        path_prefix: The path prefix for the geodata and mobility files.
    """

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
            self.create_confuse_subview(), path_prefix=self.path_prefix
        )

    @classmethod
    def create_mock_input(
        cls,
        tmp_path: Path,
        subpop_config: dict[str, Any],
        geodata: pd.DataFrame,
        mobility: (
            npt.NDArray[np.number] | pd.DataFrame | scipy.sparse.sparray | None
        ) = None,
    ) -> "MockSubpopulationStructureInput":
        """
        Helper to create mock input with handling for temporary files.

        Args:
            tmp_path: The temporary path to create files in.
            subpop_config: The subpopulation configuration dictionary.
            geodata: The geodata DataFrame.
            mobility: The mobility DataFrame or matrix.

        Returns:
            A `MockSubpopulationStructureInput` instance.
        """
        geodata_file = tmp_path / subpop_config["geodata"]
        with geodata_file.open(
            "w" + ("b" if geodata_file.suffix == ".parquet" else "")
        ) as f:
            if geodata_file.suffix == ".parquet":
                geodata.to_parquet(f, index=False)
            else:
                geodata.to_csv(f, index=False)
        if mobility is not None:
            mobility_file = tmp_path / subpop_config["mobility"]
            if mobility_file.suffix == ".npz":
                scipy.sparse.save_npz(mobility_file, mobility)
            else:
                with mobility_file.open(
                    "w" + ("b" if mobility_file.suffix == ".parquet" else "")
                ) as f:
                    if mobility_file.suffix == ".parquet":
                        mobility.to_parquet(f, index=False)
                    elif mobility_file.suffix == ".csv":
                        mobility.to_csv(f, index=False)
                    elif mobility_file.suffix == ".txt":
                        np.savetxt(f, mobility)
        return cls(
            subpop_config=subpop_config,
            path_prefix=tmp_path,
            geodata=geodata,
            mobility=mobility,
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
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "people": 100,
                },
                {
                    "subpop": "Canada",
                    "people": 0,
                },
            ]
        ),
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
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
        },
        pd.DataFrame.from_records(
            [
                {
                    "country": "USA",
                    "population": 100,
                },
                {
                    "country": "Canada",
                    "population": 50,
                },
            ]
        ),
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
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
        },
        pd.DataFrame.from_records(
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
        ),
    )


def duplicate_subpops_in_geodata_factory(tmp_path: Path) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains duplicate subpopulation names.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        duplicate subpopulation names.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
        },
        pd.DataFrame.from_records(
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
        ),
    )


def valid_2pop_geodata_csv_only_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
    )


def valid_2pop_geodata_parquet_only_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "populations.parquet",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
    )


def valid_2pop_with_txt_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.txt",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
        mobility=np.array([[0, 1], [2, 0]]),
    )


def valid_2pop_with_csv_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
        mobility=pd.DataFrame.from_records(
            [
                {
                    "ori": "USA",
                    "dest": "Canada",
                    "amount": 1,
                },
                {
                    "ori": "Canada",
                    "dest": "USA",
                    "amount": 2,
                },
            ]
        ),
    )


def valid_2pop_with_parquet_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.parquet",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
        mobility=pd.DataFrame.from_records(
            [
                {
                    "ori": "USA",
                    "dest": "Canada",
                    "amount": 1,
                },
                {
                    "ori": "Canada",
                    "dest": "USA",
                    "amount": 2,
                },
            ]
        ),
    )


def valid_2pop_with_npz_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.npz",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
        mobility=scipy.sparse.csr_matrix([[0, 1], [2, 0]]),
    )


def valid_selected_pop_with_no_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    """
    Factory for geodata file that contains two valid subpopulations and a mobility matrix.

    Returns:
        A `MockSubpopulationStructureInput` instance with a geodata file that contains
        two valid subpopulations and a mobility matrix.
    """
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "selected": "USA",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
    )


def valid_multiple_selected_pop_with_no_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "selected": ["USA", "Mexico"],
        },
        pd.DataFrame.from_records(
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
                    "subpop": "Mexico",
                    "population": 25,
                },
                {
                    "subpop": "Cuba",
                    "population": 5,
                },
                {
                    "subpop": "Greenland",
                    "population": 1,
                },
            ]
        ),
    )


def valid_multiple_selected_pop_with_mobility_factory(
    tmp_path: Path,
) -> MockSubpopulationStructureInput:
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.parquet",
            "selected": ["USA", "Mexico"],
            "mobility": "mobility.parquet",
        },
        pd.DataFrame.from_records(
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
                    "subpop": "Mexico",
                    "population": 25,
                },
            ]
        ),
        mobility=pd.DataFrame.from_records(
            [
                {
                    "ori": "USA",
                    "dest": "Canada",
                    "amount": 1,
                },
                {
                    "ori": "Canada",
                    "dest": "USA",
                    "amount": 2,
                },
                {
                    "ori": "Mexico",
                    "dest": "USA",
                    "amount": 3,
                },
                {
                    "ori": "USA",
                    "dest": "Mexico",
                    "amount": 4,
                },
            ]
        ),
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
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        pd.DataFrame(
            data={
                "subpop": ["USA", "Canada"],
                "population": [100, 50],
            }
        ),
        mobility=pd.DataFrame(
            data={
                "ori": ["USA", "Canada"],
                "dest": ["Canada", "USA"],
                "amount": [101, 1],
            }
        ),
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
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        pd.DataFrame(
            data={
                "subpop": ["USA", "Canada"],
                "population": [100, 50],
            },
        ),
        mobility=pd.DataFrame(
            data={
                "ori": ["USA", "Canada"],
                "dest": ["Canada", "USA"],
                "amount": [101, 51],
            },
        ),
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
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "mobility": "mobility.csv",
        },
        pd.DataFrame(
            data={
                "subpop": ["USA", "Canada", "Mexico"],
                "population": [100, 50, 25],
            },
        ),
        mobility=pd.DataFrame(
            data={
                "ori": ["USA", "USA", "Canada"],
                "dest": ["Canada", "Mexico", "USA"],
                "amount": [60, 60, 2],
            },
        ),
    )


def mobility_zero_or_less_factory(tmp_path: Path) -> MockSubpopulationStructureInput:
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.parquet",
            "mobility": "mobility.parquet",
        },
        pd.DataFrame(
            data={
                "subpop": ["USA", "Canada"],
                "population": [100, 50],
            },
        ),
        mobility=pd.DataFrame(
            data={
                "ori": ["USA", "Canada"],
                "dest": ["Canada", "USA"],
                "amount": [0, -1],
            },
        ),
    )


def selected_missing_from_geodata(tmp_path: Path) -> MockSubpopulationStructureInput:
    return MockSubpopulationStructureInput.create_mock_input(
        tmp_path,
        {
            "geodata": "geodata.csv",
            "selected": "Mexico",
        },
        pd.DataFrame.from_records(
            [
                {
                    "subpop": "USA",
                    "population": 100,
                },
                {
                    "subpop": "Canada",
                    "population": 50,
                },
            ]
        ),
    )


VALID_FACTORIES: Final = [
    valid_2pop_geodata_csv_only_factory,
    valid_2pop_geodata_parquet_only_factory,
    valid_2pop_with_txt_mobility_factory,
    valid_2pop_with_csv_mobility_factory,
    valid_2pop_with_parquet_mobility_factory,
    valid_2pop_with_npz_mobility_factory,
    valid_selected_pop_with_no_mobility_factory,
    valid_multiple_selected_pop_with_no_mobility_factory,
    valid_multiple_selected_pop_with_mobility_factory,
]


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (subpop_pop_key_not_in_geodata_factory, "population"),
        (subpop_names_key_not_in_geodata_factory, "subpop"),
    ],
)
def test_geodata_missing_required_field_raises_validation_error_error(
    tmp_path: Path,
    factory: Callable[[Path], MockSubpopulationStructureInput],
    field: Literal["population", "subpop"],
) -> None:
    """Test that"""
    mock_input = factory(tmp_path)
    assert field not in mock_input.geodata.columns
    raises_match = rf"^{len(mock_input.geodata)} validation errors.*"
    raises_match += ".*".join(
        [rf"{i}\.{field}\s+field required" for i in range(len(mock_input.geodata))]
    )
    with pytest.raises(
        ValidationError,
        match=re.compile(raises_match, flags=re.DOTALL + re.IGNORECASE + re.MULTILINE),
    ):
        mock_input.create_subpopulation_structure_instance()


def test_geodata_subpop_with_zero_population_raises_value_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when a subpopulation has zero population."""
    mock_input = subpop_with_zero_population_in_geodata_factory(tmp_path)
    assert (mock_input.geodata["population"] == 0).any()
    with pytest.raises(
        ValueError,
        match=r"1.population\s+Input should be greater than 0",
    ):
        mock_input.create_subpopulation_structure_instance()


def test_geodata_duplicate_subpop_names_raises_value_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when there are duplicate subpopulation names."""
    mock_input = duplicate_subpops_in_geodata_factory(tmp_path)
    assert mock_input.geodata["subpop"].nunique() < len(mock_input.geodata)
    with pytest.raises(
        ValueError,
        match="The following subpopulation names are duplicated in the geodata file: .*",
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
    selected = mock_input.subpop_config.get("selected", [])
    selected = selected if isinstance(selected, list) else [selected]
    assert subpop_struct.nsubpops == len(selected) or len(mock_input.geodata)
    geodata = (
        mock_input.geodata[mock_input.geodata["subpop"].isin(selected)]
        if selected
        else mock_input.geodata
    )
    assert (subpop_struct.subpop_pop == geodata["population"].to_numpy()).all()
    assert subpop_struct.subpop_names == geodata["subpop"].tolist()
    assert subpop_struct.data.equals(geodata)
    assert scipy.sparse.issparse(subpop_struct.mobility)
    assert subpop_struct.mobility.shape == (
        len(selected) or len(mock_input.geodata),
        len(selected) or len(mock_input.geodata),
    )
    if mock_input.mobility is None:
        assert subpop_struct.mobility.nnz == 0
    if mock_input.subpop_config.get("mobility", "").endswith(".txt"):
        warn = recwarn.pop(PendingDeprecationWarning)
        assert str(warn.message) == (
            "Mobility files as matrices are not recommended. "
            "Please switch to long form csv files."
        )
    assert len(caplog.records) == int(mock_input.mobility is None)


def test_mobility_zero_or_less_raises_validation_error(tmp_path: Path) -> None:
    """Test that a ValueError is raised when mobility is zero or less."""
    mock_input = mobility_zero_or_less_factory(tmp_path)
    assert (zero_or_less := (mock_input.mobility["amount"] <= 0).sum()) > 0
    raises_match = rf"^{zero_or_less} validation errors.*"
    raises_match += ".*".join(
        [rf"{i}\.amount\s+Input should be greater than 0" for i in range(zero_or_less)]
    )
    with pytest.raises(
        ValidationError,
        match=re.compile(raises_match, flags=re.DOTALL + re.IGNORECASE + re.MULTILINE),
    ):
        mock_input.create_subpopulation_structure_instance()


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


def test_selected_subpop_not_in_geodata_raises_value_error(tmp_path: Path) -> None:
    """Test `ValueError` is raised when the selected subpopulation is not in the geodata."""
    mock_input = selected_missing_from_geodata(tmp_path)
    selected = mock_input.subpop_config["selected"]
    selected = selected if isinstance(selected, list) else [selected]
    missing_subpops = [s for s in selected if s not in mock_input.geodata["subpop"].values]
    raises_match = r"The following selected subpopulations are not in the geodata:"
    for i in range(len(missing_subpops)):
        raises_match += f"(?:{'' if i == 0 else ','} {'|'.join(missing_subpops)})"
    raises_match += r"\.$"
    with pytest.raises(
        ValueError,
        match=re.compile(raises_match),
    ):
        mock_input.create_subpopulation_structure_instance()
