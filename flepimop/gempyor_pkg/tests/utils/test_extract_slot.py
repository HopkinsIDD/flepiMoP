"""Unit tests for `gempyor.utils.extract_slot`."""

from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from gempyor.utils import extract_slot


dummy_df = pd.DataFrame(
    data={
        "letters": ["a", "b", "c"],
        "numbers": [1, 2, 3],
        "value": [0.1, 0.2, 0.3],
    }
)


@pytest.mark.parametrize("file_ext", ["csv", "parquet"])
def test_slot_column_present_raises_value_error(
    tmp_path: Path, file_ext: Literal["csv", "parquet"]
) -> None:
    """Test that ValueError is raised when 'slot' column is present in the DataFrame."""
    df = dummy_df.copy()
    df["slot"] = [1, 2, 3]
    file = tmp_path / f"test_file.{file_ext}"
    if file_ext == "csv":
        df.to_csv(file, index=False)
    else:
        df.to_parquet(file, index=False)
    with pytest.raises(
        ValueError,
        match=f"^Column 'slot' already exists in the DataFrame from file: {file}.$",
    ):
        extract_slot(file)


@pytest.mark.parametrize("file_ext", ["csv", "parquet"])
@pytest.mark.parametrize(
    ("filename", "slot"),
    [("1.foo", 1), ("00017.seir.bar", 17), ("-001.outcomes.hosp", -1), ("1234", 1234)],
)
def test_reads_slot_from_prefix(
    tmp_path: Path, file_ext: Literal["csv", "parquet"], filename: str, slot: int
) -> None:
    """Test that the slot is read from the prefix of the filename."""
    df = dummy_df.copy()
    file = tmp_path / f"{filename}.{file_ext}"
    if file_ext == "csv":
        df.to_csv(file, index=False)
    else:
        df.to_parquet(file, index=False)
    result = extract_slot(file)
    assert "slot" in result.columns
    assert result["slot"].nunique() == 1
    assert result["slot"].unique().item() == slot
