import shutil
import pytest
import numpy as np
from pathlib import Path
import pandas as pd

from gempyor.model_builder import ModelBuilder


@pytest.fixture(scope="module")
def modelbuilder_from_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("model_input")

    original_dir = Path(__file__).resolve()
    while original_dir.name != "flepimop":
        original_dir = original_dir.parent
    tutorial_dir = original_dir.parent / "examples/tutorials"

    input_dir = tmp_path / "model_input"
    input_dir.mkdir()
    input_files = [
        "geodata_sample_2pop.csv",
        "ic_2pop.csv",
        "mobility_sample_2pop.csv",
    ]
    for fname in input_files:
        shutil.copyfile(tutorial_dir / "model_input" / fname, input_dir / fname)

    config_file = "config_sample_2pop_modifiers.yml"
    config_path = tmp_path / config_file
    shutil.copyfile(tutorial_dir / config_file, config_path)

    return ModelBuilder(
        config_path=config_path,
        ic_file=input_dir / "ic_2pop.csv",
        compartment_labels=["S", "E", "I", "R"],
        n_nodes=2,
        node_col="subpop",
        comp_col="mc_name",
        count_col="amount",
        allow_missing_subpops=True,
        allow_missing_compartments=True,
    )


def test_modelbuilder_initial_array(modelbuilder_from_config):
    y0 = modelbuilder_from_config.initial_array
    assert isinstance(y0, np.ndarray), "Output should be a NumPy array"
    assert y0.shape == (4, 2), f"Expected shape (4, 2), got {y0.shape}"
    assert np.all(y0 >= 0), "Initial condition values should be non-negative"
    assert np.isfinite(y0).all(), "Initial condition contains NaNs or infs"
    assert y0.sum() > 0, "Initial array should not be all zeros"


def test_modelbuilder_exact_values(modelbuilder_from_config):
    y0 = modelbuilder_from_config.initial_array
    assert y0[0, 1] == 1000.0  # "S" in small_province (node 1)
    assert y0[1, 0] == 5.0  # "E" in large_province (node 0)
    assert y0[0, 0] == 8995.0  # "S" in large_province


def test_compartment_index_mappings(modelbuilder_from_config):
    expected = {"S": 0, "E": 1, "I": 2, "R": 3}
    inverse = {v: k for k, v in expected.items()}
    assert modelbuilder_from_config.compartment_to_index == expected
    assert modelbuilder_from_config.index_to_compartment == inverse


def test_missing_compartment_raises(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bad_input")
    test_file = tmp_path / "ic_bad.csv"
    test_file.write_text("subpop,mc_name,amount\n0,X,5\n")

    with pytest.raises(ValueError, match="not in expected"):
        ModelBuilder(
            config_path=Path("dummy.yml"),
            ic_file=test_file,
            compartment_labels=["S", "E", "I", "R"],
            n_nodes=2,
            node_col="subpop",
            comp_col="mc_name",
            count_col="amount",
            allow_missing_compartments=False,
        )


def test_missing_subpop_raises_with_allowed_labels(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bad_input")
    test_file = tmp_path / "ic_bad.csv"

    # Use pandas to write properly-typed CSV with unknown label
    df = pd.DataFrame([{"subpop": "ghost_node", "mc_name": "S", "amount": 50}])
    df.to_csv(test_file, index=False)

    with pytest.raises(ValueError, match="Unknown node label"):
        from gempyor.builder_utilities import build_initial_array

        build_initial_array(
            ic_file=test_file,
            compartment_labels=["S", "E", "I", "R"],
            n_nodes=2,
            node_col="subpop",
            comp_col="mc_name",
            count_col="amount",
            allow_missing_subpops=False,
            allow_missing_compartments=False,
            proportional_ic=False,
            allowed_node_labels=["small_province", "large_province"],
        )


def test_missing_column_in_ic_file(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("bad_input")
    test_file = tmp_path / "ic_missing_col.csv"

    # Missing 'amount' column
    test_file.write_text("subpop,mc_name\nsmall_province,S\n")

    with pytest.raises(ValueError, match="Missing required columns"):
        ModelBuilder(
            config_path=tmp_path / "dummy.yml",
            ic_file=test_file,
            compartment_labels=["S", "E", "I", "R"],
            n_nodes=2,
            node_col="subpop",
            comp_col="mc_name",
            count_col="amount",
        )


def test_duplicate_entries_aggregate(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dupes")
    test_file = tmp_path / "ic_dupes.csv"

    df = pd.DataFrame(
        [
            {"subpop": "small_province", "mc_name": "S", "amount": 50},
            {"subpop": "small_province", "mc_name": "S", "amount": 150},
        ]
    )
    df.to_csv(test_file, index=False)

    builder = ModelBuilder(
        config_path=tmp_path / "dummy.yml",
        ic_file=test_file,
        compartment_labels=["S", "E", "I", "R"],
        n_nodes=2,
        node_col="subpop",
        comp_col="mc_name",
        count_col="amount",
        allow_missing_subpops=True,
        allowed_node_labels=["large_province", "small_province"],
    )

    node_idx = builder.allowed_node_labels.index("small_province")
    comp_idx = builder.compartment_to_index["S"]

    assert builder.initial_array[comp_idx, node_idx] == 200.0


def test_proportional_ic_raises(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("proportional")
    test_file = tmp_path / "ic_prop.csv"

    df = pd.DataFrame([{"subpop": "small_province", "mc_name": "S", "amount": 100}])
    df.to_csv(test_file, index=False)

    with pytest.raises(NotImplementedError):
        ModelBuilder(
            config_path=tmp_path / "dummy.yml",
            ic_file=test_file,
            compartment_labels=["S", "E", "I", "R"],
            n_nodes=2,
            node_col="subpop",
            comp_col="mc_name",
            count_col="amount",
            proportional_ic=True,
            allowed_node_labels=["small_province", "large_province"],
        )
