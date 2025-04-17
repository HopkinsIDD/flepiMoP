import numpy as np
import pandas as pd
import scipy.sparse
import logging
import os
import tempfile

import pytest

from gempyor import subpopulation_structure
from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


def test_subpopulation_structure_mobility():
    mobility_file = f"{DATA_DIR}/mobility.csv"

    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility.csv
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    subpop_struct = subpopulation_structure.SubpopulationStructure(config["subpop_setup"])

    mobility_data = pd.read_csv(mobility_file)
    mobility_data = mobility_data.pivot(index="ori", columns="dest", values="amount")
    mobility_data = mobility_data.fillna(0)
    assert np.array_equal(subpop_struct.mobility.toarray(), mobility_data.to_numpy())


def test_subpopulation_structure_mobility_txt():
    config.clear()
    config.read(user=False)
    mobility_file = f"{DATA_DIR}/mobility.txt"

    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility.csv
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    subpop_struct = subpopulation_structure.SubpopulationStructure(config["subpop_setup"])
    mobility_data = scipy.sparse.csr_matrix(np.loadtxt(mobility_file), dtype=int)
    assert np.array_equal(subpop_struct.mobility.toarray(), mobility_data.toarray())


def test_subpopulation_structure_subpop_population_zero_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata0.csv
            mobility: {DATA_DIR}/mobility.csv
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=(
            r"Input should be greater than 0 "
            r"\[type\=greater\_than\, input\_value\=0\, input\_type\=int]"
        ),
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


def test_subpopulation_structure_dulpicate_subpop_names_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata_dup.csv
            mobility: {DATA_DIR}/mobility.csv
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"The following subpopulation names are duplicated in the geodata file: .*",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


@pytest.mark.filterwarnings(
    "ignore:Mobility files as matrices are not recommended. "
    "Please switch to long form csv files.:PendingDeprecationWarning"
)
def test_subpopulation_structure_mobility_shape_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility_2x3.txt
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"^Mobility data has shape of .*, but should match geodata shape of .*.$",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


def test_subpopulation_structure_mobility_fluxes_same_ori_and_dest_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility_same_ori_dest.csv
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"Origin and destination subpopulations cannot be the same, '10001'",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


def test_subpopulation_structure_mobility_npz_shape_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility_2x3.npz
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"^Mobility data has shape of .*, but should match geodata shape of .*.$",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


def test_subpopulation_structure_mobility_no_extension_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"^Mobility data must either be either a txt, csv, or npz file, but was given mobility file of '.*'.$",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


def test_subpopulation_structure_mobility_exceed_source_node_pop_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility1001.csv
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"The following subpopulations have mobility exceeding their population.*",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


@pytest.mark.filterwarnings(
    "ignore:Mobility files as matrices are not recommended. "
    "Please switch to long form csv files.:PendingDeprecationWarning"
)
def test_subpopulation_structure_mobility_rows_exceed_source_node_pop_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata_3x3.csv
            mobility: {DATA_DIR}/mobility_row_exceeed.txt
    """

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"The following subpopulations have mobility exceeding their population.*",
    ):
        subpopulation_structure.SubpopulationStructure(config["subpop_setup"])


def test_subpopulation_structure_mobility_no_mobility_matrix_specified():
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
    """
    config.clear()
    config.read(user=False)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            config["subpop_setup"]
        )
    assert np.array_equal(subpop_struct.mobility.toarray(), np.zeros((2, 2)))
