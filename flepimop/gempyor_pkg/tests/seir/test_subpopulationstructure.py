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

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    subpop_struct = subpopulation_structure.SubpopulationStructure(
        setup_name=TEST_SETUP_NAME,
        subpop_config=config["subpop_setup"],
    )

    mobility_data = pd.read_csv(mobility_file)
    mobility_data = mobility_data.pivot(index="ori", columns="dest", values="amount")
    # mobility_data = mobility_data.reindex(index=subpop_struct.subpop_names, columns=subpop_struct.subpop_names)
    mobility_data = mobility_data.fillna(0)

    mobility_matrix = subpop_struct.mobility.toarray()  # convert to dense matrix

    # print(subpop_struct.mobility.toarray())
    # print(mobility_data.to_numpy())

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

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    subpop_struct = subpopulation_structure.SubpopulationStructure(
        setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
    )
    mobility_data = scipy.sparse.csr_matrix(np.loadtxt(mobility_file), dtype=int)
    # mobility_data = mobility_data.pivot(index="ori", columns="dest", values="amount")
    # mobility_data = mobility_data.reindex(index=subpop_struct.subpop_names, columns=subpop_struct.subpop_names)
    # mobility_data = mobility_data.fillna(0)

    mobility_matrix = subpop_struct.mobility.toarray()  # convert to dense matrix

    print(subpop_struct.mobility.tocsr())
    print(mobility_data)

    assert np.array_equal(subpop_struct.mobility.toarray(), mobility_data.toarray())


def test_subpopulation_structure_subpop_population_zero_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata0.csv
            mobility: {DATA_DIR}/mobility.csv
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(ValueError, match=r".*subpops with population zero.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            subpop_config=config["subpop_setup"],
        )


def test_subpopulation_structure_dulpicate_subpop_names_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata_dup.csv
            mobility: {DATA_DIR}/mobility.csv
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(ValueError, match=r"There are duplicate subpop_names.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_shape_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility_2x3.txt
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError, match=r"mobility data must have dimensions of length of geodata.*"
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_fluxes_same_ori_and_dest_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility_same_ori_dest.csv
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError, match=r"Mobility fluxes with same origin and destination.*"
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_npz_shape_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility_2x3.npz
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError, match=r"mobility data must have dimensions of length of geodata.*"
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_no_extension_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(ValueError, match=r"Mobility data must either be a.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_exceed_source_node_pop_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
            mobility: {DATA_DIR}/mobility1001.csv
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"The following entries in the mobility data exceed the source subpop populations.*",
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_rows_exceed_source_node_pop_fail():
    config.clear()
    config.read(user=False)
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata_3x3.csv
            mobility: {DATA_DIR}/mobility_row_exceeed.txt
    """

    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

    with pytest.raises(
        ValueError,
        match=r"The following entries in the mobility data exceed the source subpop populations.*",
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )


def test_subpopulation_structure_mobility_no_mobility_matrix_specified():
    subpop_config_str = f"""
        subpop_setup:
            geodata: {DATA_DIR}/geodata.csv
    """
    config.clear()
    config.read(user=False)
    with tempfile.NamedTemporaryFile(
        delete=False
    ) as temp_file:  # Creates a temporary file
        temp_file.write(subpop_config_str.encode("utf-8"))  # Write the content
        temp_file.close()  # Ensure the file is closed
        config.set_file(temp_file.name)  # Load from the temporary file path

        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME, subpop_config=config["subpop_setup"]
        )

    # target = np.array([[0, 0], [0, 0]]) # 2x2, just in this case
    assert np.array_equal(subpop_struct.mobility.toarray(), np.zeros((2, 2)))
