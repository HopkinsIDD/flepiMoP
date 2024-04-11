import numpy as np
import pandas as pd
import scipy.sparse
import logging
import os

import pytest

from gempyor import subpopulation_structure

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


def test_subpopulation_structure_mobility():
    mobility_file = f"{DATA_DIR}/mobility.csv"
    subpop_struct = subpopulation_structure.SubpopulationStructure(
        setup_name=TEST_SETUP_NAME,
        geodata_file=f"{DATA_DIR}/geodata.csv",
        mobility_file=mobility_file,
        subpop_pop_key="population",
        subpop_names_key="subpop",
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
    mobility_file = f"{DATA_DIR}/mobility.txt"
    subpop_struct = subpopulation_structure.SubpopulationStructure(
        setup_name=TEST_SETUP_NAME,
        geodata_file=f"{DATA_DIR}/geodata.csv",
        mobility_file=mobility_file,
        subpop_pop_key="population",
        subpop_names_key="subpop",
    )

    mobility_data = scipy.sparse.csr_matrix(np.loadtxt(mobility_file), dtype=int)
    # mobility_data = mobility_data.pivot(index="ori", columns="dest", values="amount")
    # mobility_data = mobility_data.reindex(index=subpop_struct.subpop_names, columns=subpop_struct.subpop_names)
    # mobility_data = mobility_data.fillna(0)

    mobility_matrix = subpop_struct.mobility.toarray()  # convert to dense matrix

    print(subpop_struct.mobility.tocsr())
    print(mobility_data)

    assert np.array_equal(subpop_struct.mobility.toarray(), mobility_data.toarray())


def test_subpopulation_structure_not_existed_subpop_pop_key_fail():
    with pytest.raises(ValueError, match=r"subpop_pop_key.*does not correspond.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.csv",
            subpop_pop_key="population_not_existed",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_subpop_population_zero_fail():
    with pytest.raises(ValueError, match=r".*subpops with population zero.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata0.csv",
            mobility_file=f"{DATA_DIR}/mobility.csv",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_not_existed_subpop_names_key_fail():
    with pytest.raises(ValueError, match=r"subpop_names_key.*does not correspond.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.csv",
            subpop_pop_key="population",
            subpop_names_key="no_subpop",
        )


def test_subpopulation_structure_dulpicate_subpop_names_fail():
    with pytest.raises(ValueError, match=r"There are duplicate subpop_names.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata_dup.csv",
            mobility_file=f"{DATA_DIR}/mobility.csv",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_shape_fail():
    with pytest.raises(ValueError, match=r"mobility data must have dimensions of length of geodata.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility_2x3.txt",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_fluxes_same_ori_and_dest_fail():
    with pytest.raises(ValueError, match=r"Mobility fluxes with same origin and destination.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility_same_ori_dest.csv",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_npz_shape_fail():
    with pytest.raises(ValueError, match=r"mobility data must have dimensions of length of geodata.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility_2x3.npz",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_no_extension_fail():
    with pytest.raises(ValueError, match=r"Mobility data must either be a.*"):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_exceed_source_node_pop_fail():
    with pytest.raises(
        ValueError, match=r"The following entries in the mobility data exceed the source subpop populations.*"
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility1001.csv",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_rows_exceed_source_node_pop_fail():
    with pytest.raises(
        ValueError, match=r"The following entries in the mobility data exceed the source subpop populations.*"
    ):
        subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata_3x3.csv",
            mobility_file=f"{DATA_DIR}/mobility_row_exceeed.txt",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )


def test_subpopulation_structure_mobility_no_mobility_matrix_specified():
    subpop_struct = subpopulation_structure.SubpopulationStructure(
        setup_name=TEST_SETUP_NAME,
        geodata_file=f"{DATA_DIR}/geodata.csv",
        mobility_file=None,
        subpop_pop_key="population",
        subpop_names_key="subpop",
    )
    # target = np.array([[0, 0], [0, 0]]) # 2x2, just in this case
    assert np.array_equal(subpop_struct.mobility.toarray(), np.zeros((2, 2)))
