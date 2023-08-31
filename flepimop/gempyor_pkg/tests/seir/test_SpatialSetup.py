import datetime
import numpy as np
import os
import pandas as pd
import pytest
import confuse

from gempyor import setup

from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestSpatialSetup:
    def test_SpatialSetup_success(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.txt", # but warning message presented
            popnodes_key="population",
            nodenames_key="geoid",
        )
    def test_SpatialSetup_success2(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.csv",
            popnodes_key="population",
            nodenames_key="geoid",
        )
    '''
    def test_SpatialSetup_npz_success3(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.npz",
            popnodes_key="population",
            nodenames_key="geoid",
        )
    '''
    def test_SpatialSetup_wihout_mobility_success3(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility0.csv",
            popnodes_key="population",
            nodenames_key="geoid",
        )

    def test_bad_popnodes_key_fail(self):
        # Bad popnodes_key error
        with pytest.raises(ValueError, match=r".*popnodes_key.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_small.txt",
                popnodes_key="wrong",
                nodenames_key="geoid",
            )

    def test_population_0_nodes_fail(self):
        with pytest.raises(ValueError, match=r".*population.*zero.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata0.csv",
                mobility_file=f"{DATA_DIR}/mobility.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_fileformat_fail(self):
        with pytest.raises(ValueError, match=r".*Mobility.*longform.*matrix.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_bad_nodenames_key_fail(self):
        with pytest.raises(ValueError, match=r".*nodenames_key.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility.txt",
                popnodes_key="population",
                nodenames_key="wrong",
            )

    def test_duplicate_nodenames_key_fail(self):
        with pytest.raises(ValueError, match=r".*duplicate.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata_dup.csv",
                mobility_file=f"{DATA_DIR}/mobility.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_shape_in_npz_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*Actual.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_2x3.npz",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_dimensions_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*dimensions.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_small.txt",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_same_ori_dest_fail(self):
        with pytest.raises(ValueError, match=r".*Mobility.*same.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_same_ori_dest.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_too_big_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*population.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_big.txt",
                popnodes_key="population",
                nodenames_key="geoid",
            )
    def test_mobility_data_exceeded_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*exceed.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility1001.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )
