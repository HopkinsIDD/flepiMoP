import datetime
import numpy as np
import os
import pandas as pd
import pytest
import confuse

from gempyor import setup, subpopulation_structure

from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestSubpopulationStructure:
    def test_SubpopulationStructure_success(self):
        ss = subpopulation_structure.SubpopulationStructure(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.txt",
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )

    def test_bad_subpop_pop_key_fail(self):
        # Bad subpop_pop_key error
        with pytest.raises(ValueError, match=r".*subpop_pop_key.*"):
            subpopulation_structure.SubpopulationStructure(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_small.txt",
                subpop_pop_key="wrong",
                subpop_names_key="subpop",
            )

    def test_bad_subpop_names_key_fail(self):
        with pytest.raises(ValueError, match=r".*subpop_names_key.*"):
            subpopulation_structure.SubpopulationStructure(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility.txt",
                subpop_pop_key="population",
                subpop_names_key="wrong",
            )

    def test_mobility_dimensions_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*dimensions.*"):
            subpopulation_structure.SubpopulationStructure(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_small.txt",
                subpop_pop_key="population",
                subpop_names_key="subpop",
            )

    def test_mobility_too_big_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*population.*"):
            subpopulation_structure.SubpopulationStructure(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_big.txt",
                subpop_pop_key="population",
                subpop_names_key="subpop",
            )
