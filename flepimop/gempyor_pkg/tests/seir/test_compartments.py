import numpy as np
import os
import pytest
import warnings
import shutil


import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
import filecmp

from gempyor import compartments, seir, NPI, file_paths, model_info, subpopulation_structure

from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


def test_check_transitions_parquet_creation():
    config.clear()
    config.read(user=False)
    config.set_file(f"{DATA_DIR}/config_compartmental_model_format.yml")
    original_compartments_file = f"{DATA_DIR}/parsed_compartment_compartments.parquet"
    original_transitions_file = f"{DATA_DIR}/parsed_compartment_transitions.parquet"
    lhs = compartments.Compartments(
        seir_config=config["seir"], compartments_config=config["compartments"]
    )
    rhs = compartments.Compartments(
        seir_config=config["seir"],
        compartments_file=original_compartments_file,
        transitions_file=original_transitions_file,
    )

    assert lhs.times_set == 1
    assert rhs.times_set == 1
    # assert(lhs.parameters == rhs.parameters) ## parameters objects do not have an == operator
    assert (lhs.compartments == rhs.compartments).all().all()
    assert (lhs.transitions == rhs.transitions).all().all()
    assert lhs == rhs


def test_check_transitions_parquet_writing_and_loading():
    config.clear()
    config.read(user=False)
    config.set_file(f"{DATA_DIR}/config_compartmental_model_format.yml")
    lhs = compartments.Compartments(
        seir_config=config["seir"], compartments_config=config["compartments"]
    )
    temp_compartments_file = f"{DATA_DIR}/parsed_compartment_compartments.test.parquet"
    temp_transitions_file = f"{DATA_DIR}/parsed_compartment_transitions.test.parquet"
    lhs.toFile(
        compartments_file=temp_compartments_file,
        transitions_file=temp_transitions_file,
        write_parquet=True,
    )
    rhs = compartments.Compartments(
        seir_config=config["seir"],
        compartments_file=temp_compartments_file,
        transitions_file=temp_transitions_file,
    )

    assert lhs.times_set == 1
    assert rhs.times_set == 1
    assert (lhs.compartments == rhs.compartments).all().all()
    assert (lhs.transitions == rhs.transitions).all().all()
    assert lhs == rhs


@pytest.mark.filterwarnings(
    "ignore:Mobility files as matrices are not recommended. "
    "Please switch to long form csv files.:PendingDeprecationWarning"
)
def test_ModelInfo_has_compartments_component():
    os.chdir(os.path.dirname(__file__))
    config.clear()
    config.read(user=False)
    config.set_file(f"{DATA_DIR}/config.yml")

    s = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
    )
    assert type(s.compartments) == compartments.Compartments
    assert type(s.compartments) == compartments.Compartments

    config.clear()
    config.read(user=False)
    config.set_file(f"{DATA_DIR}/config_compartmental_model_full.yml")

    s = model_info.ModelInfo(
        config=config,
        nslots=1,
        write_csv=False,
    )
    assert type(s.compartments) == compartments.Compartments
    assert type(s.compartments) == compartments.Compartments
