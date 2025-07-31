import numpy as np
import pytest
from numba import njit
from scipy.sparse import csr_matrix
from pathlib import Path
import shutil
import confuse

from gempyor.model_info import ModelInfo
from gempyor.vectorization_experiments import (
    prod_along_axis0,
    compute_proportion_sums_exponents,
    compute_transition_amounts_serial,
    compute_transition_amounts_parallel,
    assemble_flux,
)


@pytest.fixture(scope="module")
def modelinfo_from_config(tmp_path_factory):
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

    config = confuse.Configuration("TestModel", __name__)
    config.set_file(str(config_path))

    return (
        ModelInfo(
            config=config,
            config_filepath=str(config_path),
            path_prefix=str(tmp_path),
            setup_name="sample_2pop",
            seir_modifiers_scenario="Ro_all",
        ),
        config,
    )


@pytest.fixture
def model_and_inputs(modelinfo_from_config):
    model, config = modelinfo_from_config

    initial_array = model.initial_conditions.get_from_config(sim_id=0, modinf=model)

    (unique_strings, transitions, transition_sum_compartments, proportion_info) = (
        model.compartments.get_transition_array()
    )

    parsed_params = model.compartments.parse_parameters(
        model.parameters.parameters_quick_draw(model.n_days, model.nsubpops),
        config["seir"]["parameters"].get(),
        unique_strings,
    )

    mobility_csr: csr_matrix = model.mobility
    population = model.subpop_pop

    mobility_data = mobility_csr.data
    mobility_data_indices = mobility_csr.indptr
    mobility_row_indices = mobility_csr.indices

    proportion_who_move = np.zeros(model.nsubpops)
    for i in range(model.nsubpops):
        total_flux = mobility_data[
            mobility_data_indices[i] : mobility_data_indices[i + 1]
        ].sum()
        proportion_who_move[i] = min(total_flux / population[i], 1.0)




    return {
        "initial_array": initial_array,
        "params": parsed_params,
        "transitions": transitions,
        "proportion_info": proportion_info,
        "transition_sum_compartments": transition_sum_compartments,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_data_indices,
        "mobility_row_indices": mobility_row_indices,
        "proportion_who_move": proportion_who_move,
        "population": population,
        "dt": 0.1,
    }

def test_fastmath_equivalence_prod(model_and_inputs):
    arr = np.random.rand(4, model_and_inputs["initial_array"].shape[1]).astype(np.float32)
    fn_fast = prod_along_axis0
    fn_safe = njit(fastmath=False)(fn_fast.py_func)
    out1 = fn_safe(arr)
    out2 = fn_fast(arr)
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)

def test_fastmath_equivalence_proportion_sums_exponents(model_and_inputs):
    out = model_and_inputs
    args = (
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        0,
    )
    fn_fast = compute_proportion_sums_exponents
    fn_safe = njit(fastmath=False)(fn_fast.py_func)
    total1, src1 = fn_safe(*args)
    total2, src2 = fn_fast(*args)
    assert np.allclose(total1, total2, rtol=1e-5, atol=1e-7)
    assert np.allclose(src1, src2, rtol=1e-5, atol=1e-7)

def test_fastmath_equivalence_transition_amounts_serial(model_and_inputs):
    out = model_and_inputs
    rates, sources = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        0,
    )
    for method in ["rk4", "euler"]:
        fn_fast = compute_transition_amounts_serial
        fn_safe = njit(fastmath=False)(fn_fast.py_func)
        out1 = fn_safe(sources, rates, method, out["dt"])
        out2 = fn_fast(sources, rates, method, out["dt"])
        assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)

def test_fastmath_equivalence_transition_amounts_parallel(model_and_inputs):
    out = model_and_inputs
    rates, sources = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        0,
    )
    fn_fast = compute_transition_amounts_parallel
    fn_safe = njit(parallel=True, fastmath=False)(fn_fast.py_func)
    out1 = fn_safe(sources, rates, out["dt"])
    out2 = fn_fast(sources, rates, out["dt"])
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)

def test_fastmath_equivalence_flux(model_and_inputs):
    out = model_and_inputs
    rates, sources = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        0,
    )
    amounts = compute_transition_amounts_serial(sources, rates, "euler", out["dt"])
    fn_fast = assemble_flux
    fn_safe = njit(fastmath=False)(fn_fast.py_func)
    out1 = fn_safe(amounts, out["transitions"], out["initial_array"].shape[0], out["initial_array"].shape[1])
    out2 = fn_fast(amounts, out["transitions"], out["initial_array"].shape[0], out["initial_array"].shape[1])
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)
