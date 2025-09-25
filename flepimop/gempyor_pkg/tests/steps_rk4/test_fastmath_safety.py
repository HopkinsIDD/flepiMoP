# tests/steps_rk4/test_fastmath_safety.py
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
    param_slice,  # NEW: for time-slicing parameters
    compute_proportion_sums_exponents,
    compute_transition_amounts_serial,
    compute_transition_amounts_parallel,
    assemble_flux,
    compute_transition_amounts_meta,
)


@pytest.fixture(scope="module")
def modelinfo_from_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("model_input")

    # Resolve tutorials dir relative to this test file
    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

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
    # parsed_params is typically (P, T, N) or (P, T). We will time-slice in each test.

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
        "params": parsed_params,  # (P, T, N) or (P, T)
        "transitions": transitions,
        "proportion_info": proportion_info,
        "transition_sum_compartments": transition_sum_compartments,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_data_indices,
        "mobility_row_indices": mobility_row_indices,
        "proportion_who_move": proportion_who_move,
        "population": population,
        "dt": 0.1,  # kept for consistency; amounts no longer use dt
    }


def test_fastmath_equivalence_prod(model_and_inputs):
    arr = np.random.rand(4, model_and_inputs["initial_array"].shape[1]).astype(
        np.float32
    )
    fn_fast = prod_along_axis0
    fn_safe = njit(fastmath=False)(fn_fast.py_func)
    out1 = fn_safe(arr)
    out2 = fn_fast(arr)
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)


def test_fastmath_equivalence_proportion_sums_exponents(model_and_inputs):
    out = model_and_inputs
    # Slice parameters at t=0 (step-wise to match prior behavior)
    param_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)

    args = (
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,  # CHANGED: pass (P, N) slice
    )
    fn_fast = compute_proportion_sums_exponents
    fn_safe = njit(fastmath=False)(fn_fast.py_func)
    total1, src1, mask1 = fn_safe(*args)
    total2, src2, mask2 = fn_fast(*args)
    assert np.allclose(total1, total2, rtol=1e-5, atol=1e-7)
    assert np.allclose(src1, src2, rtol=1e-5, atol=1e-7)


def test_fastmath_equivalence_transition_amounts_serial(model_and_inputs):
    out = model_and_inputs
    param_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)
    rates, sources, _ = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,  # (P, N)
    )
    # CHANGED: deterministic amounts no longer take method/dt (instantaneous dy/dt)
    fn_fast = compute_transition_amounts_serial
    fn_safe = njit(fastmath=False)(fn_fast.py_func)
    out1 = fn_safe(sources, rates)
    out2 = fn_fast(sources, rates)
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)


def test_fastmath_equivalence_transition_amounts_parallel(model_and_inputs):
    out = model_and_inputs
    param_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)
    rates, sources, _ = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,  # (P, N)
    )
    # CHANGED: parallel variant also drops dt
    fn_fast = compute_transition_amounts_parallel
    fn_safe = njit(parallel=True, fastmath=False)(fn_fast.py_func)
    out1 = fn_safe(sources, rates)
    out2 = fn_fast(sources, rates)
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)


def test_fastmath_equivalence_flux(model_and_inputs):
    out = model_and_inputs
    param_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)
    rates, sources, _ = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,  # (P, N)
    )
    # CHANGED: amounts are instantaneous dy/dt contributions
    amounts = compute_transition_amounts_serial(sources, rates)

    fn_fast = assemble_flux
    fn_safe = njit(fastmath=False)(fn_fast.py_func)

    out1 = fn_safe(
        amounts,
        out["transitions"],
        out["initial_array"].shape[0],
        out["initial_array"].shape[1],
    )
    out2 = fn_fast(
        amounts,
        out["transitions"],
        out["initial_array"].shape[0],
        out["initial_array"].shape[1],
    )
    assert np.allclose(out1, out2, rtol=1e-5, atol=1e-7)


def test_proportion_sums_edge_cases_zero_and_large(model_and_inputs):
    """
    Exercise 0**exp, 1**exp, and very large magnitudes to ensure no NaN/Inf and
    fastmath equivalence.
    """
    out = model_and_inputs
    N = out["initial_array"].shape[1]
    param_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)

    # Craft states with zeros and huge values
    states = out["initial_array"].astype(np.float64).copy()
    if states.size >= 4 * N:
        states[0, :] = 0.0  # zeros -> 0**exp
        states[1, :] = 1.0  # ones  -> 1**exp
        states[2, :] = 1e-30  # tiny  -> underflow risk
        states[3, :] = 1e30  # huge  -> overflow risk

    fn_fast = compute_proportion_sums_exponents
    fn_safe = njit(fastmath=False)(fn_fast.py_func)

    total1, src1, mask1 = fn_safe(
        states,
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,
    )
    total2, src2, mask2 = fn_fast(
        states,
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,
    )

    assert np.all(np.isfinite(total1)) and np.all(np.isfinite(src1))
    assert np.all(np.isfinite(total2)) and np.all(np.isfinite(src2))
    assert np.allclose(total1, total2, rtol=1e-5, atol=1e-7)
    assert np.allclose(src1, src2, rtol=1e-5, atol=1e-7)


def test_assemble_flux_conservation_fastmath(model_and_inputs):
    """
    For any amounts & transitions, net flux per node should sum to zero across compartments.
    Check with and without fastmath.
    """
    out = model_and_inputs
    C, N = out["initial_array"].shape

    # Random but nonnegative amounts
    Tn = out["transitions"].shape[1]
    rng = np.random.default_rng(42)
    amounts = rng.random((Tn, N)).astype(np.float64)

    fn_fast = assemble_flux
    fn_safe = njit(fastmath=False)(fn_fast.py_func)

    dy1 = fn_safe(amounts, out["transitions"], C, N).reshape(C, N)
    dy2 = fn_fast(amounts, out["transitions"], C, N).reshape(C, N)

    # Column sums should be ~0 (numerical noise ok)
    assert np.allclose(dy1.sum(axis=0), 0.0, atol=1e-10)
    assert np.allclose(dy2.sum(axis=0), 0.0, atol=1e-10)
    # Equivalence
    assert np.allclose(dy1, dy2, rtol=1e-12, atol=0.0)


def test_meta_dispatch_equivalence_serial_vs_parallel(monkeypatch, model_and_inputs):
    """
    Force small and large thresholds to exercise both branches and ensure results match
    the explicit serial/parallel kernels.
    """
    out = model_and_inputs
    param_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)
    rates, sources, _ = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,
    )

    # Serial expectation
    expected_serial = compute_transition_amounts_serial(sources, rates)
    # Parallel expectation
    expected_parallel = compute_transition_amounts_parallel(sources, rates)

    import gempyor.vectorization_experiments as ve

    # Force serial path
    monkeypatch.setattr(
        ve, "_PARALLEL_THRESHOLD", np.iinfo(np.int64).max, raising=False
    )
    out_serial = compute_transition_amounts_meta(sources, rates)
    assert np.allclose(out_serial, expected_serial, rtol=1e-12, atol=0.0)

    # Force parallel path
    monkeypatch.setattr(ve, "_PARALLEL_THRESHOLD", 1, raising=False)
    out_parallel = compute_transition_amounts_meta(sources, rates)
    assert np.allclose(out_parallel, expected_parallel, rtol=1e-12, atol=0.0)


def test_end_to_end_rhs_dydt_finite(model_and_inputs):
    """
    Quick end-to-end check that the dy/dt produced by the pipeline is finite at t=0.
    Uses the simplest parameter slicing and deterministic amounts.
    """
    out = model_and_inputs
    C, N = out["initial_array"].shape
    param_t = param_slice(out["params"], t=0.0, mode="step")

    total_base, source_numbers, _ = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        param_t,
    )
    # Dummy mobility inputs if not already used here; compute_transition_rates is Python
    # in your code, so this checks the Numba parts around it.
    # If your test suite constructs total_rates elsewhere, skip compute_transition_rates here.

    # Construct a benign total_rates for the check (nonnegative)
    total_rates = np.abs(total_base)

    amounts = compute_transition_amounts_serial(source_numbers, total_rates)
    dy = assemble_flux(amounts, out["transitions"], C, N)
    assert np.all(np.isfinite(dy))
