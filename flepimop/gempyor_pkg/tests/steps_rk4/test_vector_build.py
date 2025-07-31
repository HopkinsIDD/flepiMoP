import pytest
import numpy as np
import shutil
from pathlib import Path
import confuse
from scipy.sparse import csr_matrix

from gempyor.model_info import ModelInfo
from gempyor.vectorization_experiments import (
    compute_proportion_sums_exponents,
    compute_transition_rates,
    compute_transition_amounts_meta,
    assemble_flux,
    run_solver,
    build_rhs,
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

    # This is the correct call — no .parameter_names used
    parsed_params = model.compartments.parse_parameters(
        model.parameters.parameters_quick_draw(model.n_days, model.nsubpops),
        config["seir"]["parameters"].get(),
        unique_strings,
    )

    # === Mobility setup ===
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

    # === Time grid ===
    offset = model.ti.toordinal()
    t0 = 0  # start at day 0
    t1 = model.tf.toordinal() - offset  # number of days between tf and ti
    dt = 0.1
    n_steps = int((t1 - t0) / dt) + 1
    time_grid = np.linspace(t0, t1, n_steps).astype(np.float32)

    out = {
        "model": model,
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
        "time_grid": time_grid,
        "dt": dt,
        "percent_day_away": 0.5,
        "today_idx": 0,
    }

    return out


import numpy as np
import pytest

from gempyor.vectorization_experiments import (
    compute_proportion_sums_exponents,
    compute_transition_rates,
    compute_transition_amounts_meta,
    assemble_flux,
    run_solver,
    build_rhs,
)


def test_proportion_sums_and_sources(model_and_inputs):
    out = model_and_inputs
    prop_sums, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        out["today_idx"],
    )
    assert prop_sums.shape == source_nums.shape
    assert np.all(np.isfinite(prop_sums))
    assert np.all(source_nums >= 0)


def test_transition_rate_shape_and_finiteness(model_and_inputs):
    out = model_and_inputs

    prop_sums, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        out["today_idx"],
    )
    rates = compute_transition_rates(
        prop_sums,
        source_nums,
        out["transitions"],
        out["params"],
        out["today_idx"],
        out["percent_day_away"],
        out["proportion_who_move"],
        out["mobility_data"],
        out["mobility_data_indices"],
        out["mobility_row_indices"],
        out["population"],
    )
    assert rates.shape == source_nums.shape
    assert np.all(np.isfinite(rates))


def test_transition_amounts_meta_rk4(model_and_inputs):
    out = model_and_inputs
    _, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        out["today_idx"],
    )
    rates = np.ones_like(source_nums)
    amounts = compute_transition_amounts_meta(
        source_nums, rates, method="rk4", dt=out["dt"]
    )
    assert amounts.shape == rates.shape
    assert np.all(amounts >= 0)


def test_flux_assembly_validity(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    ntrans = len(out["transitions"])
    mock_amounts = np.random.rand(ntrans, nloc).astype(np.float32)
    flux = assemble_flux(mock_amounts, out["transitions"], ncomp, nloc)
    assert flux.shape == (ncomp * nloc,)
    assert np.all(np.isfinite(flux))


def test_run_solver_vectorized_backend(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    rhs_fn = build_rhs(
        ncomp,
        nloc,
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        "rk4",
        out["dt"],
        out["percent_day_away"],
        out["proportion_who_move"],
        out["mobility_data"],
        out["mobility_data_indices"],
        out["mobility_row_indices"],
        out["population"],
    )
    states, fluxes = run_solver(
        rhs_fn,
        out["initial_array"],
        out["time_grid"],
        method="rk4",
        record_daily=True,
        ncompartments=ncomp,
        nspatial_nodes=nloc,
    )

    assert states.shape == (len(out["time_grid"]), ncomp, nloc)
    assert fluxes.shape == states.shape
    assert np.all(np.isfinite(states))
    assert np.all(np.isfinite(fluxes))


def test_rhs_output_matches_input_shape(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    rhs_fn = build_rhs(
        ncomp,
        nloc,
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        "rk4",
        out["dt"],
        out["percent_day_away"],
        out["proportion_who_move"],
        out["mobility_data"],
        out["mobility_data_indices"],
        out["mobility_row_indices"],
        out["population"],
    )
    flat_y = out["initial_array"].ravel()
    dy = rhs_fn(out["today_idx"], flat_y)
    assert dy.shape == flat_y.shape
    assert np.all(np.isfinite(dy))


def test_mass_conservation_for_dummy_flux(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    _, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        out["params"],
        out["today_idx"],
    )
    dummy_rates = np.ones_like(source_nums)
    dummy_amounts = compute_transition_amounts_meta(
        source_nums, dummy_rates, method="rk4", dt=out["dt"]
    )
    flux = assemble_flux(dummy_amounts, out["transitions"], ncomp, nloc)
    total_flux_per_node = flux.reshape((ncomp, nloc)).sum(axis=0)
    assert np.allclose(total_flux_per_node, 0.0, atol=1e-5)
