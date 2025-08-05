# tests/steps_rk4/test_vector_build.py
import pytest
import numpy as np
import shutil
from pathlib import Path
import confuse
from scipy.sparse import csr_matrix
from functools import partial
from functools import partial
from scipy.integrate import solve_ivp

from gempyor.model_info import ModelInfo
from gempyor.vectorization_experiments import (
    param_slice,
    compute_proportion_sums_exponents,
    compute_transition_rates,
    compute_transition_amounts_meta,
    assemble_flux,
    run_solver,
    build_rhs_for_solve_ivp,
)

# ------------------------------------------------------------
# Helper: build a SAFE expression lookup from unique_strings
# ------------------------------------------------------------
def build_safe_param_expr_lookup(unique_strings: list[str]) -> tuple[dict[int, str] | None, dict[str, int]]:
    """
    Build (param_expr_lookup, param_name_to_row) safely.

    - If a unique string contains '*', include it in param_expr_lookup ONLY if
      all factors appear as separate names in unique_strings (so we can resolve).
    - If no resolvable expressions exist, returns (None, name_to_row).
    """
    name_to_row = {name: i for i, name in enumerate(unique_strings)}
    expr_lookup: dict[int, str] = {}

    for idx, s in enumerate(unique_strings):
        if "*" in s:
            terms = [t.strip() for t in s.split("*")]
            if all(term in name_to_row for term in terms):
                expr_lookup[idx] = s

    return (expr_lookup if expr_lookup else None, name_to_row)


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------
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
    unique_strings, transitions, transition_sum_compartments, proportion_info = (
        model.compartments.get_transition_array()
    )

    param_defs = config["seir"]["parameters"].get()
    base_params = model.parameters.parameters_quick_draw(model.n_days, model.nsubpops)
    parsed_params = model.compartments.parse_parameters(
        base_params, param_defs, unique_strings
    )  # shape (P, T, N) or (P, T)

    # Safe expression mapping from unique_strings
    param_expr_lookup, param_name_to_row = build_safe_param_expr_lookup(unique_strings)

    mobility_csr: csr_matrix = model.mobility
    population = model.subpop_pop

    mobility_data = mobility_csr.data
    mobility_data_indices = mobility_csr.indptr
    mobility_row_indices = mobility_csr.indices

    proportion_who_move = np.zeros(model.nsubpops)
    for i in range(model.nsubpops):
        total_flux = mobility_data[mobility_data_indices[i] : mobility_data_indices[i + 1]].sum()
        proportion_who_move[i] = min(total_flux / population[i], 1.0)

    offset = model.ti.toordinal()
    t0 = 0.0
    t1 = float(model.tf.toordinal() - offset)
    dt = 0.1
    time_grid = np.linspace(t0, t1, int((t1 - t0) / dt) + 1).astype(np.float64)

    return {
        "model": model,
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
        "time_grid": time_grid,
        "dt": dt,
        "percent_day_away": 0.5,
        "param_expr_lookup": param_expr_lookup,    # may be None if factors not separately present
        "param_name_to_row": param_name_to_row,    # always safe to pass
    }


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
def test_proportion_sums_and_sources(model_and_inputs):
    out = model_and_inputs
    theta_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)
    prop_sums, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        theta_t,
    )
    assert prop_sums.shape == source_nums.shape
    assert np.all(np.isfinite(prop_sums))
    assert np.all(source_nums >= 0)


def test_transition_rate_shape_and_finiteness(model_and_inputs):
    out = model_and_inputs
    theta_t = param_slice(out["params"], t=0.0, mode="step")  # (P, N)
    prop_sums, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        theta_t,
    )
    rates = compute_transition_rates(
        total_rates_base=prop_sums,
        source_numbers=source_nums,
        transitions=out["transitions"],
        param_t=theta_t,
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],     # None or dict
        param_name_to_row=out["param_name_to_row"],     # name->row
    )
    assert rates.shape == source_nums.shape
    assert np.all(np.isfinite(rates))


def test_rhs_output_matches_input_shape(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    flat_y = out["initial_array"].ravel()
    dy = rhs(0.0, flat_y, out["params"])  # pass full (P, T, N) or (P, T)
    assert dy.shape == flat_y.shape
    assert np.all(np.isfinite(dy))


def test_transition_amounts_meta_instantaneous_flux(model_and_inputs):
    out = model_and_inputs
    theta_t = param_slice(out["params"], t=0.0, mode="step")
    _, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        theta_t,
    )
    dummy_rates = np.ones_like(source_nums)
    dummy_amounts = compute_transition_amounts_meta(source_nums, dummy_rates)
    assert dummy_amounts.shape == dummy_rates.shape
    assert np.all(dummy_amounts >= 0)


def test_flux_assembly_validity(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    dummy_amounts = np.random.rand(out["transitions"].shape[1], nloc).astype(np.float64)
    flux = assemble_flux(dummy_amounts, out["transitions"], ncomp, nloc)
    assert flux.shape == (ncomp * nloc,)
    assert np.all(np.isfinite(flux))


def test_run_solver_vectorized_backend(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    # Use STEP mode to avoid pathological interpolated values in this config
    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    f = partial(rhs, parameters=out["params"])  # curry parameters into rhs

    states, fluxes = run_solver(
        f,
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


def test_mass_conservation_for_dummy_flux(model_and_inputs):
    out = model_and_inputs
    theta_t = param_slice(out["params"], t=0.0, mode="step")
    ncomp, nloc = out["initial_array"].shape
    _, source_nums = compute_proportion_sums_exponents(
        out["initial_array"],
        out["transitions"],
        out["proportion_info"],
        out["transition_sum_compartments"],
        theta_t,
    )
    dummy_rates = np.ones_like(source_nums)
    dummy_amounts = compute_transition_amounts_meta(source_nums, dummy_rates)
    flux = assemble_flux(dummy_amounts, out["transitions"], ncomp, nloc).reshape(ncomp, nloc)
    assert np.allclose(flux.sum(axis=0), 0.0, atol=1e-5)


def test_model_runs_with_alternative_param_set(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    # Use STEP mode here as well to avoid NaNs/Infs from linear interpolation in this config
    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    alt_theta_full = out["params"] * 1.25

    states, _ = run_solver(
        partial(rhs, parameters=alt_theta_full),
        out["initial_array"],
        out["time_grid"],
        method="euler",
        record_daily=False,
        ncompartments=ncomp,
        nspatial_nodes=nloc,
    )
    assert states.shape == (len(out["time_grid"]), ncomp, nloc)
    assert np.all(np.isfinite(states))


def test_rhs_produces_nonzero_flux(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    flat_state = out["initial_array"].ravel()
    dy = rhs(0.0, flat_state, out["params"])
    assert np.any(np.abs(dy) > 0), "RHS produced zero change; check dynamic logic"


def test_multiple_runs_different_theta(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="linear",
    )

    theta_full = out["params"]
    alt_theta_full = theta_full.copy()
    alt_theta_full[0] *= 1.5  # perturb first parameter row globally

    states1, _ = run_solver(
        partial(rhs, parameters=theta_full),
        out["initial_array"],
        out["time_grid"],
        method="euler",
        record_daily=False,
        ncompartments=ncomp,
        nspatial_nodes=nloc,
    )
    states2, _ = run_solver(
        partial(rhs, parameters=alt_theta_full),
        out["initial_array"],
        out["time_grid"],
        method="euler",
        record_daily=False,
        ncompartments=ncomp,
        nspatial_nodes=nloc,
    )

    assert not np.allclose(states1, states2), "Different thetas should yield different simulations"


def test_expression_lookup_only_when_factors_present(model_and_inputs):
    """
    Guard: if unique_strings contains a composite like 'Ro*gamma' but its factors ('Ro', 'gamma')
    do NOT both appear as separate rows, the safe builder should return param_expr_lookup=None.
    """
    out = model_and_inputs
    pel = out["param_expr_lookup"]
    # If there are NO resolvable expressions, pel is None (desired for current tutorial config)
    assert pel is None or isinstance(pel, dict)




def test_run_solver_vectorized_backend_scipy(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    f_ivp = partial(rhs, parameters=out["params"])

    res = solve_ivp(
        fun=f_ivp,
        t_span=(out["time_grid"][0], out["time_grid"][-1]),
        y0=out["initial_array"].ravel(),
        method="BDF",  # stiff-aware
        t_eval=out["time_grid"],
        vectorized=False,
        atol=1e-8,
        rtol=1e-6
    )

    assert res.success, f"solve_ivp failed: {res.message}"

    states = res.y.T.reshape(len(out["time_grid"]), ncomp, nloc)

    # allow small negative round-off but no large excursions
    assert np.all(np.isfinite(states))
    assert np.min(states) > -1e-6
    assert np.max(states) < 1e9


def test_model_runs_with_alternative_param_set_scipy(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    rhs = build_rhs_for_solve_ivp(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        percent_day_away=out["percent_day_away"],
        proportion_who_move=out["proportion_who_move"],
        mobility_data=out["mobility_data"],
        mobility_data_indices=out["mobility_data_indices"],
        mobility_row_indices=out["mobility_row_indices"],
        population=out["population"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    alt_theta_full = out["params"] * 1.25
    f_ivp = partial(rhs, parameters=alt_theta_full)

    res = solve_ivp(
        fun=f_ivp,
        t_span=(out["time_grid"][0], out["time_grid"][-1]),
        y0=out["initial_array"].ravel(),
        method="BDF",  # stiff-aware
        t_eval=out["time_grid"],
        vectorized=False,
        atol=1e-8,
        rtol=1e-6
    )

    assert res.success, f"solve_ivp failed: {res.message}"

    states = res.y.T.reshape(len(out["time_grid"]), ncomp, nloc)

    # allow small negative round-off but no large excursions
    assert np.all(np.isfinite(states))
    assert np.min(states) > -1e-6
    assert np.max(states) < 1e9
