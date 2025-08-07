import numpy as np
import pytest
from functools import partial
from scipy.integrate import solve_ivp

from gempyor.steps_rk4 import rk4_integration
from gempyor.vectorization_experiments import build_rhs_for_solve_ivp

import shutil
from pathlib import Path
import confuse
from scipy.sparse import csr_matrix

# --- plotting (headless) ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gempyor.model_info import ModelInfo


# ------------------------------------------------------------
# Helper: build a SAFE expression lookup from unique_strings
# ------------------------------------------------------------
def build_safe_param_expr_lookup(
    unique_strings: list[str],
) -> tuple[dict[int, str] | None, dict[str, int]]:
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
    )

    param_expr_lookup, param_name_to_row = build_safe_param_expr_lookup(unique_strings)

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

    offset = model.ti.toordinal()
    t0 = 0.0
    t1 = float(model.tf.toordinal() - offset)
    dt = 0.1
    time_grid = np.linspace(t0, t1, int((t1 - t0) / dt) + 1).astype(np.float64)

    return {
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
        "param_expr_lookup": param_expr_lookup,
        "param_name_to_row": param_name_to_row,
    }


# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
@pytest.mark.benchmark(group="solver_performance")
def test_legacy_solver_performance(benchmark, model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    ndays_daily = out["model"].n_days

    def run_legacy():
        rk4_integration(
            ncompartments=ncomp,
            nspatial_nodes=nloc,
            ndays=ndays_daily,
            parameters=out["params"],
            dt=out["dt"],
            transitions=out["transitions"],
            proportion_info=out["proportion_info"],
            transition_sum_compartments=out["transition_sum_compartments"],
            initial_conditions=out["initial_array"],
            seeding_data={"day_start_idx": np.zeros(ndays_daily, dtype=int)},
            seeding_amounts=np.zeros(0),
            mobility_data=out["mobility_data"],
            mobility_row_indices=out["mobility_row_indices"],
            mobility_data_indices=out["mobility_data_indices"],
            population=out["population"],
            method="rk4",
            silent=True,
        )

    benchmark(run_legacy)


@pytest.mark.benchmark(group="solver_performance")
@pytest.mark.parametrize("eval_step", [1.0, 0.1])
def test_vectorized_solver_bdf_param_eval_grid(benchmark, model_and_inputs, eval_step):
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
    f = partial(rhs, parameters=out["params"])

    t0 = 0.0
    t1 = float(out["model"].n_days - 1)
    if eval_step == 1.0:
        t_eval = np.arange(t0, t1 + 1e-12, 1.0, dtype=np.float64)
    else:
        t_eval = np.arange(t0, t1 + 1e-12, eval_step, dtype=np.float64)

    def run_bdf_with_eval_grid():
        res = solve_ivp(
            fun=f,
            t_span=(t_eval[0], t_eval[-1]),
            y0=out["initial_array"].ravel(),
            method="BDF",
            t_eval=t_eval,
            vectorized=False,
            rtol=1e-6,
            atol=1e-8,
        )
        return res.y.T.reshape(len(t_eval), ncomp, nloc)

    states = benchmark(run_bdf_with_eval_grid)
    assert states.shape == (len(t_eval), ncomp, nloc)


def test_overlay_rhs(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    # Daily grid aligned with legacy indexing
    ndays = out["model"].n_days
    t_daily = np.arange(0.0, float(ndays), 1.0, dtype=np.float64)

    # Legacy RK4
    states_legacy, _ = rk4_integration(
        ncompartments=ncomp,
        nspatial_nodes=nloc,
        ndays=ndays,
        parameters=out["params"],
        dt=out["dt"],
        transitions=out["transitions"],
        proportion_info=out["proportion_info"],
        transition_sum_compartments=out["transition_sum_compartments"],
        initial_conditions=out["initial_array"],
        seeding_data={"day_start_idx": np.zeros(ndays, dtype=int)},
        seeding_amounts=np.zeros(0),
        mobility_data=out["mobility_data"],
        mobility_row_indices=out["mobility_row_indices"],
        mobility_data_indices=out["mobility_data_indices"],
        population=out["population"],
        method="rk4",
        silent=True,
    )

    # Vectorized RHS + BDF (existing path)
    rhs_vec = build_rhs_for_solve_ivp(
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
    f_vec = partial(rhs_vec, parameters=out["params"])
    sol_vec = solve_ivp(
        fun=f_vec,
        t_span=(t_daily[0], t_daily[-1]),
        y0=out["initial_array"].ravel(),
        method="BDF",
        t_eval=t_daily,
        vectorized=False,
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0,
    )
    assert sol_vec.success, f"Vectorized BDF failed: {sol_vec.message}"
    states_vec = sol_vec.y.T.reshape(len(t_daily), ncomp, nloc)

    # From here you can plot or assert…
    # (keep your plotting code as before)

    # Overlay of I(t) total across nodes
    I_idx = 2
    y_legacy = states_legacy[:, I_idx, :].sum(axis=1)
    y_vec = states_vec[:, I_idx, :].sum(axis=1)

    test_dir = Path(__file__).parent
    outpath = test_dir / "solver_overlay_I_total_daily_manual.png"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(t_daily, y_legacy, label="Legacy RK4 (daily)", linewidth=1.5)
    ax.plot(
        t_daily, y_vec, label="Vectorized + BDF (daily)", linestyle="--", linewidth=1.5
    )
    ax.set_title("Overlay: I(t) total across nodes — Daily Grid")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Individuals")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    assert outpath.exists(), f"Expected plot at {outpath} not found"
