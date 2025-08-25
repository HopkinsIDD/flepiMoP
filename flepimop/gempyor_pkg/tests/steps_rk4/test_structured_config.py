import numpy as np
import pytest
from functools import partial
from scipy.integrate import solve_ivp

from gempyor.steps_rk4 import rk4_integration
from gempyor.vectorization_experiments import RHSfactory

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


def param_lookup(param, param_names):
    if param in param_names:
        return np.where(param_names == param)[0][0]
    return None


def compartment_lookup(compartment: str, compartments_df):
    mask = compartments_df["infection_stage"].astype(str).str.startswith(compartment)
    return compartments_df[mask].index


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------


@pytest.fixture(scope="module")
def modelinfo_from_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("model_input")

    # Resolve tutorials dir relative to this test file
    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

    # --- Copy Structured_Example inputs ---
    src_structured = tutorial_dir / "model_input" / "Structured_Example"
    dst_structured = tmp_path / "model_input" / "Structured_Example"
    shutil.copytree(src_structured, dst_structured, dirs_exist_ok=True)

    # --- Copy initial_condition directory (plugin + CSVs) ---
    src_ic = tutorial_dir / "model_input" / "initial_condition"
    dst_ic = tmp_path / "model_input" / "initial_condition"
    shutil.copytree(src_ic, dst_ic, dirs_exist_ok=True)

    # --- Copy config file ---
    config_file = "Structured_Example.yml"
    config_path = tmp_path / config_file
    shutil.copyfile(tutorial_dir / config_file, config_path)

    # --- Workaround: patch YAML to absolute paths ---
    cfg_text = config_path.read_text()
    cfg_text = cfg_text.replace("model_input/", str(tmp_path / "model_input") + "/")
    config_path.write_text(cfg_text)

    # --- Load config ---
    config = confuse.Configuration("TestModel", __name__)
    config.set_file(str(config_path))

    # --- Build ModelInfo ---
    model = ModelInfo(
        config=config,
        config_filepath=str(config_path),
        path_prefix=str(tmp_path),  # unused due to bug, but keep for API consistency
        setup_name="Structured_Example",
        seir_modifiers_scenario="none",
    )

    return model, config


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

    # Safe fraction who move (avoid divide-by-zero)
    proportion_who_move = np.zeros(model.nsubpops, dtype=np.float64)
    for i in range(model.nsubpops):
        pop_i = float(population[i])
        total_flux = float(
            mobility_data[mobility_data_indices[i] : mobility_data_indices[i + 1]].sum()
        )
        if pop_i > 0.0:
            proportion_who_move[i] = min(total_flux / pop_i, 1.0)
        else:
            proportion_who_move[i] = 0.0

    offset = model.ti.toordinal()
    t0 = 0.0
    t1 = float(model.tf.toordinal() - offset)
    dt = 0.1
    time_grid = np.linspace(t0, t1, int((t1 - t0) / dt) + 1).astype(np.float64)

    # Build the precomputed dict for RHSfactory (seeding intentionally omitted: OFF)
    precomputed = {
        "ncompartments": initial_array.shape[0],
        "nspatial_nodes": initial_array.shape[1],
        "transitions": transitions,
        "proportion_info": proportion_info,
        "transition_sum_compartments": transition_sum_compartments,
        "percent_day_away": 0.5,  # same as below
        "proportion_who_move": proportion_who_move,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_data_indices,
        "mobility_row_indices": mobility_row_indices,
        "population": population,
        # NOTE: no 'seeding_data' / 'seeding_amounts' keys => seeding OFF
    }

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
        "precomputed": precomputed,
        "param_names": np.array(list(param_defs.keys())),
        "compartments": model.compartments.compartments,
    }


# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
@pytest.mark.benchmark(group="solver_performance", min_rounds=20)
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
            seeding_data={"day_start_idx": np.zeros(ndays_daily, dtype=int)},  # OFF
            seeding_amounts=np.zeros(0),
            mobility_data=out["mobility_data"],
            mobility_row_indices=out["mobility_row_indices"],
            mobility_data_indices=out["mobility_data_indices"],
            population=out["population"],
            method="rk4",
            silent=True,
        )

    benchmark(run_legacy)


@pytest.mark.benchmark(group="solver_performance", min_rounds=20)
@pytest.mark.parametrize("eval_step", [1.0, 0.1])
def test_vectorized_solver__param_eval_grid(benchmark, model_and_inputs, eval_step):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    # Build factory (param_time_mode='step' to match legacy stepping)
    factory = RHSfactory(
        precomputed=out["precomputed"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    t0 = 0.0
    t1 = float(out["model"].n_days - 1)
    if eval_step == 1.0:
        t_eval = np.arange(t0, t1 + 1e-12, 1.0, dtype=np.float64)
    else:
        t_eval = np.arange(t0, t1 + 1e-12, eval_step, dtype=np.float64)

    def run__with_eval_grid():
        res = factory.solve(
            y0=out["initial_array"].ravel(),
            parameters=out["params"],
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-2,
            atol=1e-4,
        )
        return res.y.T.reshape(len(t_eval), ncomp, nloc)

    states = benchmark(run__with_eval_grid)
    assert states.shape == (len(t_eval), ncomp, nloc)


def test_overlay_rhs(model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    # Daily grid aligned with legacy indexing
    ndays = out["model"].n_days
    t_daily = np.arange(0.0, float(ndays), 1.0, dtype=np.float64)

    # Legacy RK4 (seeding OFF)
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
        seeding_data={"day_start_idx": np.zeros(ndays, dtype=int)},  # OFF
        seeding_amounts=np.zeros(0),
        mobility_data=out["mobility_data"],
        mobility_row_indices=out["mobility_row_indices"],
        mobility_data_indices=out["mobility_data_indices"],
        population=out["population"],
        method="rk4",
        silent=True,
    )

    # Vectorized RHS +  via factory (fast path: precomputed params)
    factory = RHSfactory(
        precomputed=out["precomputed"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )
    sol_vec = factory.solve(
        y0=out["initial_array"].ravel(),
        parameters=out["params"],
        t_span=(t_daily[0], t_daily[-1]),
        t_eval=t_daily,
        method="RK45",
        rtol=1e-2,
        atol=1e-4,
        max_step=1.0,
    )
    assert sol_vec.success, f"Vectorized  failed: {sol_vec.message}"
    states_vec = sol_vec.y.T.reshape(len(t_daily), ncomp, nloc)

    # Overlay of I(t) total across nodes
    I_idx = compartment_lookup("I", out["compartments"])
    y_legacy = states_legacy[:, I_idx, :].sum(axis=(1, 2))  # (T,)
    y_vec = states_vec[:, I_idx, :].sum(axis=(1, 2))  # (T,)

    test_dir = Path(__file__).parent
    outpath = test_dir / "solver_overlay_I_total_daily_manual.png"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(t_daily, y_legacy, label="Legacy (daily)", linewidth=1.5)
    ax.plot(t_daily, y_vec, label="Vectorized (daily)", linestyle="--", linewidth=1.5)
    ax.set_title("Overlay: I(t) total across nodes — Daily Grid")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Individuals")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    assert outpath.exists(), f"Expected plot at {outpath} not found"
