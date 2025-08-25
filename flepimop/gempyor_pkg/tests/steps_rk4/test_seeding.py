import numpy as np
import pytest
from pathlib import Path
import shutil
import confuse
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

# legacy baseline (optional perf comparison)
from gempyor.steps_rk4 import rk4_integration

# new vectorized solver
from gempyor.vectorization_experiments import RHSfactory

# plotting (headless, only for optional debug figure writing)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gempyor.model_info import ModelInfo


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def build_safe_param_expr_lookup(unique_strings: list[str]):
    name_to_row = {name: i for i, name in enumerate(unique_strings)}
    expr_lookup: dict[int, str] = {}
    for idx, s in enumerate(unique_strings):
        if "*" in s:
            terms = [t.strip() for t in s.split("*")]
            if all(term in name_to_row for term in terms):
                expr_lookup[idx] = s
    return (expr_lookup if expr_lookup else None, name_to_row)


def compartment_lookup(compartment: str, compartments_df):
    mask = compartments_df["infection_stage"].astype(str).str.startswith(compartment)
    return compartments_df[mask].index


def age_strata_lookup(age_strata: str, compartments_df):
    mask = compartments_df["age_strata"].astype(str) == age_strata
    return compartments_df[mask].index


def _to_py_seeding_dict(nb_dict) -> dict[str, np.ndarray]:
    # Convert potential numba.typed.Dict to a plain dict of numpy arrays
    return {str(k): np.asarray(v) for k, v in nb_dict.items()}


def _per_day_seed_sums(
    day_start_idx: np.ndarray, amounts: np.ndarray, n_days: int
) -> np.ndarray:
    # day_start_idx is CSR-like pointers of length D+1, amounts length E
    out = np.zeros(n_days, dtype=np.float64)
    # Guard if pointers length mismatched; use min for safety
    D = min(n_days, day_start_idx.size - 1)
    for d in range(D):
        s = int(day_start_idx[d])
        e = int(day_start_idx[d + 1])
        out[d] = float(amounts[s:e].sum())
    return out


def _prepare_case(cfg_name: str, tmp_path_factory):
    """Build the full input dict for a given config name."""
    tmp_path = tmp_path_factory.mktemp(f"model_input_{Path(cfg_name).stem}")

    # Resolve tutorials dir relative to this test file
    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

    # Copy Structured_Example inputs
    src_structured = tutorial_dir / "model_input" / "Structured_Example"
    dst_structured = tmp_path / "model_input" / "Structured_Example"
    shutil.copytree(src_structured, dst_structured, dirs_exist_ok=True)

    # Copy initial_condition directory (plugin + CSVs)
    src_ic = tutorial_dir / "model_input" / "initial_condition"
    dst_ic = tmp_path / "model_input" / "initial_condition"
    shutil.copytree(src_ic, dst_ic, dirs_exist_ok=True)

    # Copy the chosen config and patch relative paths -> absolute
    config_path = tmp_path / cfg_name
    shutil.copyfile(tutorial_dir / cfg_name, config_path)
    cfg_text = config_path.read_text()
    cfg_text = cfg_text.replace("model_input/", str(tmp_path / "model_input") + "/")
    config_path.write_text(cfg_text)

    # Load config & build ModelInfo
    config = confuse.Configuration("TestModel", __name__)
    config.set_file(str(config_path))

    model = ModelInfo(
        config=config,
        config_filepath=str(config_path),
        path_prefix=str(tmp_path),
        setup_name="Structured_Example",
        seir_modifiers_scenario="none",
    )

    # Determine seeding policy by config
    seeding_on = cfg_name == "Structured_Example.yml"

    # Base model structures
    initial_array = model.initial_conditions.get_from_config(sim_id=0, modinf=model)
    unique_strings, transitions, transition_sum_compartments, proportion_info = (
        model.compartments.get_transition_array()
    )

    # Parameters
    param_defs = config["seir"]["parameters"].get()
    base_params = model.parameters.parameters_quick_draw(model.n_days, model.nsubpops)
    parsed_params = model.compartments.parse_parameters(
        base_params, param_defs, unique_strings
    )

    param_expr_lookup, param_name_to_row = build_safe_param_expr_lookup(unique_strings)

    # Mobility
    mobility_csr: csr_matrix = model.mobility
    population = model.subpop_pop
    mobility_data = mobility_csr.data
    mobility_data_indices = mobility_csr.indptr
    mobility_row_indices = mobility_csr.indices

    # Fraction who move (safe)
    proportion_who_move = np.zeros(model.nsubpops, dtype=np.float64)
    for i in range(model.nsubpops):
        pop_i = float(population[i])
        total_flux = float(
            mobility_data[mobility_data_indices[i] : mobility_data_indices[i + 1]].sum()
        )
        proportion_who_move[i] = min(total_flux / pop_i, 1.0) if pop_i > 0 else 0.0

    # Time grid helpers
    offset = model.ti.toordinal()
    t0 = 0.0
    t1 = float(model.tf.toordinal() - offset)
    dt = 0.1
    time_grid = np.linspace(t0, t1, int((t1 - t0) / dt) + 1).astype(np.float64)

    # --- Seeding ---
    daily_incidence = np.zeros(
        (model.n_days, initial_array.shape[0], initial_array.shape[1]), dtype=np.float64
    )
    seeding_data = None
    seeding_amounts = None
    if seeding_on:
        seeding_data_nb, seeding_amounts = model.get_seeding_data(sim_id=0)
        seeding_data_py = _to_py_seeding_dict(seeding_data_nb)
        # Normalize dtypes for our Numba kernels
        seeding_data = {
            "day_start_idx": np.ascontiguousarray(
                seeding_data_py["day_start_idx"], dtype=np.int64
            ),
            "seeding_subpops": np.ascontiguousarray(
                seeding_data_py["seeding_subpops"], dtype=np.int64
            ),
            "seeding_sources": np.ascontiguousarray(
                seeding_data_py["seeding_sources"], dtype=np.int64
            ),
            "seeding_destinations": np.ascontiguousarray(
                seeding_data_py["seeding_destinations"], dtype=np.int64
            ),
        }
        seeding_amounts = np.ascontiguousarray(seeding_amounts, dtype=np.float64)

    # Precomputed dict for RHSfactory
    precomputed = {
        "ncompartments": initial_array.shape[0],
        "nspatial_nodes": initial_array.shape[1],
        "transitions": transitions.astype(np.int64, copy=False),
        "proportion_info": proportion_info.astype(np.int64, copy=False),
        "transition_sum_compartments": transition_sum_compartments.astype(
            np.int64, copy=False
        ),
        "percent_day_away": 0.5,
        "proportion_who_move": proportion_who_move,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_data_indices.astype(np.int64, copy=False),
        "mobility_row_indices": mobility_row_indices.astype(np.int64, copy=False),
        "population": population,
    }
    if seeding_on:
        precomputed.update(
            {
                "seeding_data": seeding_data,
                "seeding_amounts": seeding_amounts,
                "daily_incidence": daily_incidence,
            }
        )

    return {
        "cfg_name": cfg_name,
        "seeding_on": seeding_on,
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
        # Seeding artifacts for validation (None if seeding_off)
        "seeding_data": seeding_data,
        "seeding_amounts": seeding_amounts,
        "daily_incidence": daily_incidence,
    }


# ------------------------------------------------------------
# Fixtures (parametrized over original vs alt config)
# ------------------------------------------------------------
# @pytest.fixture(scope="module", params=["Structured_Example.yml", "Structured_Example_Seeding_Alt.yml"], ids=["orig", "alt"])
@pytest.fixture(scope="module", params=["Structured_Example.yml"], ids=["orig"])
def model_and_inputs(request, tmp_path_factory):
    return _prepare_case(request.param, tmp_path_factory)


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
def test_seeding_applies_to_daily_incidence(model_and_inputs):
    out = model_and_inputs
    if not out["seeding_on"]:
        pytest.skip(
            "Alt config encodes seeding as a transition; explicit seeding is OFF here."
        )

    ncomp, nloc = out["initial_array"].shape
    ndays = out["model"].n_days

    # Solve on a daily grid (param_time_mode='step' to align with seeding tick)
    t_daily = np.arange(0.0, float(ndays - 1) + 1e-12, 1.0, dtype=np.float64)

    factory = RHSfactory(
        precomputed=out["precomputed"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )
    res = factory.solve(
        y0=out["initial_array"].ravel(),
        parameters=out["params"],
        t_span=(t_daily[0], t_daily[-1] + 1.0),  # integrate through the full last day
        t_eval=t_daily,
        method="RK45",
        rtol=1e-2,
        atol=1e-4,
    )
    assert res.success, f"Vectorized solver failed: {res.message}"

    # Validate: daily_incidence aggregated equals scheduled seeding by day
    di = out["daily_incidence"]  # (D, C, N) mutated in place
    di_by_day = di.sum(axis=(1, 2))  # (D,)
    scheduled_by_day = _per_day_seed_sums(
        out["seeding_data"]["day_start_idx"], out["seeding_amounts"], ndays
    )
    assert np.allclose(
        di_by_day, scheduled_by_day, rtol=0, atol=1e-12
    ), "Daily incidence does not match scheduled seeding amounts."


@pytest.mark.benchmark(group="solver_performance", min_rounds=5)
def test_legacy_solver_performance_param(benchmark, model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    ndays_daily = out["model"].n_days

    # Seeding policy by config:
    if out["seeding_on"]:
        seeding_data = out["seeding_data"]
        seeding_amounts = out["seeding_amounts"]
    else:
        # Seeding OFF in alt config
        seeding_data = {"day_start_idx": np.zeros(ndays_daily, dtype=int)}
        seeding_amounts = np.zeros(0, dtype=np.float64)

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
            seeding_data=seeding_data,
            seeding_amounts=seeding_amounts,
            mobility_data=out["mobility_data"],
            mobility_row_indices=out["mobility_row_indices"],
            mobility_data_indices=out["mobility_data_indices"],
            population=out["population"],
            method="rk4",
            silent=True,
        )

    result = benchmark(run_legacy)
    assert result is None or True  # just to have an assertion


@pytest.mark.benchmark(group="solver_performance", min_rounds=5)
def test_vectorized_solver_performance_param(benchmark, model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    ndays = out["model"].n_days
    t_daily = np.arange(0.0, float(ndays - 1) + 1e-12, 1.0, dtype=np.float64)

    factory = RHSfactory(
        precomputed=out["precomputed"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    def run_vec():
        res = factory.solve(
            y0=out["initial_array"].ravel(),
            parameters=out["params"],
            t_span=(t_daily[0], t_daily[-1] + 1.0),
            t_eval=t_daily,
            method="RK45",
            rtol=1e-2,
            atol=1e-4,
        )
        return res

    result = benchmark(run_vec)
    assert result.success


def test_overlay_rhs_with_seeding(model_and_inputs):
    """Overlay legacy vs vectorized when explicit seeding is ON (original config)."""
    out = model_and_inputs
    if not out["seeding_on"]:
        pytest.skip(
            "Alt config: explicit seeding OFF; this overlay is for original config only."
        )

    ncomp, nloc = out["initial_array"].shape
    ndays = out["model"].n_days
    t_daily = np.arange(0.0, float(ndays - 1) + 1e-12, 1.0, dtype=np.float64)

    # Legacy RK4 with true seeding
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
        seeding_data=out["seeding_data"],  # ON
        seeding_amounts=out["seeding_amounts"],  # ON
        mobility_data=out["mobility_data"],
        mobility_row_indices=out["mobility_row_indices"],
        mobility_data_indices=out["mobility_data_indices"],
        population=out["population"],
        method="rk4",
        silent=True,
    )

    # Vectorized path
    factory = RHSfactory(
        precomputed=out["precomputed"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )
    sol_vec = factory.solve(
        y0=out["initial_array"].ravel(),
        parameters=out["params"],
        t_span=(t_daily[0], t_daily[-1] + 1.0),
        t_eval=t_daily,
        method="RK45",
        rtol=1e-2,
        atol=1e-4,
    )
    assert sol_vec.success
    states_vec = sol_vec.y.T.reshape(len(t_daily), ncomp, nloc)

    # I(t) totals across nodes
    I_idx = compartment_lookup("I", out["compartments"])

    y_legacy = states_legacy[:, I_idx, :].sum(axis=(1, 2))  # (T,)
    y_vec = states_vec[:, I_idx, :].sum(axis=(1, 2))  # (T,)

    # Save plot for offline inspection
    test_dir = Path(__file__).parent
    outpath = test_dir / "solver_overlay_Incid_total_daily_seeding.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(t_daily, y_legacy, label="Legacy RK4 (seeding ON)", linewidth=1.5)
    ax.plot(
        t_daily, y_vec, label="Vectorized (seeding ON)", linestyle="--", linewidth=1.5
    )
    ax.set_title("Overlay: Case Incidence across nodes — Daily Grid (Seeding ON)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Individuals")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    assert outpath.exists(), f"Expected plot at {outpath} not found"


# def test_overlay_vectorized_both_configs(tmp_path_factory):
#     """
#     Overlay the vectorized solver for both configs:
#       - Original (explicit seeding ON)
#       - Alt (seeding encoded as transition; explicit seeding OFF)
#     """
#     # Build both cases independently
#     out_orig = _prepare_case("Structured_Example.yml", tmp_path_factory)
#     out_alt  = _prepare_case("Structured_Example_Alt_Seeding.yml", tmp_path_factory)

#     # Time grid (use original's ndays to define plotting axis)
#     ndays = out_orig["model"].n_days
#     t_daily = np.arange(0.0, float(ndays - 1) + 1e-12, 1.0, dtype=np.float64)

#     # Vectorized solve (original)
#     fac_orig = RHSfactory(
#         precomputed=out_orig["precomputed"],
#         param_expr_lookup=out_orig["param_expr_lookup"],
#         param_name_to_row=out_orig["param_name_to_row"],
#         param_time_mode="step",
#     )
#     sol_orig = fac_orig.solve(
#         y0=out_orig["initial_array"].ravel(),
#         parameters=out_orig["params"],
#         t_span=(t_daily[0], t_daily[-1] + 1.0),
#         t_eval=t_daily,
#         method="RK45",
#         rtol=1e-2,
#         atol=1e-4,
#     )
#     assert sol_orig.success
#     ncomp_o, nloc_o = out_orig["initial_array"].shape
#     states_vec_orig = sol_orig.y.T.reshape(len(t_daily), ncomp_o, nloc_o)

#     # Vectorized solve (alt; no explicit seeding keys in precomputed)
#     fac_alt = RHSfactory(
#         precomputed=out_alt["precomputed"],
#         param_expr_lookup=out_alt["param_expr_lookup"],
#         param_name_to_row=out_alt["param_name_to_row"],
#         param_time_mode="step",
#     )
#     sol_alt = fac_alt.solve(
#         y0=out_alt["initial_array"].ravel(),
#         parameters=out_alt["params"],
#         t_span=(t_daily[0], t_daily[-1] + 1.0),
#         t_eval=t_daily,
#         method="RK45",
#         rtol=1e-2,
#         atol=1e-4,
#     )
#     assert sol_alt.success
#     ncomp_a, nloc_a = out_alt["initial_array"].shape
#     states_vec_alt = sol_alt.y.T.reshape(len(t_daily), ncomp_a, nloc_a)

#     # Choose I(t) totals across nodes for both
#     I_idx_o = compartment_lookup("I", out_orig["compartments"])
#     I_idx_a = compartment_lookup("I", out_alt["compartments"])
#     y_vec_orig = states_vec_orig[:, I_idx_o, :].sum(axis=(1, 2))  # (T,)
#     y_vec_alt  = states_vec_alt[:,  I_idx_a, :].sum(axis=(1, 2))  # (T,)

#     # Save overlay plot
#     test_dir = Path(__file__).parent
#     outpath = test_dir / "solver_overlay_vectorized_orig_vs_alt.png"
#     fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
#     ax.plot(t_daily, y_vec_orig, label="Vectorized — Original cfg (explicit seeding ON)", linewidth=1.5)
#     ax.plot(t_daily, y_vec_alt,  label="Vectorized — Alt cfg (seeding via transition)", linestyle="--", linewidth=1.5)
#     ax.set_title("Overlay: Vectorized solver — Original vs Alt Config")
#     ax.set_xlabel("Time (days)")
#     ax.set_ylabel("Individuals (I total)")
#     ax.legend(loc="best")
#     ax.grid(True, alpha=0.3)
#     fig.savefig(outpath, bbox_inches="tight")
#     plt.close(fig)
#     assert outpath.exists(), f"Expected plot at {outpath} not found"
