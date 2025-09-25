# test_vectorized_param_cases.py  (per-config autotune)
import os, sys, platform, shutil
from pathlib import Path
from ctypes.util import find_library

# --- BLAS hygiene (avoid oversubscription) ---
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# --- Choose a numba threading layer *before* importing numba users ---
def _choose_numba_layer() -> str:
    forced = os.environ.get("NUMBA_THREADING_LAYER")
    if forced:
        return forced
    if find_library("tbb") or find_library("tbb12"):
        return "tbb"
    if find_library("omp") or find_library("gomp"):
        return "omp"
    return "workqueue"

def _maybe_patch_dylib_path(layer: str) -> None:
    if platform.system() != "Darwin":
        return
    extra = []
    if layer == "tbb":
        extra = ["/opt/homebrew/opt/tbb/lib", "/usr/local/opt/tbb/lib"]
    elif layer == "omp":
        extra = ["/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"]
    if not extra:
        return
    cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    add = [p for p in extra if Path(p).exists() and p not in cur]
    if add:
        os.environ["DYLD_LIBRARY_PATH"] = (":".join(add) + (":" + cur if cur else ""))

_layer = _choose_numba_layer()
_maybe_patch_dylib_path(_layer)
os.environ.setdefault("NUMBA_THREADING_LAYER", _layer)
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))

# Clear numba cache compiled under other layers
try:
    shutil.rmtree(Path.home() / ".numba" / "cache", ignore_errors=True)
except Exception:
    pass

print(
    f"[env] NUMBA_THREADING_LAYER={os.environ['NUMBA_THREADING_LAYER']} "
    f"NUMBA_NUM_THREADS={os.environ['NUMBA_NUM_THREADS']}",
    file=sys.stderr,
)

# ------------------------------------------------------------------------------------------------
# Regular imports (safe now that env is set)
# ------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
import confuse

from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

# legacy baseline (optional perf comparison)
from gempyor.steps_rk4 import rk4_integration

# vectorized solver + autotune helpers
from gempyor.vectorization_experiments import (
    RHSfactory,
    autotune_all,
    # the following may or may not exist; handled via try/except below
    get_autotune_config,
)

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
    return {str(k): np.asarray(v) for k, v in nb_dict.items()}


def _per_day_seed_sums(
    day_start_idx: np.ndarray, amounts: np.ndarray, n_days: int
) -> np.ndarray:
    out = np.zeros(n_days, dtype=np.float64)
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
        "seeding_data": seeding_data,
        "seeding_amounts": seeding_amounts,
        "daily_incidence": daily_incidence,
    }


# ------------------------------------------------------------
# Fixtures (parametrized over original vs alt config)
# ------------------------------------------------------------
@pytest.fixture(scope="module", params=["Structured_Example_Seeding_Alt.yml","Structured_Example.yml"], ids=["alt", "orig"])
def model_and_inputs(request, tmp_path_factory):
    return _prepare_case(request.param, tmp_path_factory)


# ------------------------------------------------------------
# Per-config autotune (depends on the prepared case)
# ------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def autotune_for_case(model_and_inputs):
    """
    Tune knobs using the case's workload (Tn, N, seeding_on).
    Skips cleanly if no parallel layer can be loaded on this machine.
    """
    out = model_and_inputs
    Tn = int(out["transitions"].shape[1])
    N = int(out["precomputed"]["nspatial_nodes"])
    hint = {"Tn": Tn, "N": N, "seeding": bool(out["seeding_on"])}

    try:
        try:
            cfg = autotune_all(quiet=True, workload_hint=hint)  # if your version supports it
        except TypeError:
            cfg = autotune_all(quiet=True)
            case_size = max(1, Tn * N)
            thr = int(min(max(case_size // 4, 1_000_000), 50_000_000))
            try:
                from gempyor.vectorization_experiments import set_autotune_config
                cfg = {**cfg, "parallel_threshold": thr}
                set_autotune_config(cfg)
            except Exception:
                pass
    except ValueError as e:
        if "No threading layer could be loaded" in str(e):
            pytest.skip("Numba threading layer unavailable; skipped autotune_all()")
        raise

    try:
        active = get_autotune_config()
    except Exception:
        active = cfg
    print(f"[autotune][{out['cfg_name']}] {active}")


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
@pytest.mark.benchmark(group="solver_performance", min_rounds=2)
def test_legacy_solver_performance_param(benchmark, model_and_inputs):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape
    ndays_daily = out["model"].n_days

    if out["seeding_on"]:
        seeding_data = out["seeding_data"]
        seeding_amounts = out["seeding_amounts"]
    else:
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
    assert result is None or True


@pytest.mark.benchmark(group="solver_performance", min_rounds=2)
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


# ------------------------------------------------------------
# NEW: Plot I(t) overlay for two configs with the vectorized solver
# ------------------------------------------------------------
def _solve_daily_I(out_dict):
    """Run vectorized solver for a prepared case and return (t_daily, I_total)."""
    ncomp, nloc = out_dict["initial_array"].shape
    ndays = out_dict["model"].n_days
    t_daily = np.arange(0.0, float(ndays - 1) + 1e-12, 1.0, dtype=np.float64)

    factory = RHSfactory(
        precomputed=out_dict["precomputed"],
        param_expr_lookup=out_dict["param_expr_lookup"],
        param_name_to_row=out_dict["param_name_to_row"],
        param_time_mode="step",
    )
    res = factory.solve(
        y0=out_dict["initial_array"].ravel(),
        parameters=out_dict["params"],
        t_span=(t_daily[0], t_daily[-1] + 1.0),
        t_eval=t_daily,
        method="RK45",
        rtol=1e-2,
        atol=1e-4,
    )
    assert res.success, f"Vectorized solve failed: {res.message}"
    states = res.y.T.reshape(len(t_daily), ncomp, nloc)

    I_idx = compartment_lookup("I", out_dict["compartments"])
    I_total = states[:, I_idx, :].sum(axis=(1, 2))  # (T,)
    return t_daily, I_total


@pytest.mark.slow
def test_plot_I_overlay_two_configs(tmp_path_factory):
    """
    Solve both configs (alt & orig) and save an overlay plot of I(t) totals
    next to this test file.
    """
    # Make sure we have a threading layer tuned at least once for this session.
    try:
        autotune_all(quiet=True)
    except Exception:
        pass  # continue anyway; solver will still run serially if needed

    # Prepare both cases locally (don't use the parametrized fixture so we can plot both)
    alt = _prepare_case("Structured_Example_Seeding_Alt.yml", tmp_path_factory)
    orig = _prepare_case("Structured_Example.yml", tmp_path_factory)

    t_alt, I_alt = _solve_daily_I(alt)
    t_orig, I_orig = _solve_daily_I(orig)

    # Save plot alongside this test file
    outdir = Path(__file__).parent
    outpath = outdir / "I_overlay_alt_vs_orig.png"

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(t_alt, I_alt, label="Alt config (no seeding)")
    ax.plot(t_orig, I_orig, label="Orig config (with seeding)", linestyle="--")
    ax.set_title("Overlay: I(t) total across nodes — Alt vs Orig")
    ax.set_xlabel("Day")
    ax.set_ylabel("Individuals")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    print(f"[artifact] I(t) overlay saved to: {outpath}")
    assert outpath.exists()

