# tests/pipeline/test_r0_propagates.py
from __future__ import annotations

# ---------------- env hygiene BEFORE importing gempyor -----------------
import os, sys, platform, shutil
from pathlib import Path
from ctypes.util import find_library

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
try:
    shutil.rmtree(Path.home() / ".numba" / "cache", ignore_errors=True)
except Exception:
    pass

print(
    f"[env] NUMBA_THREADING_LAYER={os.environ['NUMBA_THREADING_LAYER']} "
    f"NUMBA_NUM_THREADS={os.environ['NUMBA_NUM_THREADS']}",
    file=sys.stderr,
)

# ---------------- regular imports --------------------------------------
import datetime as dt
import numpy as np
import pytest
import confuse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gempyor.vectorized_modifiers import compile_seir_modifiers
from gempyor.vectorization_experiments import RHSfactory
from gempyor.hosp_weekly_pipeline import (
    build_pipeline_from_config,
    _compile_outcomes_from_config,
    _param_slice_step,
    _compute_amounts_for_step,
)

# ---------------- helpers ----------------------------------------------
def _materialize_structured_example(tmp_path_factory) -> Path:
    tmp_root = tmp_path_factory.mktemp("r0_propagates_case")

    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

    # Inputs
    src_struct = tutorial_dir / "model_input" / "Structured_Example"
    dst_struct = tmp_root / "model_input" / "Structured_Example"
    shutil.copytree(src_struct, dst_struct, dirs_exist_ok=True)

    src_ic = tutorial_dir / "model_input" / "initial_condition"
    dst_ic = tmp_root / "model_input" / "initial_condition"
    shutil.copytree(src_ic, dst_ic, dirs_exist_ok=True)

    # Config
    cfg_name = "Structured_Example.yml"
    cfg_path = tmp_root / cfg_name
    shutil.copyfile(tutorial_dir / cfg_name, cfg_path)

    # Patch relative -> absolute
    text = cfg_path.read_text()
    text = text.replace("model_input/", str(tmp_root / "model_input") + "/")
    cfg_path.write_text(text)

    return cfg_path


def _extract_leaf_value(spec) -> float:
    v = spec.get("value", None)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        v1 = v.get("value", v.get("val", v.get("mult", None)))
        if isinstance(v1, (int, float)):
            return float(v1)
        if isinstance(v1, dict):
            v2 = v1.get("value", v1.get("val", v1.get("mult", None)))
            if isinstance(v2, (int, float)):
                return float(v2)
    return 1.0


def _leaf_defaults_in_order(conf, leaf_order: tuple[str, ...]) -> np.ndarray:
    sm = conf["seir_modifiers"].get()
    mods = sm["modifiers"]
    vals = []
    for nm in leaf_order:
        vals.append(_extract_leaf_value(mods[nm]))
    return np.asarray(vals, dtype=np.float64)


def _leaves_targeting_r0_overlapping_window(conf, start: dt.date, end: dt.date) -> set[str]:
    """Leaf names where method==SinglePeriodModifier, parameter=='r0', and period overlaps [start,end]."""
    sm = conf["seir_modifiers"].get()
    mods = sm["modifiers"]
    out = set()
    for nm, spec in mods.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("method") != "SinglePeriodModifier":
            continue
        if str(spec.get("parameter", "")).lower() != "r0":
            continue
        d0 = dt.date.fromisoformat(str(spec["period_start_date"]))
        d1 = dt.date.fromisoformat(str(spec["period_end_date"]))
        if max(d0, start) <= min(d1, end):  # overlap
            out.add(nm)
    return out


def _sum_with_fallback(series_dict: dict[str, np.ndarray], top_key: str, prefix: str) -> np.ndarray:
    """
    Prefer top-level key if present and nonzero; otherwise sum all leaf series with the given prefix.
    Returns a 1D vector over time (summed over locations).
    """
    if top_key in series_dict and np.any(series_dict[top_key]):
        return series_dict[top_key].sum(axis=1)
    total = None
    for k, v in series_dict.items():
        if k.startswith(prefix):
            s = v.sum(axis=1)
            total = s if total is None else (total + s)
    if total is None:
        any_arr = next(iter(series_dict.values()))
        total = np.zeros(any_arr.shape[0], dtype=np.float64)
    return total


def _build_series_on_step_grid(pipe, params_base_like):
    """
    Build incidI/incidH on the pipeline step grid using compiled outcomes.
    Pattern #2: parse BASE -> UNIQUE once; direct-indexing RHS & amounts.
    """
    # Compile outcomes for this step width
    outcomes_cfg = pipe.outcomes_cfg
    resolve_map, prob_map, delay_steps_map, sum_map, order = _compile_outcomes_from_config(
        outcomes_cfg,
        pipe.model.compartments.compartments,
        pipe.transitions,
        pipe.NL,
        bin_width_days=pipe.dt,
    )

    # ---- BASE -> UNIQUE conversion (align P with transitions[2,*]) ----
    params_unique = pipe.model.compartments.parse_parameters(
        params_base_like,          # (P_base, T, L)
        pipe.param_defs,           # base param defs
        pipe.unique_strings,       # unique names (incl. expanded expressions)
    )  # -> (P_unique, T, L)

    # Direct-indexing RHS (no expression wiring)
    factory = RHSfactory(
        precomputed=pipe.precomputed,
        param_time_mode="step",
    )

    total_days = float(pipe.T - 1)
    n_steps = max(1, int(round(total_days / pipe.dt)))
    t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

    res = factory.solve(
        y0=pipe.initial_array.ravel(),
        parameters=params_unique,            # UNIQUE tensor
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-3, atol=1e-6,
    )
    assert res.success, f"Integration failed: {res.message}"
    C, L = pipe.NC, pipe.NL
    states = res.y.T.reshape(len(t_eval), C, L)

    # Leaf-incidence on step grid
    series = {name: np.zeros((n_steps, L), dtype=np.float64) for name in order}
    pc = pipe.precomputed
    for i in range(n_steps):
        t0 = t_eval[i]
        param_t_slice = _param_slice_step(params_unique, t0)  # (P_unique, L)
        amounts, _ = _compute_amounts_for_step(
            states_current=states[i],
            transitions=pipe.transitions,
            proportion_info=pipe.proportion_info,
            transition_sum_compartments=pipe.transition_sum_compartments,
            param_t_slice=param_t_slice,
            percent_day_away=pc["percent_day_away"],
            prop_who_move=pc["proportion_who_move"],
            mobility_data=pc["mobility_data"],
            mobility_indptr=pc["mobility_data_indices"],
            mobility_indices=pc["mobility_row_indices"],
            population=pc["population"],
            # DIRECT indexing path to match UNIQUE tensor
            param_expr_lookup=None,
            param_name_to_row=None,
        )
        for name, rows in resolve_map.items():
            if rows.size == 0:
                continue
            inc_vec = amounts[rows, :].sum(axis=0)
            p_vec = prob_map.get(name, np.ones(L))
            shift = int(delay_steps_map.get(name, 0))
            j = i + shift
            if j < n_steps:
                series[name][j, :] += inc_vec * p_vec

    # Sums / aliases
    unresolved = set(sum_map.keys())
    guard = 0
    while unresolved and guard < 10000:
        guard += 1
        progressed = False
        for name in list(unresolved):
            kids = sum_map[name]
            if not all(k in series for k in kids):
                continue
            combined = np.zeros_like(series[name])
            for k in kids:
                combined += series[k]
            p_vec = prob_map.get(name, np.ones(L))
            shift = int(delay_steps_map.get(name, 0))
            if shift:
                out = np.zeros_like(combined)
                if shift < combined.shape[0]:
                    out[shift:, :] = combined[:-shift, :] * p_vec
            else:
                out = combined * p_vec
            series[name] = out
            unresolved.remove(name)
            progressed = True
        if not progressed:
            break
    assert not unresolved, f"Unresolved sums remain: {unresolved}"

    return series, delay_steps_map, prob_map


# ----------------------------- fixture ---------------------------------

@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)
    try:
        return build_pipeline_from_config(cfg_path, dt_days=0.1)
    except ValueError as e:
        if "No threading layer could be loaded" in str(e):
            pytest.skip("Numba threading layer unavailable; skipping r0 propagation tests.")
        raise


# ------------------------------ tests ----------------------------------

@pytest.mark.slow
def test_r0_window_bump_increases_incidI_and_incidH(pipe):
    """
    Multiply all r0 SinglePeriodModifier leaves overlapping a mid-season window by 1.3,
    then verify incidI increases within the window and incidH increases after the
    compiled delay; also overall totals should increase.
    """
    conf = confuse.Configuration("R0Bump", __name__)
    conf.set_file(str(pipe.config_path))

    applier = compile_seir_modifiers(
        seir_modifiers_cfg=conf["seir_modifiers"].get(),
        start_date=pipe.start_date,
        n_days=pipe.T,
        n_loc=pipe.NL,
        param_names=pipe.param_names,
    )
    leaves = pipe.modifier_order()
    defaults = _leaf_defaults_in_order(conf, leaves)

    # ~6-week mid-season window
    win_start = pipe.start_date + dt.timedelta(days=120)
    win_end   = win_start + dt.timedelta(days=42)

    # bump r0 leaves that overlap window by x1.3
    r0_leaves = _leaves_targeting_r0_overlapping_window(conf, win_start, win_end)
    bumped = defaults.copy()
    for i, name in enumerate(leaves):
        if name in r0_leaves:
            bumped[i] = float(bumped[i]) * 1.3

    # Apply modifiers to BASE params (converted to UNIQUE inside builder)
    params_base = applier.apply_to_params(pipe.base_params, leaf_value_array=defaults, scenario="none")
    params_bump = applier.apply_to_params(pipe.base_params, leaf_value_array=bumped,   scenario="none")

    series_base, delay_map, _ = _build_series_on_step_grid(pipe, params_base)
    series_bump, _, _          = _build_series_on_step_grid(pipe, params_bump)

    # robust aggregation
    I_base = _sum_with_fallback(series_base, "incidI", "incidI_")
    I_bump = _sum_with_fallback(series_bump, "incidI", "incidI_")
    H_base = _sum_with_fallback(series_base, "incidH", "incidH_")
    H_bump = _sum_with_fallback(series_bump, "incidH", "incidH_")

    # step indices for window (steps index by start-of-step day)
    step_per_day = int(round(1.0 / pipe.dt))  # 10 for 0.1
    i0 = max(0, ((win_start - pipe.start_date).days) * step_per_day)
    i1 = min(I_base.shape[0] - 1, ((win_end - pipe.start_date).days) * step_per_day)

    # incidI should be larger in-window
    sum_I_base = float(I_base[i0:i1+1].sum())
    sum_I_bump = float(I_bump[i0:i1+1].sum())
    assert sum_I_bump > sum_I_base * 1.02, f"incidI didn't increase enough: {sum_I_bump} vs {sum_I_base}"

    # incidH should be larger after a common delay
    any_h = next((k for k in delay_map if k.startswith("incidH_") and isinstance(delay_map[k], int)), None)
    shift = int(delay_map.get(any_h, 0))
    j0 = min(H_base.shape[0]-1, i0 + shift)
    j1 = min(H_base.shape[0]-1, i1 + shift)
    sum_H_base = float(H_base[j0:j1+1].sum())
    sum_H_bump = float(H_bump[j0:j1+1].sum())
    assert sum_H_bump > sum_H_base * 1.02, f"incidH didn't increase enough: {sum_H_bump} vs {sum_H_base}"

    # overall totals should also increase
    assert H_bump.sum() > H_base.sum()
    assert I_bump.sum() > I_base.sum()


@pytest.mark.slow
def test_plot_incidence_panels_baseline_vs_bump(pipe):
    """
    Save panels for quick inspection: daily incidI and incidH (summed over locations)
    baseline vs r0-bumped scenario.
    """
    conf = confuse.Configuration("R0BumpPlot", __name__)
    conf.set_file(str(pipe.config_path))
    applier = compile_seir_modifiers(
        seir_modifiers_cfg=conf["seir_modifiers"].get(),
        start_date=pipe.start_date,
        n_days=pipe.T,
        n_loc=pipe.NL,
        param_names=pipe.param_names,
    )
    leaves = pipe.modifier_order()
    defaults = _leaf_defaults_in_order(conf, leaves)

    win_start = pipe.start_date + dt.timedelta(days=120)
    win_end   = win_start + dt.timedelta(days=42)
    r0_leaves = _leaves_targeting_r0_overlapping_window(conf, win_start, win_end)
    bumped = defaults.copy()
    for i, name in enumerate(leaves):
        if name in r0_leaves:
            bumped[i] = float(bumped[i]) * 1.3

    params_base = applier.apply_to_params(pipe.base_params, leaf_value_array=defaults, scenario="none")
    params_bump = applier.apply_to_params(pipe.base_params, leaf_value_array=bumped,   scenario="none")

    series_base, _, _ = _build_series_on_step_grid(pipe, params_base)
    series_bump, _, _ = _build_series_on_step_grid(pipe, params_bump)

    # robust incidence aggregation
    I_base = _sum_with_fallback(series_base, "incidI", "incidI_")
    I_bump = _sum_with_fallback(series_bump, "incidI", "incidI_")
    H_base = _sum_with_fallback(series_base, "incidH", "incidH_")
    H_bump = _sum_with_fallback(series_bump, "incidH", "incidH_")

    # derive timeline length robustly
    any_arr = next(iter(series_base.values()))
    t = np.arange(any_arr.shape[0]) * pipe.dt  # days since t0

    outdir = Path(__file__).parent
    outpng = outdir / "incidence_panels_baseline_vs_r0_bump.png"

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=120, sharex=True)
    ax = axes[0]
    ax.plot(t, I_base, label="incidI baseline", linewidth=1.5)
    ax.plot(t, I_bump, "--", label="incidI r0-bump", linewidth=1.5)
    ax.set_ylabel("incidI (sum over L)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(t, H_base, label="incidH baseline", linewidth=1.5)
    ax.plot(t, H_bump, "--", label="incidH r0-bump", linewidth=1.5)
    ax.set_xlabel("day")
    ax.set_ylabel("incidH (sum over L)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.suptitle("Incidence baseline vs r0-bumped scenario (summed over locations)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

    print(f"[artifact] saved: {outpng}")
    assert outpng.exists()
