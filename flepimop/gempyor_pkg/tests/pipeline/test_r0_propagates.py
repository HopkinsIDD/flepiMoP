# tests/modifiers/test_r0_propagates_to_incidence.py
from __future__ import annotations

import os, sys, platform, shutil
from pathlib import Path
from ctypes.util import find_library

# ----------------------------- env hygiene (BEFORE any gempyor imports) ----
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

# Clear numba cache compiled under other layers (best-effort)
try:
    shutil.rmtree(Path.home() / ".numba" / "cache", ignore_errors=True)
except Exception:
    pass

print(
    f"[env] NUMBA_THREADING_LAYER={os.environ['NUMBA_THREADING_LAYER']} "
    f"NUMBA_NUM_THREADS={os.environ['NUMBA_NUM_THREADS']}",
    file=sys.stderr,
)

# ----------------------------- regular imports ------------------------------
import datetime as dt
import numpy as np
import pytest
import confuse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import AFTER env is fully set
from gempyor.hosp_weekly_pipeline import (
    build_pipeline_from_config,
    _compile_outcomes_from_config,
)
from gempyor.vectorized_modifiers import compile_seir_modifiers


# ----------------------------- helpers ---------------------------------

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
    """Return leaf names where method==SinglePeriodModifier, parameter=='r0', and the period overlaps [start,end]."""
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


def _build_series_on_step_grid(pipe, params):
    """
    Build only incidI/incidH on the pipeline step grid using the compiled outcomes.
    Returns dict(name)->(n_steps,L), and the delay/prob maps.
    """
    # Compile outcome maps on the same bin width as the pipeline
    outcomes_cfg = pipe.outcomes_cfg
    resolve_map, prob_map, delay_steps_map, sum_map, order = _compile_outcomes_from_config(
        outcomes_cfg,
        pipe.model.compartments.compartments,
        pipe.transitions,
        pipe.NL,
        bin_width_days=pipe.dt,
    )

    # integrate
    total_days = float(pipe.T - 1)
    n_steps = max(1, int(round(total_days / pipe.dt)))
    t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)
    res = pipe.factory.solve(
        y0=pipe.initial_array.ravel(),
        parameters=params,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-3, atol=1e-6,
    )
    assert res.success, f"Integration failed: {res.message}"
    C, L = pipe.NC, pipe.NL
    states = res.y.T.reshape(len(t_eval), C, L)

    # step loop (leaf-incidence only), then sums/aliases
    series = {name: np.zeros((n_steps, L), dtype=np.float64) for name in order}
    pc = pipe.precomputed
    from gempyor.hosp_weekly_pipeline import _param_slice_step, _compute_amounts_for_step  # reuse same helpers

    for i in range(n_steps):
        t0 = t_eval[i]
        param_t_slice = _param_slice_step(params, t0)  # (P,L)
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

    # sums/aliases
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
    Multiply all r0 SinglePeriodModifier leaves that overlap a chosen window by 1.3,
    then verify incidI increases within the window and incidH increases after the
    compiled delay. Also check overall totals increase.
    """
    conf = confuse.Configuration("R0Bump", __name__)
    conf.set_file(str(pipe.config_path))

    # modifier applier
    applier = compile_seir_modifiers(
        seir_modifiers_cfg=conf["seir_modifiers"].get(),
        start_date=pipe.start_date,
        n_days=pipe.T,
        n_loc=pipe.NL,
        param_names=pipe.param_names,
    )
    leaves = pipe.modifier_order()
    defaults = _leaf_defaults_in_order(conf, leaves)

    # choose a mid-season window (~6 weeks)
    win_start = pipe.start_date + dt.timedelta(days=120)
    win_end   = win_start + dt.timedelta(days=42)

    # bump r0 leaves that overlap window by x1.3
    r0_leaves = _leaves_targeting_r0_overlapping_window(conf, win_start, win_end)
    bumped = defaults.copy()
    for i, name in enumerate(leaves):
        if name in r0_leaves:
            bumped[i] = float(bumped[i]) * 1.3

    # params & series: baseline vs bumped
    params_base = applier.apply_to_params(pipe.base_params, leaf_value_array=defaults, scenario="none")
    params_bump = applier.apply_to_params(pipe.base_params, leaf_value_array=bumped,   scenario="none")

    series_base, delay_map, prob_map = _build_series_on_step_grid(pipe, params_base)
    series_bump, _, _ = _build_series_on_step_grid(pipe, params_bump)

    # helper to sum over nodes
    def _sum_nodes(arr):  # (T,L)->(T,)
        return arr.sum(axis=1)

    I_base = _sum_nodes(series_base["incidI"])
    I_bump = _sum_nodes(series_bump["incidI"])
    H_base = _sum_nodes(series_base["incidH"])
    H_bump = _sum_nodes(series_bump["incidH"])

    # step indices for window (use day indices from 0..T-2 since steps index by start-of-step day)
    step_per_day = int(round(1.0 / pipe.dt))  # 10 for 0.1
    i0 = max(0, ((win_start - pipe.start_date).days) * step_per_day)
    i1 = min(I_base.shape[0] - 1, ((win_end - pipe.start_date).days) * step_per_day)

    # incidI should be larger in-window
    sum_I_base = float(I_base[i0:i1+1].sum())
    sum_I_bump = float(I_bump[i0:i1+1].sum())
    assert sum_I_bump > sum_I_base * 1.02, f"incidI didn't increase enough in window: {sum_I_bump} vs {sum_I_base}"

    # incidH should be larger after an average delay; use the most common delay_steps for H leaves
    any_h = next((k for k in delay_map if k.startswith("incidH_") and isinstance(delay_map[k], int)), None)
    shift = int(delay_map.get(any_h, 0))
    j0 = min(H_base.shape[0]-1, i0 + shift)
    j1 = min(H_base.shape[0]-1, i1 + shift)
    sum_H_base = float(H_base[j0:j1+1].sum())
    sum_H_bump = float(H_bump[j0:j1+1].sum())
    assert sum_H_bump > sum_H_base * 1.02, f"incidH didn't increase enough after delay: {sum_H_bump} vs {sum_H_base}"

    # overall totals should also increase
    assert H_bump.sum() > H_base.sum()
    assert I_bump.sum() > I_base.sum()


@pytest.mark.slow
def test_plot_incidence_panels_baseline_vs_bump(pipe):
    """
    Save visual panels for quick inspection: daily incidI and incidH (summed over locations)
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

    t = np.arange(series_base["incidI"].shape[0]) * pipe.dt  # in days since t0
    I_base = series_base["incidI"].sum(axis=1)
    I_bump = series_bump["incidI"].sum(axis=1)
    H_base = series_base["incidH"].sum(axis=1)
    H_bump = series_bump["incidH"].sum(axis=1)

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
