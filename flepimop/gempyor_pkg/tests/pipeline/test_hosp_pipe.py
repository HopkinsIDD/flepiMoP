# tests/pipeline/test_hosp_pipe.py
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
import math
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
    _param_slice_step,
    _compute_amounts_for_step,
    _steps_to_weeks_assign,
    _mmwr_assign_from_daily,
)

# ----------------------------- helpers ---------------------------------
def _materialize_structured_example(tmp_path_factory) -> Path:
    tmp_root = tmp_path_factory.mktemp("weekly_pipeline_case")

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


def _extract_yaml_value(spec) -> float:
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


def _read_yaml_defaults_in_leaf_order(config_path: Path, leaf_order: tuple[str, ...]) -> np.ndarray:
    conf = confuse.Configuration("WeeklyHospPipelineTest", __name__)
    conf.set_file(str(config_path))
    sm = conf["seir_modifiers"].get()
    mods = sm["modifiers"]
    vals = []
    for nm in leaf_order:
        spec = mods[nm]
        vals.append(_extract_yaml_value(spec))
    return np.asarray(vals, dtype=np.float64)


def _mmwr_assign_from(start: dt.date, T: int) -> tuple[np.ndarray, int]:
    return _mmwr_assign_from_daily(start, T)


def _integrate_states(pipe, params_base, dt_days: float, *, rtol=1e-3, atol=1e-6):
    """Integrate after parsing base params into unique_strings space."""
    # Parse into unique_strings space
    params_parsed = pipe.model.compartments.parse_parameters(
        params_base,
        list(pipe.param_defs.keys()),
        pipe.unique_strings,
    )

    total_days = float(pipe.T - 1)
    n_steps = int(round(total_days / float(dt_days)))
    n_steps = max(1, n_steps)
    t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

    res = pipe.factory.solve(
        y0=pipe.initial_array.ravel(),
        parameters=params_parsed,
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    assert res.success, f"Integration failed: {res.message}"
    states = res.y.T.reshape(len(t_eval), pipe.NC, pipe.NL)
    return t_eval, states, n_steps


def _build_outcome_series_on_grid(pipe, *, dt_days: float, params: np.ndarray):
    """
    Compile outcomes, integrate on the requested step grid, reconstruct per-step
    transition amounts with the SAME parameter-expression handling as the pipeline,
    apply probs+delays to leaf-incidence outcomes, then resolve sums/aliases.
    """
    # Parse params into unique_strings space
    params_parsed = pipe.model.compartments.parse_parameters(
        params,
        list(pipe.param_defs.keys()),
        pipe.unique_strings,
    )

    outcomes_cfg = pipe.outcomes_cfg if hasattr(pipe, "outcomes_cfg") else pipe.model.outcomes_config["outcomes"].get()
    resolve_map, prob_map, delay_steps_map, sum_map, order = _compile_outcomes_from_config(
        outcomes_cfg,
        pipe.model.compartments.compartments,
        pipe.transitions,
        pipe.NL,
        bin_width_days=dt_days,
    )
    t_eval, states, n_steps = _integrate_states(pipe, params, dt_days=dt_days)

    series = {name: np.zeros((n_steps, pipe.NL), dtype=np.float64) for name in order}

    pc = pipe.precomputed
    for i in range(n_steps):
        t0 = t_eval[i]
        param_t_slice = _param_slice_step(params_parsed, t0)  # (P_unique,L)
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
            # --- CRUCIAL: use the same expression mapping as the pipeline/factory ---
            param_expr_lookup=pipe.param_expr_lookup,
            param_name_to_row=pipe.param_name_to_row_base,
        )
        for name, rows in resolve_map.items():
            if rows.size == 0:
                continue
            inc_vec = amounts[rows, :].sum(axis=0)
            p_vec = prob_map.get(name, np.ones(pipe.NL))
            shift = int(delay_steps_map.get(name, 0))
            j = i + shift
            if j < n_steps:
                series[name][j, :] += inc_vec * p_vec

    unresolved = set(sum_map.keys())
    guard = 0
    while unresolved and guard < 10000:
        guard += 1
        progress = False
        for name in list(unresolved):
            children = sum_map[name]
            if not all(ch in series for ch in children):
                continue
            combined = np.zeros_like(series[name])
            for ch in children:
                combined += series[ch]
            p_vec = prob_map.get(name, np.ones(pipe.NL))
            shift = int(delay_steps_map.get(name, 0))
            if shift:
                out = np.zeros_like(combined)
                if shift < combined.shape[0]:
                    out[shift:, :] = combined[:-shift, :] * p_vec
            else:
                out = combined * p_vec
            series[name] = out
            unresolved.remove(name)
            progress = True
        if not progress:
            break
    assert not unresolved, f"Unresolved sum outcomes remain: {unresolved}"

    compiled = (resolve_map, prob_map, delay_steps_map, sum_map, order)
    return series, t_eval, n_steps, compiled


# ----------------------------- fixtures -----------------------------------

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    # Build after environment setup; if a layer still can’t load, skip cleanly.
    cfg_path = _materialize_structured_example(tmp_path_factory)
    try:
        pipe = build_pipeline_from_config(cfg_path)
    except ValueError as e:
        if "No threading layer could be loaded" in str(e):
            pytest.skip("Numba threading layer unavailable; skipping weekly pipeline tests.")
        raise
    return pipe


# ----------------------------- baseline tests (unchanged) ------------------

def test_pipeline_builds_and_shapes(pipeline):
    weekly, age_labels, n_weeks = pipeline.evaluate(
        mod_values=_read_yaml_defaults_in_leaf_order(pipeline.config_path, pipeline.modifier_order())
    )
    A, W, L = weekly.shape
    assert A == len(age_labels) and W == n_weeks and L == pipeline.NL
    assert A > 0 and W > 0 and L > 0

    T_in = pipeline.T - 1
    assign_test, n_weeks_test = _mmwr_assign_from(pipeline.start_date, T_in)
    assert n_weeks == n_weeks_test
    assert assign_test.shape == (T_in,)
    assert assign_test.min() == 0
    assert assign_test.max() == n_weeks - 1


def test_mmwr_alignment_bounds(pipeline):
    start = pipeline.start_date
    T_in = pipeline.T - 1
    assign, n_weeks = _mmwr_assign_from(start, T_in)

    next_sun = start + dt.timedelta(days=(6 - start.weekday()) % 7)
    if next_sun == start:
        expected_first_len = min(7, T_in)
    else:
        expected_first_len = min((next_sun - start).days, T_in)

    first_len = int((assign == 0).sum())
    assert first_len == expected_first_len
    assert (assign == n_weeks - 1).any()


def test_replacement_semantics_not_multiplicative(pipeline):
    defaults = _read_yaml_defaults_in_leaf_order(pipeline.config_path, pipeline.modifier_order())
    weekly_defaults, _, _ = pipeline.evaluate(defaults)

    ones = np.ones_like(defaults)
    weekly_ones, _, _ = pipeline.evaluate(ones)

    if not np.allclose(defaults, 1.0):
        assert not np.allclose(weekly_defaults, weekly_ones, rtol=1e-10, atol=0.0)
    else:
        assert np.allclose(weekly_defaults, weekly_ones, rtol=1e-10, atol=0.0)


def test_stability_same_input_same_output(pipeline):
    defaults = _read_yaml_defaults_in_leaf_order(pipeline.config_path, pipeline.modifier_order())
    W1, _, _ = pipeline.evaluate(defaults)
    W2, _, _ = pipeline.evaluate(defaults)
    np.testing.assert_allclose(W1, W2, rtol=0, atol=0)


def test_outcome_graph_has_derived_hosp_from_incidI(pipeline):
    series, _, _, compiled = _build_outcome_series_on_grid(
        pipeline, dt_days=pipeline.dt, params=pipeline.base_params
    )
    resolve_map, prob_map, delay_steps_map, sum_map, order = compiled

    i_leaves = [n for n, rows in resolve_map.items() if n.startswith("incidI_") and "_age" in n and rows.size > 0]
    assert i_leaves, "No incidI_* leaves found"
    i_name = i_leaves[0]
    h_name = i_name.replace("incidI_", "incidH_")
    assert h_name in sum_map and len(sum_map[h_name]) == 1 and sum_map[h_name][0] == i_name
    assert float(prob_map[h_name].max()) < 0.1
    assert int(delay_steps_map[h_name]) >= 0


def test_incidH_delay_and_scale_on_synthetic_pulse(pipeline):
    outcomes_cfg = pipeline.outcomes_cfg
    resolve_map, prob_map, delay_steps_map, sum_map, order = _compile_outcomes_from_config(
        outcomes_cfg,
        pipeline.model.compartments.compartments,
        pipeline.transitions,
        pipeline.NL,
        bin_width_days=pipeline.dt,
    )

    i_leaves = [n for n, rows in resolve_map.items() if n.startswith("incidI_") and "_age" in n and rows.size > 0]
    assert i_leaves, "No incidI_* leaves with transition rows"
    i_name = i_leaves[0]
    h_name = i_name.replace("incidI_", "incidH_")
    assert h_name in sum_map and sum_map[h_name] == [i_name]

    p = float(prob_map[h_name][0])
    shift = int(delay_steps_map[h_name])

    total_days = float(pipeline.T - 1)
    n_steps = int(round(total_days / pipeline.dt))
    n_steps = max(1, n_steps)

    i_series = np.zeros((n_steps, pipeline.NL), dtype=np.float64)
    h_series = np.zeros((n_steps, pipeline.NL), dtype=np.float64)
    i_series[0, 0] = 1.0
    if shift < n_steps:
        h_series[shift, 0] += 1.0 * p

    expected = np.zeros_like(h_series)
    if shift < n_steps:
        expected[shift, 0] = i_series[0, 0] * p
    np.testing.assert_allclose(h_series, expected, rtol=0, atol=0)


@pytest.mark.slow
def test_weekly_aggregation_step_vs_daily_consistency(pipeline):
    series_step, t_eval_s, n_steps_s, _ = _build_outcome_series_on_grid(
        pipeline, dt_days=pipeline.dt, params=pipeline.base_params
    )
    assign_steps, nW1 = _steps_to_weeks_assign(
        pipeline.start_date, T_days=pipeline.T - 1, T_steps=n_steps_s, dt_days=pipeline.dt
    )
    T_days = pipeline.T - 1
    assign_days, nW2 = _mmwr_assign_from_daily(pipeline.start_date, T_days)
    assert nW1 == nW2
    nW = nW1

    day_idx = np.floor(np.arange(n_steps_s, dtype=np.float64) * float(pipeline.dt)).astype(np.int64)
    day_idx = np.clip(day_idx, 0, max(0, T_days - 1))

    def _sum_age_over_nodes(names: list[str], series: dict[str, np.ndarray]) -> np.ndarray:
        s = None
        for nm in names:
            if nm in series:
                v = series[nm].sum(axis=1)
                s = v if s is None else (s + v)
        return s if s is not None else np.zeros((n_steps_s,), dtype=np.float64)

    ages = pipeline.age_labels
    for age in ages:
        names = pipeline.age_to_names[age]
        s_step = _sum_age_over_nodes(names, series_step)
        W_step = np.zeros(nW, dtype=np.float64)
        for w in range(nW):
            mask = (assign_steps == w)
            if mask.any():
                W_step[w] = s_step[mask].sum()

        daily = np.zeros(T_days, dtype=np.float64)
        np.add.at(daily, day_idx, s_step)

        W_day = np.zeros(nW, dtype=np.float64)
        for w in range(nW):
            mask = (assign_days == w)
            if mask.any():
                W_day[w] = daily[mask].sum()

        np.testing.assert_allclose(W_step, W_day, rtol=1e-8, atol=1e-8)


def test_age_panel_grid_has_enough_axes(pipeline):
    A = len(pipeline.age_labels)
    cols = 2
    rows = math.ceil(A / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=120)
    axes = np.ravel(axes) if isinstance(axes, np.ndarray) else np.array([axes])
    assert axes.size >= A
    plt.close(fig)


@pytest.mark.slow
def test_debug_panel_plot_first4_locations(pipeline):
    defaults = _read_yaml_defaults_in_leaf_order(pipeline.config_path, pipeline.modifier_order())
    Wdef, ages, _ = pipeline.evaluate(defaults)
    Wone, _, _ = pipeline.evaluate(np.ones_like(defaults))

    A0_def = Wdef[0].sum(axis=0)
    A0_one = Wone[0].sum(axis=0)
    L = A0_def.shape[0]
    K = min(4, L)
    locs = list(range(K))

    outdir = Path(__file__).parent
    outpng = outdir / "weekly_hosp_panel_first4.png"

    fig_rows = 2
    fig_cols = 2
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(10, 6), dpi=120)
    axes = axes.ravel()
    for i, loc in enumerate(locs):
        ax = axes[i]
        ax.bar([0, 1], [A0_def[loc], A0_one[loc]])
        ax.set_xticks([0, 1], ["YAML", "Ones"])
        ax.set_title(f"{ages[0]} — Loc {loc}")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Weekly Hosp (total over weeks) — YAML defaults vs Ones — first 4 locations", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

    print(f"[artifact] saved: {outpng}")
    assert outpng.exists()


@pytest.mark.slow
def test_total_hospitalization_fraction_small(pipeline):
    # Build series on the step grid
    series, _, _, compiled = _build_outcome_series_on_grid(
        pipeline, dt_days=pipeline.dt, params=pipeline.base_params
    )
    resolve_map, prob_map, delay_steps_map, sum_map, order = compiled

    # Sum over actual (incidI_ageX -> incidH_ageX) pairs, not nonexistent root keys.
    I_total = 0.0
    H_total = 0.0
    for i_name, rows in resolve_map.items():
        if i_name.startswith("incidI_") and "_age" in i_name and rows.size > 0:
            h_name = i_name.replace("incidI_", "incidH_")
            if h_name in sum_map and sum_map[h_name] == [i_name] and h_name in series:
                I_total += float(series[i_name].sum())
                H_total += float(series[h_name].sum())

    # Guard: if there were no pairs, this test is not applicable
    assert I_total > 0.0, "No I/H pairs with nonzero incidence found."

    frac = H_total / I_total
    assert 0.0 <= frac <= 0.10, f"H/I fraction too large: {frac}"


@pytest.mark.slow
def test_pairwise_integral_ratio_matches_probability(pipeline):
    series, _, _, compiled = _build_outcome_series_on_grid(
        pipeline, dt_days=pipeline.dt, params=pipeline.base_params
    )
    resolve_map, prob_map, delay_steps_map, sum_map, order = compiled

    pairs: list[tuple[str, str]] = []
    for nm, rows in resolve_map.items():
        if nm.startswith("incidI_") and "_age" in nm and rows.size > 0:
            h = nm.replace("incidI_", "incidH_")
            if h in sum_map and sum_map[h] == [nm]:
                pairs.append((nm, h))
    assert pairs, "No I/H pairs found"

    for i_name, h_name in pairs:
        shift = int(delay_steps_map.get(h_name, 0))
        p = float(prob_map[h_name][0])
        I = series[i_name]
        H = series[h_name]
        if shift > 0 and shift < H.shape[0]:
            H_valid = H[shift:, :].sum()
            I_shift = I[:-shift, :].sum()
        else:
            H_valid = H.sum()
            I_shift = I.sum()
        if I_shift <= 1e-10:
            continue
        ratio = H_valid / I_shift
        assert np.isclose(ratio, p, rtol=5e-2, atol=1e-4), (
            f"{h_name} integral ratio {ratio:.4f} != prob {p:.4f}"
        )
