# tests/pipeline/test_mods_incid.py
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
import numpy as np
import pytest
import confuse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gempyor.model_info import ModelInfo
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
    tmp_root = tmp_path_factory.mktemp("actual_modifiers_case")

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

    # Patch relative -> absolute paths
    text = cfg_path.read_text()
    text = text.replace("model_input/", str(tmp_root / "model_input") + "/")
    cfg_path.write_text(text)
    return cfg_path


def _load_params_from_config(cfg_path: Path):
    """Return base (unparsed) parameter tensor aligned with base param_names."""
    conf = confuse.Configuration("ActualModifiers", __name__)
    conf.set_file(str(cfg_path))

    model = ModelInfo(
        config=conf,
        config_filepath=str(cfg_path),
        path_prefix=str(cfg_path.parent),
        setup_name="Structured_Example",
        seir_modifiers_scenario="none",
    )

    # Ensure components are initialized
    _ = model.initial_conditions.get_from_config(sim_id=0, modinf=model)
    _ = model.compartments.get_transition_array()

    param_defs = conf["seir"]["parameters"].get()
    base = model.parameters.parameters_quick_draw(model.n_days, model.nsubpops)  # (P_base, T, L)
    param_names = np.array(list(param_defs.keys()))
    return dict(conf=conf, model=model, params_base=base, param_names=param_names)


def _pidx(name: str, names: np.ndarray) -> int | None:
    if name in names:
        return int(np.where(names == name)[0][0])
    return None


def _sum_incidI(series_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Sum top-level incidI if present; otherwise sum all incidI_* leaves."""
    if "incidI" in series_dict:
        return series_dict["incidI"].sum(axis=1)
    vec = None
    for k, v in series_dict.items():
        if k.startswith("incidI_"):
            s = v.sum(axis=1)
            vec = s if vec is None else (vec + s)
    if vec is None:
        any_arr = next(iter(series_dict.values()))
        vec = np.zeros(any_arr.shape[0], dtype=np.float64)
    return vec


# ---------------- tests -------------------------------------------------
def test_panel_r0_baseline_vs_actual(tmp_path_factory):
    """
    Apply the **actual** YAML modifiers (scenario='none') and save a 2×2 panel
    of baseline vs modified r0(t) for the first 4 locations.
    """
    cfg_path = _materialize_structured_example(tmp_path_factory)
    payload = _load_params_from_config(cfg_path)
    conf, model, params_base, param_names = (
        payload["conf"], payload["model"], payload["params_base"], payload["param_names"]
    )

    applier = compile_seir_modifiers(
        seir_modifiers_cfg=conf["seir_modifiers"].get(),
        start_date=model.ti,
        n_days=model.n_days,
        n_loc=model.nsubpops,
        param_names=param_names,
    )
    params_mod = applier.apply_to_params(params_base, scenario="none")

    r0_idx = _pidx("r0", param_names)
    assert r0_idx is not None, "Expected 'r0' in param_names"

    baseline = params_base[r0_idx]  # (T,L)
    modified = params_mod[r0_idx]   # (T,L)

    outdir = Path(__file__).parent
    outpng = outdir / "r0_panel_actual_modifiers_first4.png"

    T, L = baseline.shape
    K = min(4, L)
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), dpi=120, sharex=True)
    axes = axes.ravel()
    for i in range(K):
        ax = axes[i]
        ax.plot(t, baseline[:, i], label="baseline r0", linewidth=1.4)
        ax.plot(t, modified[:, i], "--", label="modified r0", linewidth=1.4)
        ax.set_title(f"Location {i}")
        ax.grid(True, alpha=0.3)
        if i in (2, 3): ax.set_xlabel("day")
        if i in (0, 2): ax.set_ylabel("r0")
    for j in range(K, 4):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Baseline vs Modified r0(t) — first 4 locations", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

    # sanity: modifiers actually changed r0 somewhere
    assert not np.allclose(baseline, modified)
    print(f"[artifact] saved: {outpng}")
    assert outpng.exists()


@pytest.mark.slow
def test_panel_incidI_baseline_vs_actual(tmp_path_factory):
    """
    Propagation check (visual): using the **actual** modifiers, plot daily incidI
    (sum over locations) baseline vs modified. Skips if no threading layer can load.
    """
    cfg_path = _materialize_structured_example(tmp_path_factory)
    try:
        pipe = build_pipeline_from_config(cfg_path, dt_days=0.1)
    except ValueError as e:
        if "No threading layer could be loaded" in str(e):
            pytest.skip("Numba threading layer unavailable; skipping incidence panel.")
        raise

    # compile outcomes for the pipeline step width (affects delays)
    resolve_map, prob_map, delay_steps_map, sum_map, order = _compile_outcomes_from_config(
        pipe.outcomes_cfg,
        pipe.model.compartments.compartments,
        pipe.transitions,
        pipe.NL,
        bin_width_days=pipe.dt,
    )

    # modifiers from SAME config, applied to BASE tensor
    conf = confuse.Configuration("ActualModifiersIncid", __name__)
    conf.set_file(str(cfg_path))
    applier = compile_seir_modifiers(
        seir_modifiers_cfg=conf["seir_modifiers"].get(),
        start_date=pipe.start_date,
        n_days=pipe.T,
        n_loc=pipe.NL,
        param_names=pipe.param_names,  # base names
    )

    params_base = pipe.base_params
    params_mod  = applier.apply_to_params(pipe.base_params, scenario="none")

    # ---- IMPORTANT: for this test, run the solver on the **parsed unique** tensor
    # and keep the RHS in "direct indexing" mode (no expression wiring).
    def _make_factory():
        return RHSfactory(
            precomputed=pipe.precomputed,
            param_time_mode="step",
        )

    def _series(params_base_like):
        # Parse BASE -> UNIQUE (aligns P with transitions[2,*] indices)
        params_unique = pipe.model.compartments.parse_parameters(
            params_base_like,
            pipe.param_defs,
            pipe.unique_strings,
        )

        total_days = float(pipe.T - 1)
        n_steps = max(1, int(round(total_days / pipe.dt)))
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

        custom_factory = _make_factory()
        res = custom_factory.solve(
            y0=pipe.initial_array.ravel(),
            parameters=params_unique,  # UNIQUE tensor; factory indexes pidx directly
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-3, atol=1e-6,
        )
        assert res.success, f"Solve failed: {res.message}"
        C, L = pipe.NC, pipe.NL
        states = res.y.T.reshape(len(t_eval), C, L)

        series = {name: np.zeros((n_steps, L), dtype=np.float64) for name in order}
        pc = pipe.precomputed
        for i in range(n_steps):
            t0 = t_eval[i]
            # Slice the UNIQUE tensor
            param_t = _param_slice_step(params_unique, t0)  # (P_unique, L)
            amounts, _ = _compute_amounts_for_step(
                states_current=states[i],
                transitions=pipe.transitions,
                proportion_info=pipe.proportion_info,
                transition_sum_compartments=pipe.transition_sum_compartments,
                param_t_slice=param_t,
                percent_day_away=pc["percent_day_away"],
                prop_who_move=pc["proportion_who_move"],
                mobility_data=pc["mobility_data"],
                mobility_indptr=pc["mobility_data_indices"],
                mobility_indices=pc["mobility_row_indices"],
                population=pc["population"],
                # DIRECT INDEXING path (match the UNIQUE tensor above)
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

        # sums/aliases
        unresolved = set(sum_map.keys())
        while unresolved:
            progressed = False
            for name in list(unresolved):
                kids = sum_map[name]
                if not all(k in series for k in kids):
                    continue
                combined = np.zeros_like(series[name])
                for k in kids:
                    combined += series[k]
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
                progressed = True
            if not progressed:
                break
        return series

    S_base = _series(params_base)
    S_mod  = _series(params_mod)

    I_base = _sum_incidI(S_base)
    I_mod  = _sum_incidI(S_mod)
    t = np.arange(I_base.shape[0]) * pipe.dt

    outdir = Path(__file__).parent
    outpng = outdir / "incidI_panel_actual_modifiers.png"

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), dpi=120)
    ax.plot(t, I_base, label="incidI baseline", linewidth=1.5)
    ax.plot(t, I_mod,  "--", label="incidI modified", linewidth=1.5)
    ax.set_xlabel("day")
    ax.set_ylabel("sum over locations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.suptitle("Incidence baseline vs actual modifiers (Structured_Example)", y=0.98)
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)

    assert not np.allclose(I_base, I_mod), "incidI identical with/without modifiers"
    print(f"[artifact] saved: {outpng}")
    assert outpng.exists()
