# tests/vectorized_inference/test_pymc_weekly_e2e.py
# Prior-predictive sanity tests for WeeklyHospPipeline + WeeklyHospAndFinalSOp.
# We:
#   • build the Structured_Example case,
#   • construct PyMC models wiring modifiers -> Op -> weekly_pred,
#   • sample PRIOR predictive (draws),
#   • plot INDIVIDUAL trajectories (no HDIs) to verify modifiers are injected,
#   • explicitly test the new location-structured modifier path (4th Op arg).

import os
import platform
from ctypes.util import find_library
import pytensor.tensor as pt

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYMC_PROGRESSBAR", "1")

# --- Choose numba threading layer before importing numba users ---
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
    add = [p for p in extra if os.path.exists(p) and (p not in cur)]
    if add:
        os.environ["DYLD_LIBRARY_PATH"] = (":".join(add) + (":" + cur if cur else ""))

_layer = _choose_numba_layer()
_maybe_patch_dylib_path(_layer)
os.environ.setdefault("NUMBA_THREADING_LAYER", _layer)
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(4, max(1, os.cpu_count() or 1))))

# ---------------------------------------------------------------------------------
# Regular imports (SAFE now that env is set)
# ---------------------------------------------------------------------------------
from pathlib import Path
import shutil
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm
import confuse

from gempyor.vectorization_experiments import autotune_all, get_autotune_config
from gempyor.hosp_weekly_pipeline import build_pipeline_from_config
from gempyor.pymc_weekly_op import (
    WeeklyHospAndFinalSOp,
    build_weekly_model,
)

# ----------------------------- helpers ---------------------------------
def _materialize_structured_example(tmp_path_factory) -> Path:
    """Copy examples/tutorials/Structured_Example_Seeding_Alt.yml & inputs into a temp root with absolute paths."""
    tmp_root = tmp_path_factory.mktemp("weekly_prior_case")

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
    cfg_name = "Structured_Example_Seeding_Alt.yml"
    cfg_path = tmp_root / cfg_name
    shutil.copyfile(tutorial_dir / cfg_name, cfg_path)

    # Patch relative -> absolute
    text = cfg_path.read_text()
    text = text.replace("model_input/", str(tmp_root / "model_input") + "/")
    cfg_path.write_text(text)

    return cfg_path


def _safe_autotune():
    """Call autotune_all(), tolerant across gempyor versions."""
    try:
        autotune_all(quiet=True)
        return
    except TypeError:
        pass
    except ValueError:
        try:
            from numba.np.ufunc import parallel as nbpar
            env_n = int(os.environ.get("NUMBA_NUM_THREADS", "1"))
            env_n = max(1, env_n)
            nbpar.set_num_threads(env_n)
        except Exception:
            pass
    try:
        autotune_all(quiet=True)
    except Exception:
        pass
    try:
        print(f"[autotune active] {get_autotune_config()}")
    except Exception:
        pass


def _stack_samples(arr: np.ndarray) -> np.ndarray:
    """
    Accept prior/prior_predictive weekly_pred array with shape either
      (draw, A, W, L)  or  (chain, draw, A, W, L)
    and return (S, A, W, L) where S=total samples.
    """
    if arr.ndim == 4:
        return arr  # (draw, A, W, L)
    if arr.ndim == 5:
        c, d, A, W, L = arr.shape
        return arr.reshape(c * d, A, W, L)
    raise ValueError(f"Unexpected weekly_pred ndim={arr.ndim}, shape={arr.shape}")


def _idata_get(idata, var_name: str) -> np.ndarray:
    """
    Fetch a variable from either the `prior_predictive` or `prior` group (PyMC 5 changed behavior
    when there are no observed RVs). Returns the numpy array values.
    """
    for grp in ("prior_predictive", "prior"):
        grp_obj = getattr(idata, grp, None)
        if grp_obj is not None and var_name in grp_obj:
            return grp_obj[var_name].values
    raise AssertionError(f"Variable '{var_name}' not found in prior/prior_predictive groups.")


def _yaml_defaults_in_leaf_order(config_path: Path, leaf_order: tuple[str, ...]) -> np.ndarray:
    """Robustly extract default replacement multipliers in the pipeline’s leaf order."""
    conf = confuse.Configuration("WeeklyPriorDefaults", __name__)
    conf.set_file(str(config_path))
    sm = conf["seir_modifiers"].get()
    mods = sm["modifiers"]

    def _extract(spec):
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

    return np.asarray([_extract(mods[nm]) for nm in leaf_order], dtype=np.float64)


def _run_op_forward(op: WeeklyHospAndFinalSOp, mods: np.ndarray, pR: np.ndarray, lambda_ext: np.ndarray, mods_loc: np.ndarray | None = None):
    """Invoke Op.perform directly (3 or 4 inputs) and return (weekly, S_final) as numpy arrays."""
    outs_w = [None]
    outs_s = [None]
    if mods_loc is None:
        op.perform(None, [mods, pR, lambda_ext], [outs_w, outs_s])
    else:
        op.perform(None, [mods, pR, lambda_ext, mods_loc], [outs_w, outs_s])
    return np.asarray(outs_w[0], dtype=np.float64), np.asarray(outs_s[0], dtype=np.float64)


# ----------------------------- fixtures --------------------------------

@pytest.fixture(scope="module")
def pipeline_and_op(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)
    pipe = build_pipeline_from_config(cfg_path)
    _safe_autotune()
    op = WeeklyHospAndFinalSOp(pipe)
    return pipe, op


# ------------------------------ tests -----------------------------------

@pytest.mark.slow
def test_prior_predictive_trajectories_show_modifier_injection(pipeline_and_op, tmp_path):
    """
    Prior-predictive check (3-input Op path): sample 200 draws and plot INDIVIDUAL weekly trajectories
    (sum over age) for a couple of locations. If modifiers are injected correctly,
    trajectories should visibly vary (and stats should confirm nonzero variance).
    """
    pipe, op = pipeline_and_op

    # Build model WITHOUT observed data — we want pure prior predictive.
    with build_weekly_model(pipe, op=op, y_obs=None) as model:
        idata = pm.sample_prior_predictive(samples=200, random_seed=123, var_names=["weekly_pred", "mods"])

    # Extract weekly prior(-predictive) samples and modifiers
    weekly_vals = _idata_get(idata, "weekly_pred")   # (draw, A, W, L) or (chain, draw, A, W, L)
    weekly = _stack_samples(weekly_vals)             # -> (S, A, W, L)

    mods_vals = _idata_get(idata, "mods")            # (draw, M) or (chain, draw, M)
    if mods_vals.ndim == 2:
        mods_samples = mods_vals                     # (S, M)
    else:
        c, d, M = mods_vals.shape
        mods_samples = mods_vals.reshape(c * d, M)

    S, A, W, L = weekly.shape
    assert S >= 1 and A >= 1 and W >= 1 and L >= 1

    # ----- NUMERIC sanity: require *some* variance across samples per week, per plotted location
    locs_to_plot = [0, min(L - 1, L // 2)]
    for loc in locs_to_plot:
        series = weekly[:, :, :, loc].sum(axis=1)  # (S, W): sum over age
        var_per_week = series.var(axis=0)
        assert float(var_per_week.max()) > 0.0, (
            "No variation in prior-predictive weekly trajectories — modifiers may not be injected."
        )

    # ----- PLOTS: individual trajectories (no HDIs)
    outdir_env = os.environ.get("E2E_OUTDIR", "").strip()
    outdir = Path(outdir_env) if outdir_env else Path(tmp_path)
    outdir.mkdir(parents=True, exist_ok=True)

    weeks = np.arange(W)

    # 1) Per-location trajectory spaghetti plots (sum over age)
    for loc in locs_to_plot:
        series = weekly[:, :, :, loc].sum(axis=1)  # (S, W)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
        for s in range(min(200, S)):
            ax.plot(weeks, series[s], alpha=0.8, linewidth=1.0)
        ax.set_title(f"Prior predictive trajectories — sum over ages, location {loc}")
        ax.set_xlabel("MMWR week")
        ax.set_ylabel("Weekly hospitalizations (sum over age)")
        ax.grid(True, alpha=0.3)
        outpng = outdir / f"prior_trajs_loc{loc}.png"
        fig.tight_layout()
        fig.savefig(outpng, bbox_inches="tight")
        plt.close(fig)
        print(f"[artifact] saved: {outpng}")
        assert outpng.exists()

    # 2) Quick view of modifier samples (first ~12 leaves)
    M_show = min(12, mods_samples.shape[1])
    fig, axes = plt.subplots(M_show, 1, figsize=(10, 1.6 * M_show), dpi=120, sharex=True)
    if M_show == 1:
        axes = [axes]
    for i in range(M_show):
        ax = axes[i]
        ax.plot(mods_samples[:min(200, mods_samples.shape[0]), i], ".", alpha=0.7, markersize=4)
        ax.set_ylabel(f"mod[{i}]")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("sample index (prior)")
    fig.suptitle("Modifier samples (subset) — should vary under prior", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outpng = outdir / "prior_modifiers_subset.png"
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)
    print(f"[artifact] saved: {outpng}")
    assert outpng.exists()


@pytest.mark.slow
def test_location_modifier_injection_affects_locations_differently(pipeline_and_op):
    """
    Direct Op check (4-input path): bump a single leaf in a single location via mods_loc
    and confirm the change in that location is more concentrated than a shared bump.
    This exercises the new per-location modifier injection path.
    """
    pipe, op = pipeline_and_op
    A, W, L = op.weekly_shape
    M = len(pipe.modifier_order())

    # Baseline controls
    defaults = _yaml_defaults_in_leaf_order(pipe.config_path, pipe.modifier_order())
    pR = np.full((A, L), 0.40, dtype=np.float64)
    lambda_ext = np.ones(L, dtype=np.float64)

    weekly_base, _ = _run_op_forward(op, defaults, pR, lambda_ext, mods_loc=None)
    agg_base = weekly_base.sum(axis=0).sum(axis=0)  # (L,)

    # Pick a leaf to bump (use the first by convention; absolute change will be measured)
    leaf_idx = 0
    bump = 1.20

    # Shared bump (3-input path): bump the same leaf in ALL locations
    shared = defaults.copy()
    shared[leaf_idx] *= bump
    weekly_shared, _ = _run_op_forward(op, shared, pR, lambda_ext, mods_loc=None)
    agg_shared = weekly_shared.sum(axis=0).sum(axis=0)  # (L,)
    delta_shared = np.abs(agg_shared - agg_base)        # (L,)

    # Location-specific bump (4-input path): bump only location 0 for the same leaf
    mods_loc = np.ones((M, L), dtype=np.float64)
    mods_loc[leaf_idx, 0] = bump
    weekly_loc, _ = _run_op_forward(op, defaults, pR, lambda_ext, mods_loc=mods_loc)
    agg_loc = weekly_loc.sum(axis=0).sum(axis=0)        # (L,)
    delta_loc = np.abs(agg_loc - agg_base)              # (L,)

    # Concentration check: the per-location bump should concentrate more change in loc 0
    eps = 1e-9
    frac_other_shared = (delta_shared.sum() - delta_shared[0]) / (delta_shared.sum() + eps)
    frac_other_loc = (delta_loc.sum() - delta_loc[0]) / (delta_loc.sum() + eps)

    assert delta_loc[0] > 0.0, "Per-location bump produced no change in target location."
    assert frac_other_loc < 0.95 * frac_other_shared, (
        "Per-location modifier did not concentrate change in the targeted location as expected."
    )


@pytest.mark.slow
def test_prior_predictive_with_location_modifiers_compiles_and_varies(pipeline_and_op):
    """
    Build a minimal PyMC model that **passes mods_loc** (4th argument) into the Op,
    then sample prior predictive and verify between-location variability exists.
    """
    pipe, op = pipeline_and_op
    A, W, L = op.weekly_shape
    mod_names = tuple(pipe.modifier_order())
    M = len(mod_names)

    coords = {
        "age": np.array(pipe.age_labels, dtype=object),
        "week": np.arange(W),
        "location": np.arange(L),
        "modifier": np.array(mod_names, dtype=object),
    }

    defaults = _yaml_defaults_in_leaf_order(pipe.config_path, mod_names)

    with pm.Model(coords=coords) as m:
        # Shared center on modifiers
        shared_mods = pm.LogNormal("mods_shared", mu=np.log(defaults + 1e-12), sigma=0.4, dims=("modifier",))
        # Location deviations (LogNormal ~ centered at 1.0)
        mods_loc = pm.LogNormal("mods_loc", mu=0.0, sigma=0.25, dims=("modifier", "location"))
        # pR constant for simplicity (replicated across age)
        pR = pm.Deterministic("pR", pt.full((A, L), 0.40), dims=("age", "location"))
        # lambda_ext per location ~ LogNormal close to 1.0
        lambda_ext_loc = pm.LogNormal("lambda_ext_loc", mu=0.0, sigma=0.05, dims=("location",))

        weekly_t, _ = op(shared_mods, pR, lambda_ext_loc, mods_loc)  # <-- 4-input call
        weekly = pm.Deterministic("weekly_pred", weekly_t, dims=("age", "week", "location"))

        idata = pm.sample_prior_predictive(samples=100, random_seed=321, var_names=["weekly_pred", "mods_shared", "mods_loc"])

    weekly_vals = _idata_get(idata, "weekly_pred")   # (draw, A, W, L) or (chain, draw, A, W, L)
    weekly = _stack_samples(weekly_vals)             # (S, A, W, L)
    S, A, W, L = weekly.shape

    # Aggregate and check there is between-location variability across samples
    agg = weekly.sum(axis=1).sum(axis=1)  # (S, L)
    var_across_samples = agg.var(axis=0)  # per-location variance across samples
    assert np.isfinite(var_across_samples).all()
    assert (var_across_samples > 0).any(), "No variability across locations under prior with location modifiers."
