# Fast-mode + no-mobility calibration test against real CSV weekly hosp data (ALL locations).
# Runs the Op with fast integrator tolerances and mobility disabled so each
# location can improve independently. Includes light prior spaghetti, a small
# posterior sample, PPC, and plotting artifacts for ALL states present in data.

import os
import platform
from ctypes.util import find_library

# ---- threading env (set before imports that touch BLAS/Numba) ----
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYMC_PROGRESSBAR", "1")

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

# -----------------------------------------------------------------------
# Regular imports
# -----------------------------------------------------------------------
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm
import re

from gempyor.vectorization_experiments import autotune_all, get_autotune_config
from gempyor.hosp_weekly_pipeline import build_pipeline_from_config, WeeklyHospPipeline
from gempyor.pymc_weekly_op import (
    WeeklyHospAndFinalSOp,
    build_weekly_model,
    _yaml_defaults_in_leaf_order,   # for prior-mean seasonal line
)
from gempyor.vectorized_modifiers import compile_seir_modifiers


# ============================= helpers =============================

def _materialize_structured_example(tmp_path_factory) -> Path:
    """Copy Structured_Example.yml & inputs into a temp root with absolute paths (seeding ON)."""
    tmp_root = tmp_path_factory.mktemp("weekly_infer_realdata_all_states")

    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

    # Inputs
    src_struct = tutorial_dir / "model_input" / "Structured_Example"
    dst_struct = tmp_root / "model_input" / "Structured_Example"
    shutil.copytree(src_struct, dst_struct, dirs_exist_ok=True)

    src_ic = tutorial_dir / "model_input" / "initial_condition"
    dst_ic = tmp_root / "model_input" / "initial_condition"
    shutil.copytree(src_ic, dst_ic, dirs_exist_ok=True)

    # Config with seeding support + updated modifiers
    cfg_name = "Structured_Example.yml"
    cfg_path = tmp_root / cfg_name
    shutil.copyfile(tutorial_dir / cfg_name, cfg_path)

    # Patch relative -> absolute
    text = cfg_path.read_text()
    text = text.replace("model_input/", str(tmp_root / "model_input") + "/")
    cfg_path.write_text(text)
    return cfg_path


def _safe_autotune():
    """Run autotune; keep threads within NUMBA_NUM_THREADS if needed."""
    try:
        autotune_all(quiet=False)
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
        autotune_all(quiet=False)
    except Exception:
        pass
    try:
        print(f"[autotune active] {get_autotune_config()}")
    except Exception:
        pass


def _mmwr_week_assign(model_start_date, n_days: int) -> np.ndarray:
    """Map each day index (0..n_days-1) to an MMWR week index aligned to the model start."""
    if n_days <= 0:
        return np.zeros(0, dtype=np.int64)
    w = model_start_date.weekday()  # Mon=0..Sun=6
    offset_to_sun = (6 - w) % 7
    first_len = 7 if offset_to_sun == 0 else offset_to_sun
    first_len = min(first_len, n_days)
    assign = np.empty(n_days, dtype=np.int64)
    assign[:first_len] = 0
    if n_days > first_len:
        rest = n_days - first_len
        assign[first_len:] = 1 + (np.arange(rest, dtype=np.int64) // 7)
    return assign


def _load_and_align_csv_to_weeks(
    csv_path: Path,
    pipe: WeeklyHospPipeline,
    op: WeeklyHospAndFinalSOp,
    subset_sources: tuple[str, ...] | None = None,
):
    """
    Read long CSV of *daily* total hospitalizations -> weekly aggregated matrix (W,L) aligned to model start_date.
    CSV columns expected: 'date', 'source', 'incidH'.
    If `subset_sources` is provided, only those sources are considered and columns are ordered exactly as given.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"date", "source", "incidH"}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.copy()
    df["incidH"] = pd.to_numeric(df["incidH"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Optional source subset (enforce requested ordering)
    if subset_sources:
        df = df[df["source"].isin(subset_sources)]
        if df.empty:
            raise ValueError(f"No rows left after filtering to {subset_sources}")
        # use a categorical to preserve the requested order throughout pivots/sorts
        df["source"] = pd.Categorical(df["source"], categories=list(subset_sources), ordered=True)
        loc_names = tuple(subset_sources)
    else:
        # Preserve first-appearance order within the (aligned) data
        loc_names = None

    # Collapse to daily totals per (source, date)
    df = df.groupby(["source", "date"], sort=False, as_index=False)["incidH"].sum()

    # Align to model horizon
    start_date = pipe.start_date
    T = pipe.T
    df["day_index"] = (df["date"].dt.date - start_date).apply(lambda d: d.days)
    df = df[(df["day_index"] >= 0) & (df["day_index"] < T)].copy()
    if df.empty:
        raise ValueError("No rows overlap the model horizon after alignment.")

    # Map to model-aligned MMWR weeks
    assign = _mmwr_week_assign(start_date, T)
    df["week_idx"] = df["day_index"].map(lambda i: int(assign[i]))

    # Determine location names/order
    if loc_names is None:
        loc_names = tuple(pd.unique(df["source"]))
    L_data = len(loc_names)
    L_model = op.locations
    if L_data != L_model:
        raise AssertionError(
            f"Location count mismatch after filtering: data has {L_data} 'source' values vs model L={L_model}."
        )

    # Weekly totals per location
    weekly = (
        df.groupby(["week_idx", "source"], sort=False)["incidH"]
          .sum()
          .reset_index()
    )

    # Pivot to (W, L) using the Op’s week count and the enforced column order
    W = op.n_weeks
    pivot = (
        weekly.pivot(index="week_idx", columns="source", values="incidH")
              .reindex(range(W))
              .sort_index()
              .reindex(columns=list(loc_names))
    )

    # Observed weeks: any location has data
    obs_weeks = np.flatnonzero(pivot.notna().any(axis=1).values)

    # For inference, fill missing with 0.0 (PyMC cannot take NaN in observed)
    y_full = pivot.fillna(0.0).to_numpy(dtype=float)  # (W, L)

    # Convert category back to string for downstream labeling
    loc_names = tuple(map(str, loc_names))

    return y_full, obs_weeks, loc_names


def _plot_weekly_targets(ax: plt.Axes, y_vec: np.ndarray, *, label: str = "Target (data)", color="0.1"):
    """Plot weekly targets (zeros stand in for missing weeks)."""
    W = y_vec.shape[0]
    weeks = np.arange(W)
    ax.step(weeks, y_vec, where="mid", linewidth=1.4, alpha=0.95, label=label, color=color)


def _age_lower_bound(label: str) -> int:
    """Extract lower-edge integer from an age label (handles 'age0to4', '5–17', '65+', etc.)."""
    s = str(label).lower().replace("age", "").replace("to", "-").replace("–", "-").replace("—", "-")
    s = s.replace("plus", "+").replace("p", "+")
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _panel_per_location(idata, y_obs_full, obs_weeks, loc_names, pipe, outdir: Path):
    """
    For each location:
      - Top: aggregated posterior predictive **of y** (mean + 95% HDI) vs observed,
             aligned on the full week index (0..W-1) to avoid any offset when some
             weeks were missing in the CSV.
      - Bottom: one subplot per age group (mean + 95% HDI) in ascending age-bin order.
    """
    ages = tuple(pipe.age_labels)
    A = len(ages)
    W = y_obs_full.shape[0]

    order = np.argsort([_age_lower_bound(a) for a in ages])
    ages_sorted = [ages[i] for i in order]

    # >>> Use the observation-level predictive if available <<<
    if "y" in idata.posterior_predictive:
        agg_ppc = idata.posterior_predictive["y"].values  # (chain, draw, W, L)
        mean_label = "Posterior mean (obs model)"
    else:
        # fallback (process level)
        if "weekly_pred_sum_age_shifted" in idata.posterior_predictive:
            agg_ppc = idata.posterior_predictive["weekly_pred_sum_age_shifted"].values
        else:
            agg_ppc = idata.posterior_predictive["weekly_pred_sum_age"].values
        mean_label = "Posterior mean (process)"

    age_ppc = idata.posterior_predictive["weekly_pred"].values  # (chain, draw, A, W, L)
    weeks = np.arange(W)

    for loc_idx, loc_name in enumerate(loc_names):
        fig_width = min(30, max(14, 3.2 * A))
        fig = plt.figure(figsize=(fig_width, 7.5), dpi=120)

        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.3, 1.0], hspace=0.35)
        ax_agg = fig.add_subplot(gs[0, 0])
        bottom = gs[1].subgridspec(1, A, wspace=0.25)

        # Aggregated (ALIGN ON FULL WEEK GRID)
        samples = agg_ppc[:, :, :, loc_idx]            # (chain, draw, W)
        mean = samples.mean(axis=(0, 1))               # (W,)
        hdi = az.hdi(samples, hdi_prob=0.95)           # (W, 2)
        ax_agg.fill_between(weeks, hdi[:, 0], hdi[:, 1], alpha=0.25, step="mid", label="95% HDI")
        ax_agg.plot(weeks, mean, linewidth=1.5, label=mean_label)
        y_loc = y_obs_full[:, loc_idx]                 # full length (zeros where missing)
        _plot_weekly_targets(ax_agg, y_loc, label="Observed", color="0.1")

        ax_agg.set_title(f"{loc_name} — Aggregated hospitalizations (all ages)")
        ax_agg.set_xlabel("Week")
        ax_agg.set_ylabel("Hosp")
        ax_agg.grid(True, alpha=0.3)
        ax_agg.legend(loc="upper right")

        # Per-age
        for j, a_idx in enumerate(order):
            ax = fig.add_subplot(bottom[0, j], sharex=None if j == 0 else fig.axes[-1])
            samples_a = age_ppc[:, :, a_idx, :, loc_idx]
            mean_a = samples_a.mean(axis=(0, 1))
            hdi_a = az.hdi(samples_a, hdi_prob=0.95)
            ax.fill_between(weeks, hdi_a[:, 0], hdi_a[:, 1], alpha=0.20, step="mid")
            ax.plot(weeks, mean_a, linewidth=1.2)
            ax.set_title(str(ages_sorted[j]))
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Week")
            if j == 0:
                ax.set_ylabel("Hosp")

        fig.suptitle(f"Posterior predictive — {loc_name}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        outpng = outdir / f"ppc_panel_{loc_idx:02d}_{loc_name}.png"
        fig.savefig(outpng, bbox_inches="tight")
        plt.close(fig)


def _idata_get(idata, var_name: str):
    """Fetch a variable from either `prior_predictive` or `prior` InferenceData group."""
    for grp in ("prior_predictive", "prior"):
        grp_obj = getattr(idata, grp, None)
        if grp_obj is not None and var_name in grp_obj:
            return grp_obj[var_name].values
    raise AssertionError(f"Variable '{var_name}' not found in prior/prior_predictive groups.")

def _stack_samples(arr: np.ndarray) -> np.ndarray:
    """weekly_pred array (draw,A,W,L) or (chain,draw,A,W,L) -> (S,A,W,L)."""
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5:
        c, d, A, W, L = arr.shape
        return arr.reshape(c * d, A, W, L)
    raise ValueError(f"Unexpected ndim for weekly_pred: {arr.shape}")


def _build_age_masks(pipe: WeeklyHospPipeline):
    """Return (is_S_per_age, is_R_per_age, is_total_per_age) boolean masks over compartments."""
    df = pipe.model.compartments.compartments
    comp_age = df["age_strata"].astype(str).values
    comp_stage = df["infection_stage"].astype(str).values
    ages = tuple(pipe.age_labels)
    NC = pipe.NC
    def _norm_age(s: str) -> str:
        s = str(s).lower().replace("age", "").replace("to", "-").replace("–", "-").replace("—", "-")
        out = []
        for ch in s:
            if ch.isdigit(): out.append(ch)
            elif ch in "-_": out.append("_")
            elif ch == "+": out.append("p")
        key = []
        for c in out:
            if not (key and key[-1] == "_" and c == "_"):
                key.append(c)
        return "".join(key).strip("_")
    comp_age_norm = np.array([_norm_age(a) for a in comp_age], dtype=object)
    age_tokens = tuple(_norm_age(a) for a in ages)
    is_S = np.fromiter((str(st).startswith("S") for st in comp_stage), dtype=bool, count=NC)
    is_R = np.fromiter((str(st).startswith("R") for st in comp_stage), dtype=bool, count=NC)
    s_masks, r_masks, tot_masks = [], [], []
    for tok in age_tokens:
        m_age = (comp_age_norm == tok)
        if not m_age.any():
            m_age = np.ones(NC, dtype=bool)
        s_masks.append(m_age & is_S)
        r_masks.append(m_age & is_R)
        tot_masks.append(m_age)
    return tuple(s_masks), tuple(r_masks), tuple(tot_masks), ages


def _precompute_sr_mass0(pipe: WeeklyHospPipeline) -> np.ndarray:
    """sr_mass0[a, l] = (total mass in age a, loc l) minus (non S/R mass) at t0."""
    initial = pipe.initial_array
    s_masks, r_masks, tot_masks, ages = _build_age_masks(pipe)
    L = pipe.NL
    A = len(ages)
    sr = np.zeros((A, L), dtype=np.float64)
    for a, (m_s, m_r, m_tot) in enumerate(zip(s_masks, r_masks, tot_masks)):
        m_other = m_tot & (~m_s) & (~m_r)
        N_tot = initial[m_tot, :].sum(axis=0)
        N_other = initial[m_other, :].sum(axis=0)
        sr[a, :] = np.maximum(N_tot - N_other, 0.0)
    return sr


# ---------- r0-effective helpers (match Op's transforms) ----------

def _week_centers_from_assign(assign: np.ndarray) -> np.ndarray:
    """Float center day index for each week in assign (0..W-1)."""
    if assign.size == 0:
        return np.zeros(0, dtype=float)
    W = int(assign.max()) + 1
    centers = np.zeros(W, dtype=float)
    for w in range(W):
        idx = np.flatnonzero(assign == w)
        if idx.size:
            centers[w] = 0.5 * (float(idx[0]) + float(idx[-1]))
        else:
            centers[w] = (centers[w - 1] + 7.0) if w > 0 else 0.0
    return centers

def _interp_weekly_to_daily(scale_w: np.ndarray, centers: np.ndarray, T_days: int) -> np.ndarray:
    """Linear interp of (W,) weekly values to (T_days,) daily using week centers."""
    t = np.arange(T_days, dtype=float)
    return np.interp(t, centers, scale_w.astype(float),
                     left=float(scale_w[0]), right=float(scale_w[-1]))

def _log_hann_smooth(x_daily: np.ndarray, N: int) -> np.ndarray:
    """Apply length-N Hann smoothing in log space to a positive daily series."""
    if N is None or N < 3:
        return x_daily
    k = np.hanning(int(N)).astype(float)
    if not np.isfinite(k).all() or k.sum() <= 0:
        return x_daily
    k /= k.sum()
    lx = np.log(np.clip(x_daily, 1e-12, np.inf))
    num = np.convolve(lx, k, mode="same")
    den = np.convolve(np.ones_like(lx), k, mode="same")
    den = np.maximum(den, 1e-12)
    return np.exp(num / den)

def _effective_r0_daily(base_params: np.ndarray,
                        applier,
                        leaf_eff: np.ndarray,
                        r0_idx: int,
                        loc: int,
                        scale_w: np.ndarray | None,
                        day_to_week: np.ndarray,
                        smooth_N: int) -> np.ndarray:
    """
    Recreate the Op's effective r0(t,loc):
      base -> apply modifiers -> (optional) weekly scale (interp to days) -> log-Hann smoothing
    """
    injected = applier.apply_to_params(base_params, leaf_value_array=leaf_eff, scenario="none")
    r0 = injected[r0_idx, :, loc].astype(float).copy()               # daily from config/modifiers
    if scale_w is not None:
        centers = _week_centers_from_assign(day_to_week)
        scale_d = _interp_weekly_to_daily(scale_w, centers, r0.shape[0])
        r0 *= scale_d                                                # weekly RW scale (daily interp)
    r0 = _log_hann_smooth(r0, smooth_N)                              # same smoothing as Op (if enabled)
    return r0


# ---------- fast mode + disable mobility toggles (robust) ----------

def _disable_mobility_in_precomputed(pc: dict) -> None:
    """Force mobility off by zeroing relevant tensors in precomputed."""
    try:
        if "mobility_data" in pc and pc["mobility_data"] is not None:
            pc["mobility_data"] = np.zeros_like(np.asarray(pc["mobility_data"]))
        if "proportion_who_move" in pc and pc["proportion_who_move"] is not None:
            pc["proportion_who_move"] = np.zeros_like(np.asarray(pc["proportion_who_move"]))
        if "percent_day_away" in pc and pc["percent_day_away"] is not None:
            pc["percent_day_away"] = np.zeros_like(np.asarray(pc["percent_day_away"]))
    except Exception:
        pass


def _enable_fast_and_disable_mobility(pipe: WeeklyHospPipeline, op: WeeklyHospAndFinalSOp) -> None:
    """
    Best-effort enablement across versions:
      • Prefer constructor kwargs; also set internal fields directly for safety.
      • Always zero mobility arrays in precomputed as a fallback.
    """
    # Public flags (for visibility)
    for obj in (pipe, op, getattr(op, "_rhs", None)):
        try: setattr(obj, "fast_mode", True)
        except Exception: pass
        try: setattr(obj, "disable_mobility", True)
        except Exception: pass

    # Internal flags used by the Op implementation
    try: setattr(op, "_disable_mobility", True)
    except Exception: pass
    try:
        setattr(op, "_rtol", 5e-3)
        setattr(op, "_atol", 5e-5)
    except Exception:
        pass

    rhs = getattr(op, "_rhs", None)
    for name, val in (("rtol", 5e-3), ("atol", 5e-5)):
        try: setattr(rhs, name, val)
        except Exception: pass

    try:
        pc = getattr(pipe, "precomputed", {})
        if isinstance(pc, dict):
            _disable_mobility_in_precomputed(pc)
    except Exception:
        pass


# ============================= fixtures =============================

@pytest.fixture(scope="module")
def pipeline_and_op(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)

    # Prefer building with flags if supported
    try:
        pipe = build_pipeline_from_config(cfg_path, dt_days=0.5, fast_mode=True, disable_mobility=True)
    except TypeError:
        pipe = build_pipeline_from_config(cfg_path, dt_days=0.5)

    _safe_autotune()

    assert getattr(pipe, "seeding_on", False) is True, "Seeding should be ON for Structured_Example.yml"
    pc = getattr(pipe, "precomputed", {})
    for k in ("seeding_data", "seeding_amounts", "daily_incidence"):
        assert k in pc, f"Missing '{k}' in factory precomputed when seeding is ON"

    # Construct Op WITH desired flags so its internal tolerances/flags are set.
    op = WeeklyHospAndFinalSOp(pipe, fast_mode=True, disable_mobility=True)

    # Also enforce flags/zero-mobility defensively (works across versions)
    _enable_fast_and_disable_mobility(pipe, op)

    return pipe, op


# ============================= tests =============================

def _locate_csv_or_skip() -> Path:
    csv_env = os.environ.get("REALDATA_CSV") or "/Users/josh/Documents/test_data_alt.csv"
    csv_path = Path(csv_env)
    if not csv_path.exists():
        pytest.skip(f"REALDATA_CSV not found at {csv_path}; skipping real-data test.")
    return csv_path


@pytest.mark.slow
def test_seeding_is_active_and_op_runs_once(pipeline_and_op):
    """Smoke test + verify fast/no-mobility toggles are engaged."""
    pipe, op = pipeline_and_op
    A, W, L = op.weekly_shape
    defaults = np.ones(len(pipe.modifier_order()), dtype=np.float64)
    pR = np.full((A, L), 0.40, dtype=np.float64)   # 40% immune
    lambda_ext = np.ones(L, dtype=np.float64)

    # Assert effective flags (internal fields actually used in the Op)
    assert getattr(op, "_disable_mobility", False) is True
    assert np.isclose(getattr(op, "_rtol", 1e-3), 5e-3)
    assert np.isclose(getattr(op, "_atol", 1e-6), 5e-5)

    out_w = [None]
    out_S = [None]
    op.perform(None, [defaults, pR, lambda_ext], [out_w, out_S])
    weekly = np.asarray(out_w[0], dtype=np.float64)
    S_final = np.asarray(out_S[0], dtype=np.float64)

    assert weekly.shape == (A, W, L)
    assert S_final.shape == (A, L)
    assert np.isfinite(weekly).all()
    assert np.isfinite(S_final).all()


@pytest.mark.slow
def test_pymc_weekly_inference_all_states(pipeline_and_op):
    """
    End-to-end with fast mode and mobility disabled, using **all locations** present in the CSV.
    Prior predictive (light), then a small posterior sample, PPC, and panels.
    Also adds a 'seasonal prior-mean' r0 curve (monthly/holiday leaves at prior mean) in triads,
    and overlays *effective* r0 (after weekly RW2 scale interpolation + smoothing).
    """
    # ---------- quick knobs for a minimal-but-useful test ----------
    PRIOR_SAMPLES = int(os.environ.get("PRIOR_SAMPLES", "20"))      # spaghetti check
    TUNE = int(os.environ.get("TUNE", "3500"))                      # warmup
    DRAWS = int(os.environ.get("DRAWS", "3000"))                    # post-warmup per chain
    CHAINS = int(os.environ.get("CHAINS", "4"))
    CORES = min(CHAINS, max(1, os.cpu_count() or 1))
    # ---------------------------------------------------------------

    csv_path = _locate_csv_or_skip()
    pipe, op = pipeline_and_op

    # ---------------- Load and align CSV -> weekly matrix (W,L=ALL) ----------------
    y_full, obs_weeks, loc_names = _load_and_align_csv_to_weeks(
        csv_path, pipe, op, subset_sources=None
    )
    W, L = y_full.shape
    assert len(loc_names) == L
    assert L == op.locations, f"Data L={L} must match model L={op.locations}"
    if obs_weeks.size == 0:
        pytest.skip("No observed weeks after alignment; skipping.")

    outdir_env = os.environ.get("E2E_OUTDIR", "").strip()
    outdir = Path(outdir_env) if outdir_env else (Path.cwd() / "model_output_all_states")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- PRIOR PREDICTIVE (light) ----------------
    with build_weekly_model(pipe, op=op, y_obs=None, force_r0_weekly_scale=True) as prior_model:
        present = set(prior_model.named_vars.keys())
        # include mods_mu_log_loc if present
        requested = ["weekly_pred", "mods_loc", "mods_mu_log_loc", "r0_weekly_scale"]
        prior_vars = [v for v in requested if v in present]
        prior_idata = pm.sample_prior_predictive(
            samples=PRIOR_SAMPLES,
            random_seed=123,
            var_names=prior_vars,
        )

    weekly_vals = _idata_get(prior_idata, "weekly_pred")
    weekly = _stack_samples(weekly_vals)  # (S, A, W, L)
    S, A, Wm, Lm = weekly.shape
    assert Wm == W and Lm == L, "Weekly shape mismatch between Op and data alignment."

    # Spaghetti for ALL locations
    weeks = np.arange(W)
    for loc in range(L):
        series = weekly[:, :, :, loc].sum(axis=1)  # (S, W)
        var_per_week = series.var(axis=0)
        assert float(var_per_week.max()) > 0.0, "No variation in prior-predictive weekly trajectories."
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
        for s in range(min(PRIOR_SAMPLES, S)):
            ax.plot(weeks, series[s], alpha=0.25, linewidth=1.0)
        ax.set_title(f"Prior predictive — sum over ages, {loc_names[loc]}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Weekly hospitalizations (sum over age)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"prior_trajs_loc{loc:02d}_{loc_names[loc]}.png", bbox_inches="tight")
        plt.close(fig)

    # ---------------- Posterior fit (small) ----------------
    with build_weekly_model(pipe, op=op, y_obs=y_full, use_nb=True, force_r0_weekly_scale=True) as model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            step=pm.DEMetropolisZ(tune_interval=150,tune_drop_fraction=0.97),   # ← unblocked DEMZ
            random_seed=123,
            progressbar=True,
        )
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y", "weekly_pred_sum_age_shifted", "weekly_pred"],
            random_seed=123,
            progressbar=True,
        )

    idata.extend(ppc)

    # Save idata
    nc_path = outdir / "inference_idata.nc"
    az.to_netcdf(idata, nc_path)
    print("[artifact] saved:", nc_path)

    # Basic checks
    assert "weekly_pred" in idata.posterior_predictive
    assert "y" in idata.posterior_predictive  # <- ensure obs-level PPC is present

    summ = az.summary(idata, kind="stats", extend=True)
    assert np.isfinite(summ["mean"].values).all()

    # Trace (subset)
    try:
        # include mods_mu_log_loc if present
        trace_vars = ["r0_weekly_scale", "mods_mu_log_loc", "mods_loc",
                      "u_loc", "alpha_nb_loc", "delta_week", "age_scale", "lag_weights"]
        present_trace = [v for v in trace_vars if v in idata.posterior]
        if present_trace:
            az.plot_trace(idata, var_names=present_trace, compact=True, figsize=(14, 8))
            plt.tight_layout()
            plt.savefig(outdir / "trace_compact.png", bbox_inches="tight")
            plt.close()
    except Exception:
        pass

    # Panels for ALL locations (uses y when plotting against data)
    _panel_per_location(idata, y_full, obs_weeks, loc_names, pipe, outdir)
    (outdir / "summary.txt").write_text(summ.to_string())
    print("[artifact] wrote:", outdir / "summary.txt")
    print("[artifact] panels in:", outdir)

    # -------------- Three-panel triads: generate for ALL locations --------------
    L = len(loc_names)
    post = idata.posterior
    chains = post.dims["chain"]
    draws = post.dims["draw"]

    # choose two posterior samples to overlay
    rng = np.random.default_rng(20240831)
    flat_ix = rng.choice(chains * draws, size=2, replace=False)
    sample_pairs = [(ix // draws, ix % draws) for ix in flat_ix]

    # We will compute the *expectation* µ_t for each selected (chain, draw, loc)
    # and plot a 50% “noise” band using a normal approximation:
    #   Var_t = µ_t + µ_t^2 / alpha   (NB)   or Var_t = µ_t (Poisson)
    #   band = µ_t ± z * sqrt(Var_t),  z = 0.67448975 (≈ 50% central interval)

    # Grab needed posterior variables
    mu_shifted_name = "weekly_pred_sum_age_shifted"
    if mu_shifted_name not in post:
        # Deterministic should be saved in posterior; assert to surface issues
        raise AssertionError(f"'{mu_shifted_name}' not found in posterior.")

    mu_shifted = post[mu_shifted_name].values            # (chain, draw, W, L)
    beta0_loc = post["beta0_loc"].values                 # (chain, draw, L)
    delta_week = post["delta_week"].values               # (chain, draw, W, L)
    alpha_nb_loc = post["alpha_nb_loc"].values if "alpha_nb_loc" in post else None

    sr_mass0 = _precompute_sr_mass0(pipe)

    # r0-effective panel prep
    applier = getattr(pipe, "mod_applier", None)
    if applier is None:
        applier = compile_seir_modifiers(
            seir_modifiers_cfg=pipe.config["seir_modifiers"].get(),
            start_date=pipe.start_date,
            n_days=pipe.T,
            n_loc=pipe.NL,
            param_names=pipe.param_names,
        )

    # Locate r0 row
    param_names = np.array(list(pipe.param_defs.keys()))
    if "r0" in param_names:
        r0_idx = int(np.where(param_names == "r0")[0][0])
    else:
        aliases = ["R0", "r_0", "basic_reproduction_number"]
        found = [nm for nm in aliases if nm in param_names]
        assert found, "Could not locate 'r0' parameter row in base params."
        r0_idx = int(np.where(param_names == found[0])[0][0])

    base = pipe.base_params
    T_days = base.shape[1]
    t_days = np.arange(T_days)
    day_to_week = _mmwr_week_assign(pipe.start_date, pipe.T)

    mods_loc = post["mods_loc"].values
    mods = post["mods"].values if "mods" in post else None
    pR = post["pR"].values
    S_final = post["S_final"].values
    r0_weekly_scale_post = post["r0_weekly_scale"].values if "r0_weekly_scale" in post else None

    # Seasonal prior-mean leaf set (monthly/holiday)
    leaf_names = tuple(pipe.modifier_order())
    defaults = _yaml_defaults_in_leaf_order(pipe.config_path, leaf_names)
    mh_idx = [i for i, nm in enumerate(leaf_names) if ("month" in nm.lower()) or ("holi" in nm.lower())]

    prop_cycler = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    colors = [prop_cycler[i % len(prop_cycler)] for i in range(3)]
    # Distinct neutrals for baseline r0 and data:
    BASELINE_R0_COLOR = "0.5"   # gray
    DATA_COLOR = "0.1"          # near-black

    z50 = 0.67448975  # ~50% central interval for Normal

    for loc in range(L):
        loc_name = str(loc_names[loc])
        fig = plt.figure(figsize=(12, 9), dpi=130)
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.0, 1.2], hspace=0.28)

        # Panel 1: r0 baseline vs effective injected (after weekly interp + smoothing)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_days, base[r0_idx, :, loc], label="baseline r0", linewidth=1.6, color=BASELINE_R0_COLOR)
        s0_points, sT_points = [], []

        # Seasonal prior-mean line (only monthly/holiday leaves at prior mean; others = 1)
        if mh_idx:
            leaf_eff_prior = np.ones(len(leaf_names), dtype=float)
            leaf_eff_prior[mh_idx] = defaults[mh_idx]
            smooth_N = int(getattr(op, "_smooth_r0_days", 0) or 0)
            r0_seasonal = _effective_r0_daily(
                base, applier, leaf_eff_prior, r0_idx, loc,
                scale_w=None, day_to_week=day_to_week, smooth_N=smooth_N
            )
            ax1.plot(
                t_days, r0_seasonal, linestyle=":", linewidth=1.8, color="black",
                label="seasonal prior-mean (smoothed)"
            )

        smooth_N = int(getattr(op, "_smooth_r0_days", 0) or 0)
        for k, (c, d) in enumerate(sample_pairs):
            if mods is not None:
                mods_vec_sh = np.asarray(mods[c, d, :], dtype=float)
            else:
                mods_vec_sh = np.ones(mods_loc.shape[2], dtype=float)
            mods_loc_vec = np.asarray(mods_loc[c, d, :, loc], dtype=float)
            leaf_eff = mods_vec_sh * mods_loc_vec

            scale_w = None
            if r0_weekly_scale_post is not None:
                scale_w = np.array(r0_weekly_scale_post[c, d, :, loc], dtype=float)

            r0_eff = _effective_r0_daily(
                base, applier, leaf_eff, r0_idx, loc,
                scale_w=scale_w, day_to_week=day_to_week, smooth_N=smooth_N
            )
            ax1.plot(
                t_days, r0_eff, linestyle="--", linewidth=1.4, color=colors[k],
                label=f"sample {k+1} (eff r0)"
            )

            # Points for Panel 2
            pR_samp = pR[c, d, :, loc]
            S0_agg = np.sum((1.0 - pR_samp) * sr_mass0[:, loc]); s0_points.append(S0_agg)
            Sfinal_agg = np.sum(S_final[c, d, :, loc]); sT_points.append(Sfinal_agg)

        ax1.set_title(f"{loc_name} — r0 baseline vs effective injected (after RW2+smooth)")
        ax1.set_xlabel("day"); ax1.set_ylabel("r0(t)")
        ax1.grid(True, alpha=0.3); ax1.legend(loc="best")

        # Panel 2: S0 vs S(T)
        ax2 = fig.add_subplot(gs[1, 0])
        for k, (x, y) in enumerate(zip(s0_points, sT_points)):
            ax2.scatter([x], [y], s=36, color=colors[k], label=f"sample {k+1}")
        lo = min(s0_points + sT_points) * 0.95
        hi = max(s0_points + sT_points) * 1.05
        ax2.plot([lo, hi], [lo, hi], linewidth=1.0, alpha=0.4, color="0.3")
        ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
        ax2.set_xlabel("S0 (agg over age)"); ax2.set_ylabel("S(T) (agg over age)")
        ax2.set_title(f"{loc_name} — S0 vs S(T)"); ax2.grid(True, alpha=0.3); ax2.legend(loc="best")

        # Panel 3: Weekly hosp trajectories with 50% noise bands + target in distinct color
        ax3 = fig.add_subplot(gs[2, 0])
        w = np.arange(W)

        for k, (c, d) in enumerate(sample_pairs):
            # Expectation μ_t under obs model for this (chain, draw)
            mu_t = np.asarray(mu_shifted[c, d, :, loc], dtype=float) * np.exp(
                float(beta0_loc[c, d, loc])
            ) * np.exp(np.asarray(delta_week[c, d, :, loc], dtype=float))

            if alpha_nb_loc is not None:
                alpha_loc = float(alpha_nb_loc[c, d, loc])
                var_t = mu_t + (mu_t ** 2) / max(alpha_loc, 1e-12)
            else:
                var_t = mu_t  # Poisson fallback

            sd_t = np.sqrt(np.maximum(var_t, 1e-12))
            lo50 = np.clip(mu_t - 0.67448975 * sd_t, 0.0, np.inf)  # z for 50%
            hi50 = mu_t + 0.67448975 * sd_t

            ax3.fill_between(w, lo50, hi50, alpha=0.20, step="mid", color=colors[k],
                             label=f"sample {k+1} 50% band")
            ax3.plot(w, mu_t, linewidth=1.6, color=colors[k], label=f"sample {k+1} mean")

        _plot_weekly_targets(ax3, y_full[:, loc].astype(float), label="Target (data)", color=DATA_COLOR)
        ax3.set_xlabel("Week"); ax3.set_ylabel("Hosp (sum over age)")
        ax3.set_title(f"{loc_name} — Weekly hosp (samples + 50% noise band vs target)")
        ax3.grid(True, alpha=0.3); ax3.legend(loc="best")

        fig.suptitle(f"Three-panel summary — {loc_name}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        outfile = outdir / f"triad_loc{loc:02d}_{loc_name}.png"
        fig.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        print(f"[artifact] saved: {outfile}")
