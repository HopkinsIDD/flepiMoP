# Calibrate to real hospitalization data in a CSV using the Structured_Example config
# (with seeding), the WeeklyHospPipeline, and our Op. We:
# (1) align & aggregate the long CSV to MMWR weeks aligned to the model start,
# (2) run a PRIOR PREDICTIVE injection check & plots (before inference),
# (3) fit using the likelihood that is already defined inside build_weekly_model (use_nb=True),
# (4) save per-location posterior predictive panels (aggregated + by age),
# (5) three-panel figs (r0 baseline vs injected, S0 vs S(T) agg, hosp trajectories + targets),
# (6) SAVE full inference results to NetCDF with ArviZ.
#
# CSV must have columns: date, source, incidH
# Order: grouped/organized by location first, then by date (long format, daily totals).
#
# Set REALDATA_CSV=/path/to/your.csv to point at the data file. If missing, test is skipped.

import os
import platform
from ctypes.util import find_library

# Keep BLAS runtimes from fighting with Numba; set BEFORE heavy imports
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYMC_PROGRESSBAR", "1")

# Choose a numba threading layer early
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
# Regular imports (safe now)
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

from gempyor.vectorization_experiments import autotune_all, get_autotune_config
from gempyor.hosp_weekly_pipeline import build_pipeline_from_config, WeeklyHospPipeline
from gempyor.pymc_weekly_op import WeeklyHospAndFinalSOp, build_weekly_model

# Fallback if we need to compile an applier
from gempyor.vectorized_modifiers import compile_seir_modifiers


# ----------------------------- helpers ---------------------------------

def _materialize_structured_example(tmp_path_factory) -> Path:
    """Copy Structured_Example.yml & inputs into a temp root with absolute paths (seeding ON)."""
    tmp_root = tmp_path_factory.mktemp("weekly_infer_realdata")

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
    # cfg_name = "Structured_Example_No_Seeding.yml"
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
    """
    Map each day index (0..n_days-1) to an MMWR week index aligned to the model start.
    Mirrors the pipeline’s daily-week assignment:
      offset_to_sun = (6 - weekday) % 7; first bin length = 7 if Sunday else offset_to_sun.
    """
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


def _load_and_align_csv_to_weeks(csv_path: Path, pipe: WeeklyHospPipeline, op: WeeklyHospAndFinalSOp):
    """
    Read long CSV of *daily* total hospitalizations -> weekly aggregated matrix (W,L) aligned to model start_date.
    CSV columns expected: 'date', 'source', 'incidH'.
    Steps:
      • coerce 'incidH' numeric and nonnegative
      • collapse duplicates per (source, date) by summing  -> daily totals (all ages already)
      • restrict to model horizon and compute day_index
      • map days to MMWR weeks aligned to model start
      • aggregate to weekly totals per location
      • pivot to (W, L) with weeks [0..op.n_weeks-1]
      • preserve **first-appearance** order of 'source' (no sorting)
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"date", "source", "incidH"}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.copy()
    df["incidH"] = pd.to_numeric(df["incidH"], errors="coerce").fillna(0.0).clip(lower=0.0)

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

    # Preserve first-appearance order of locations in the (aligned) data
    loc_names = tuple(pd.unique(df["source"]))
    L = op.locations
    if len(loc_names) != L:
        raise AssertionError(
            f"Location count mismatch: data has {len(loc_names)} 'source' values vs model L={L}."
        )

    # Weekly totals per location
    weekly = (
        df.groupby(["week_idx", "source"], sort=False)["incidH"]
          .sum()
          .reset_index()
    )

    # Pivot to (W, L) using the Op’s week count
    W = op.n_weeks
    pivot = (
        weekly.pivot(index="week_idx", columns="source", values="incidH")
              .reindex(range(W))
              .sort_index()
              .reindex(columns=list(loc_names))
    )

    # Observed weeks: any location has data
    obs_weeks = np.flatnonzero(pivot.notna().any(axis=1).values)

    # IMPORTANT: for inference inside the Op, fill missing with 0.0 (PyMC cannot take NaN in observed).
    # We still pass obs_weeks separately for plotting overlays/diagnostics.
    y_full = pivot.fillna(0.0).to_numpy(dtype=float)  # (W, L)

    return y_full, obs_weeks, loc_names


# NEW: plot helper to overlay targets with gaps handled
def _plot_weekly_targets(ax: plt.Axes, y_vec: np.ndarray, *, label: str = "Target (data)"):
    """
    Plot weekly targets (possibly with zeros standing in for missing weeks) as step curves
    over the full horizon. If you want to visualize only truly observed weeks, call this
    with y_vec masked outside obs_weeks before plotting.
    """
    import numpy as _np
    W = y_vec.shape[0]
    weeks = _np.arange(W)
    ax.step(weeks, y_vec, where="mid", linewidth=1.4, alpha=0.85, label=label)


import re

def _age_lower_bound(label: str) -> int:
    """
    Extract the lower-edge integer from an age label.
    Handles forms like: 'age0to4', '0-4', '5–17', '65+', '85p', 'age18to49', 'age5to17+'.
    Falls back to 0 if no digits are found.
    """
    s = str(label).lower().replace("age", "").replace("to", "-").replace("–", "-").replace("—", "-")
    s = s.replace("plus", "+").replace("p", "+")  # treat trailing 'p' as plus
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def _panel_per_location(idata, y_obs_full, obs_weeks, loc_names, pipe, outdir: Path):
    """
    For each location:
      - Top: aggregated posterior predictive (mean + 95% HDI) vs observed (on observed weeks only)
      - Bottom: one subplot per unique age group (mean + 95% HDI) in **ascending age-bin order**.
    """
    ages = tuple(pipe.age_labels)  # labels from config
    A = len(ages)
    W = y_obs_full.shape[0]

    # --- ORDER ages by lower bound (e.g., 0, 5, 18, 50, 65, ...)
    order = np.argsort([_age_lower_bound(a) for a in ages])
    ages_sorted = [ages[i] for i in order]

    agg_ppc = idata.posterior_predictive["weekly_pred_sum_age"].values   # (chain, draw, W, L)
    age_ppc = idata.posterior_predictive["weekly_pred"].values           # (chain, draw, A, W, L)

    weeks = np.arange(W)

    for loc_idx, loc_name in enumerate(loc_names):
        fig_width = min(30, max(14, 3.2 * A))
        fig = plt.figure(figsize=(fig_width, 7.5), dpi=120)

        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.3, 1.0], hspace=0.35)
        ax_agg = fig.add_subplot(gs[0, 0])
        bottom = gs[1].subgridspec(1, A, wspace=0.25)

        # ---------- Aggregated (observed overlay on observed weeks only)
        if obs_weeks.size > 0:
            samples = agg_ppc[:, :, obs_weeks, loc_idx]           # (chain, draw, W_obs)
            mean = samples.mean(axis=(0, 1))
            hdi = az.hdi(samples, hdi_prob=0.95)                  # (W_obs, 2)

            ax_agg.fill_between(obs_weeks, hdi[:, 0], hdi[:, 1], alpha=0.25, step="mid", label="95% HDI")
            ax_agg.plot(obs_weeks, mean, linewidth=1.5, label="Posterior mean")
            y_loc = y_obs_full[obs_weeks, loc_idx]
            ax_agg.step(obs_weeks, y_loc, where="mid", linewidth=1.2, label="Observed")
        ax_agg.set_title(f"{loc_name} — Aggregated hospitalizations (all ages)")
        ax_agg.set_xlabel("Week")
        ax_agg.set_ylabel("Hosp")
        ax_agg.grid(True, alpha=0.3)
        ax_agg.legend(loc="upper right")

        # ---------- Per-age row (sorted by lower bound)
        for j, a_idx in enumerate(order):
            ax = fig.add_subplot(bottom[0, j], sharex=None if j == 0 else fig.axes[-1])

            samples_a = age_ppc[:, :, a_idx, :, loc_idx]          # (chain, draw, W)
            mean_a = samples_a.mean(axis=(0, 1))
            hdi_a = az.hdi(samples_a, hdi_prob=0.95)              # (W, 2)

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




# ---------- prior predictive helpers (PyMC5: prior vs prior_predictive) ----------

def _idata_get(idata, var_name: str):
    """Fetch a variable from either `prior_predictive` or `prior` InferenceData group."""
    for grp in ("prior_predictive", "prior"):
        grp_obj = getattr(idata, grp, None)
        if grp_obj is not None and var_name in grp_obj:
            return grp_obj[var_name].values
    raise AssertionError(f"Variable '{var_name}' not found in prior/prior_predictive groups.")

def _stack_samples(arr: np.ndarray) -> np.ndarray:
    """
    Accept weekly_pred array shaped (draw, A, W, L) or (chain, draw, A, W, L)
    and return (S, A, W, L) where S = total samples.
    """
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5:
        c, d, A, W, L = arr.shape
        return arr.reshape(c * d, A, W, L)
    raise ValueError(f"Unexpected ndim for weekly_pred: {arr.shape}")


# ------------------------- NEW: S0 helper (aggregated) -------------------

def _build_age_masks(pipe: WeeklyHospPipeline):
    """Return (is_S_per_age, is_R_per_age, is_total_per_age) boolean masks over compartments."""
    df = pipe.model.compartments.compartments  # pandas DataFrame
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
    """
    Compute sr_mass0[a, l] = (total mass in age a, loc l) minus (non S/R mass) at t0.
    """
    initial = pipe.initial_array  # (NC, L)
    s_masks, r_masks, tot_masks, ages = _build_age_masks(pipe)
    L = pipe.NL
    A = len(ages)
    sr = np.zeros((A, L), dtype=np.float64)
    for a, (m_s, m_r, m_tot) in enumerate(zip(s_masks, r_masks, tot_masks)):
        m_other = m_tot & (~m_s) & (~m_r)
        N_tot = initial[m_tot, :].sum(axis=0)
        N_other = initial[m_other, :].sum(axis=0)
        sr[a, :] = np.maximum(N_tot - N_other, 0.0)
    return sr  # (A, L)


# ----------------------------- fixtures ---------------------------------

@pytest.fixture(scope="module")
def pipeline_and_op(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)
    pipe = build_pipeline_from_config(cfg_path)

    _safe_autotune()

    assert getattr(pipe, "seeding_on", False) is True, "Seeding should be ON for Structured_Example.yml"
    pc = getattr(pipe, "precomputed", {})
    for k in ("seeding_data", "seeding_amounts", "daily_incidence"):
        assert k in pc, f"Missing '{k}' in factory precomputed when seeding is ON"

    op = WeeklyHospAndFinalSOp(pipe)
    return pipe, op


# ------------------------------- tests -----------------------------------

def _locate_csv_or_skip() -> Path:
    csv_env = os.environ.get("REALDATA_CSV") or "/Users/josh/Documents/test_data_alt.csv"
    csv_path = Path(csv_env)
    if not csv_path.exists():
        pytest.skip(f"REALDATA_CSV not found at {csv_path}; skipping real-data test.")
    return csv_path


@pytest.mark.slow
def test_seeding_is_active_and_op_runs_once(pipeline_and_op):
    """
    Smoke test:
      • confirms seeding is ON at the pipeline level and present in precomputed
      • runs the Op forward once to make sure nothing crashes
    """
    pipe, op = pipeline_and_op
    A, W, L = op.weekly_shape
    defaults = np.ones(len(pipe.modifier_order()), dtype=np.float64)
    pR = np.full((A, L), 0.40, dtype=np.float64)   # 40% immune
    lambda_ext = np.ones(L, dtype=np.float64)

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
def test_pymc_weekly_inference_with_real_csv(pipeline_and_op):
    """
    Pipeline on real CSV:
      1) Load CSV (daily) → weekly (W,L) aligned to model.
      2) PRIOR PREDICTIVE (50 draws) — verify injection & save spaghetti plots.
      3) Fit using the likelihood defined in the Op/model builder (Negative Binomial via use_nb=True).
      4) Posterior predictive panels + summary.
      5) Three-panel figs per 4 random locations (r0 baseline vs injected; S0 vs S(T); hosp trjs + target).
      6) SAVE full InferenceData (posterior + PPC) to NetCDF via ArviZ.
    """
    csv_path = _locate_csv_or_skip()

    pipe, op = pipeline_and_op

    # ---------------- Load and align CSV -> weekly matrix (W,L) ----------------
    y_full, obs_weeks, loc_names = _load_and_align_csv_to_weeks(csv_path, pipe, op)
    W, L = y_full.shape
    assert L == op.locations, f"Data L={L} must match model L={op.locations}"
    if obs_weeks.size == 0:
        pytest.skip("No observed weeks after alignment; skipping.")

    # Output directory
    outdir_env = os.environ.get("E2E_OUTDIR", "").strip()
    outdir = Path(outdir_env) if outdir_env else (Path.cwd() / "model_output")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- PRIOR PREDICTIVE INJECTION CHECK (before inference) ----------------
    # Ask only for variables that actually exist in the model (r0_weekly_scale is optional).
    with build_weekly_model(pipe, op=op, y_obs=None) as prior_model:
        present = set(prior_model.named_vars.keys())
        requested = ["weekly_pred", "mods_loc", "r0_weekly_scale"]
        prior_vars = [v for v in requested if v in present]

        prior_idata = pm.sample_prior_predictive(
            samples=50,
            random_seed=123,
            var_names=prior_vars,
        )

    weekly_vals = _idata_get(prior_idata, "weekly_pred")
    weekly = _stack_samples(weekly_vals)  # (S, A, W, L)

    S, A, Wm, Lm = weekly.shape
    assert Wm == W and Lm == L, "Weekly shape mismatch between Op and data alignment."

    # Numeric variance check to ensure injection happened
    locs_to_plot = [0, min(L - 1, max(0, L // 2))]
    weeks = np.arange(W)
    for loc in locs_to_plot:
        series = weekly[:, :, :, loc].sum(axis=1)  # (S, W)
        var_per_week = series.var(axis=0)
        assert float(var_per_week.max()) > 0.0, "No variation in prior-predictive weekly trajectories."

        # Spaghetti plots per selected location (sum over ages)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
        for s in range(min(50, S)):
            ax.plot(weeks, series[s], alpha=0.25, linewidth=1.0)
        ax.set_title(f"Prior predictive trajectories — sum over ages, location {loc}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Weekly hospitalizations (sum over age)")
        ax.grid(True, alpha=0.3)
        outpng = outdir / f"prior_trajs_loc{loc}.png"
        fig.tight_layout()
        fig.savefig(outpng, bbox_inches="tight")
        plt.close(fig)

    # ---------------- Build model WITH likelihood in the Op (Negative Binomial) ----------------
    # NOTE: We pass the *full* (W,L) matrix with zeros for missing weeks so PyMC can accept it.
    with build_weekly_model(pipe, op=op, y_obs=y_full, use_nb=True) as model:
        # Sampling
        ncores = min(4, os.cpu_count() or 1)
        idata = pm.sample(
            draws=2000,
            tune=5000,
            chains=ncores,
            cores=ncores,
            step=pm.DEMetropolisZ(),
            random_seed=123,
            progressbar=True,
        )

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y", "weekly_pred_sum_age", "weekly_pred"],
            random_seed=123,
            progressbar=True,
        )

    # Merge for ArviZ convenience
    idata.extend(ppc)

    # ---------------- SAVE full InferenceData to NetCDF ----------------
    nc_path = outdir / "inference_idata.nc"
    az.to_netcdf(idata, nc_path)
    print("[artifact] saved:", nc_path)

    # ---------------- Basic sanity assertions ----------------
    assert "weekly_pred_sum_age" in idata.posterior_predictive
    assert "weekly_pred" in idata.posterior_predictive

    summ = az.summary(
        idata,
        kind="stats",
        extend=True,
    )
    assert np.isfinite(summ["mean"].values).all()

    # ---------------- Save artifacts ----------------
    try:
        # If r0_weekly_scale exists in posterior, include it in the trace as well.
        trace_vars = ["mods_loc", "u_loc", "alpha_nb_loc"]
        if "r0_weekly_scale" in idata.posterior:
            trace_vars.insert(0, "r0_weekly_scale")
        az.plot_trace(
            idata,
            var_names=[v for v in trace_vars if v in idata.posterior],
            compact=True,
            figsize=(12, 6),
        )
        plt.tight_layout()
        plt.savefig(outdir / "trace_compact.png", bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    _panel_per_location(idata, y_full, obs_weeks, loc_names, pipe, outdir)

    (outdir / "summary.txt").write_text(summ.to_string())
    print("[artifact] wrote:", outdir / "summary.txt")
    print("[artifact] panels in:", outdir)

    # -------------------------- Three-panel figures --------------------------
    rng = np.random.default_rng(20240831)
    L = len(loc_names)
    pick_locs = rng.choice(L, size=min(4, L), replace=False)

    post = idata.posterior
    chains = post.dims["chain"]
    draws = post.dims["draw"]
    flat_ix = rng.choice(chains * draws, size=3, replace=False)
    sample_pairs = [(ix // draws, ix % draws) for ix in flat_ix]

    prop_cycler = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    colors = [prop_cycler[i % len(prop_cycler)] for i in range(3)]

    # Model parameters
    mods = post["mods"].values if "mods" in post else None          # (chain, draw, M) — ones, not plotted
    mods_loc = post["mods_loc"].values                               # (chain, draw, M, L)
    pR = post["pR"].values                                           # (chain, draw, A, L)
    S_final = post["S_final"].values                                 # (chain, draw, A, L)
    r0_weekly_scale_post = post["r0_weekly_scale"].values if "r0_weekly_scale" in post else None  # (chain,draw,W,L) or None

    wsum_ppc = idata.posterior_predictive["weekly_pred_sum_age"].values  # (chain, draw, W, L)

    sr_mass0 = _precompute_sr_mass0(pipe)  # (A, L)

    applier = getattr(pipe, "mod_applier", None)
    if applier is None:
        applier = compile_seir_modifiers(
            seir_modifiers_cfg=pipe.config["seir_modifiers"].get(),
            start_date=pipe.start_date,
            n_days=pipe.T,
            n_loc=pipe.NL,
            param_names=pipe.param_names,
        )

    param_names = np.array(list(pipe.param_defs.keys()))
    if "r0" in param_names:
        r0_idx = int(np.where(param_names == "r0")[0][0])
    else:
        aliases = ["R0", "r_0", "basic_reproduction_number"]
        found = [nm for nm in aliases if nm in param_names]
        assert found, "Could not locate 'r0' parameter row in base params."
        r0_idx = int(np.where(param_names == found[0])[0][0])

    base = pipe.base_params  # (P_base, T, L)
    T_days = base.shape[1]
    t_days = np.arange(T_days)

    # Day→week mapping for optional weekly r0 scaling visualization
    day_to_week = _mmwr_week_assign(pipe.start_date, pipe.T)

    for loc in pick_locs:
        loc_name = str(loc_names[loc])

        fig = plt.figure(figsize=(12, 10), dpi=130)
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.0, 1.2], hspace=0.28)

        # Panel (1): r0 baseline vs injected for 3 samples
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t_days, base[r0_idx, :, loc], label="baseline r0", linewidth=1.6)
        s0_points = []
        sT_points = []

        for k, (c, d) in enumerate(sample_pairs):
            # modifiers (still applied, even if many leaves are neutral)
            if mods is not None:
                mods_vec = mods[c, d, :]                      # (M,)
            else:
                # If 'mods' not stored, use ones of appropriate length from mods_loc
                mods_vec = np.ones(mods_loc.shape[2], dtype=float)
            mods_loc_vec = mods_loc[c, d, :, loc]             # (M,)
            leaf_eff = mods_vec * mods_loc_vec                # (M,)

            injected = applier.apply_to_params(base, leaf_value_array=leaf_eff, scenario="none")
            r0_injected = injected[r0_idx, :, loc].copy()

            # If weekly r0 scaling exists, include it in the injected line for fairness
            if r0_weekly_scale_post is not None:
                scale_w = r0_weekly_scale_post[c, d, :, loc]  # (W,)
                for day_idx in range(T_days):
                    w = int(day_to_week[day_idx])
                    r0_injected[day_idx] *= float(scale_w[w])

            ax1.plot(t_days, r0_injected, linestyle="--", linewidth=1.4,
                     color=colors[k], label=f"sample {k+1}")

            pR_samp = pR[c, d, :, loc]                    # (A,)
            S0_agg = np.sum((1.0 - pR_samp) * sr_mass0[:, loc])
            s0_points.append(S0_agg)

            Sfinal_agg = np.sum(S_final[c, d, :, loc])
            sT_points.append(Sfinal_agg)

        ax1.set_title(f"{loc_name} — r0 baseline vs injected (3 posterior samples)")
        ax1.set_xlabel("day")
        ax1.set_ylabel("r0(t)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        # Panel (2): S0 (agg) vs end-of-season S (agg)
        ax2 = fig.add_subplot(gs[1, 0])
        for k, (x, y) in enumerate(zip(s0_points, sT_points)):
            ax2.scatter([x], [y], s=36, color=colors[k], label=f"sample {k+1}")
        lo = min(s0_points + sT_points) * 0.95
        hi = max(s0_points + sT_points) * 1.05
        ax2.plot([lo, hi], [lo, hi], linewidth=1.0, alpha=0.4)
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.set_xlabel("S0 (aggregated over age)")
        ax2.set_ylabel("S(T) (aggregated over age)")
        ax2.set_title(f"{loc_name} — S0 vs end-of-season S (agg over age)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

        # Panel (3): Weekly hospitalization trajectories (3 samples) + target overlay
        ax3 = fig.add_subplot(gs[2, 0])
        w = np.arange(W)
        for k, (c, d) in enumerate(sample_pairs):
            traj = wsum_ppc[c, d, :, loc]  # (W,)
            ax3.plot(w, traj, linewidth=1.6, color=colors[k], label=f"sample {k+1}")

        # Overlay target (age-aggregated weekly incidence) for the full horizon (zeros where missing)
        _plot_weekly_targets(ax3, y_full[:, loc].astype(float), label="Target (data)")

        ax3.set_xlabel("Week")
        ax3.set_ylabel("Hosp (sum over age)")
        ax3.set_title(f"{loc_name} — Hospitalization trajectories (3 posterior samples)")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="best")

        fig.suptitle(f"Three-panel summary — {loc_name}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        outfile = outdir / f"triad_loc{loc:02d}_{loc_name}.png"
        fig.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
        print(f"[artifact] saved: {outfile}")
