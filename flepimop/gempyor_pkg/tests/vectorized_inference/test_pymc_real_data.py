# Calibrate to real hospitalization data in a CSV using the Structured_Example config
# (with seeding), the WeeklyHospPipeline, and our Op. We:
# (1) align & aggregate the long CSV to MMWR weeks aligned to the model start,
# (2) run a PRIOR PREDICTIVE injection check & plots (before inference),
# (3) calibrate only on observed weeks (subset-of-weeks allowed),
# (4) save per-location posterior predictive panels (aggregated + by age).
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

    # Config with seeding support
    # cfg_name = "Structured_Example_Seeding_Alt.yml"
    cfg_name = "Structured_Example_Seeding_Alt.yml"
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
      • pivot to (week, location) with weeks [0..op.n_weeks-1]
      • preserve **first-appearance** order of 'source' (no sorting)
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"date", "source", "incidH"}
    if not required.issubset(df.columns):
        # pandas.Series doesn't have issubset; fix:
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
    y_full = pivot.fillna(np.nan).to_numpy(dtype=float)  # (W, L)
    return y_full, obs_weeks, loc_names


def _panel_per_location(idata, y_obs_full, obs_weeks, loc_names, pipe, outdir: Path):
    """
    For each location:
      - Top: aggregated posterior predictive (mean + 95% HDI) vs observed (on observed weeks only)
      - Bottom: one subplot per unique age group (mean + 95% HDI), in a single row
    """
    ages = tuple(pipe.age_labels)  # preserve order from config
    A = len(ages)
    W = y_obs_full.shape[0]

    agg_ppc = idata.posterior_predictive["weekly_pred_sum_age"].values   # (chain, draw, W, L)
    age_ppc = idata.posterior_predictive["weekly_pred"].values           # (chain, draw, A, W, L)

    weeks = np.arange(W)

    for loc_idx, loc_name in enumerate(loc_names):
        # Auto width so all ages fit on one row; cap to avoid comically huge figures
        fig_width = min(30, max(14, 3.2 * A))
        fig = plt.figure(figsize=(fig_width, 7.5), dpi=120)

        # 2-row layout: top agg, bottom 1xA subplots
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.3, 1.0], hspace=0.35)
        ax_agg = fig.add_subplot(gs[0, 0])
        bottom = gs[1].subgridspec(1, A, wspace=0.25)

        # ----- Aggregated (observed overlay on observed weeks only)
        if obs_weeks.size > 0:
            samples = agg_ppc[:, :, obs_weeks, loc_idx]   # (chain, draw, W_obs)
            mean = samples.mean(axis=(0, 1))
            hdi = az.hdi(samples, hdi_prob=0.95)          # (W_obs, 2)

            ax_agg.fill_between(obs_weeks, hdi[:, 0], hdi[:, 1], alpha=0.25, step="mid", label="95% HDI")
            ax_agg.plot(obs_weeks, mean, linewidth=1.5, label="Posterior mean")
            y_loc = y_obs_full[obs_weeks, loc_idx]
            ax_agg.step(obs_weeks, y_loc, where="mid", linewidth=1.2, label="Observed")
        ax_agg.set_title(f"{loc_name} — Aggregated hospitalizations (all ages)")
        ax_agg.set_xlabel("MMWR week")
        ax_agg.set_ylabel("Hosp")
        ax_agg.grid(True, alpha=0.3)
        ax_agg.legend(loc="upper right")

        # ----- Per-age row (one subplot per age)
        for a_idx in range(A):
            ax = fig.add_subplot(bottom[0, a_idx], sharex=None if a_idx == 0 else fig.axes[-1])

            samples_a = age_ppc[:, :, a_idx, :, loc_idx]  # (chain, draw, W)
            mean_a = samples_a.mean(axis=(0, 1))
            hdi_a = az.hdi(samples_a, hdi_prob=0.95)      # (W, 2)

            ax.fill_between(weeks, hdi_a[:, 0], hdi_a[:, 1], alpha=0.20, step="mid")
            ax.plot(weeks, mean_a, linewidth=1.2)
            ax.set_title(str(ages[a_idx]))
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("MMWR week")
            # Y label only on the first age panel to save space
            if a_idx == 0:
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


# ----------------------------- fixtures ---------------------------------

@pytest.fixture(scope="module")
def pipeline_and_op(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)
    pipe = build_pipeline_from_config(cfg_path)

    # hardware optimization
    _safe_autotune()

    # Assert seeding is wired in (this config uses seeding)
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
    # pR baseline: 40% immune, uniform (shape A×L)
    pR = np.full((A, L), 0.40, dtype=np.float64)
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
      2) PRIOR PREDICTIVE (50 draws) — verify modifier injection & save spaghetti plots.
      3) Build likelihood on observed weeks and run a short posterior.
      4) Posterior predictive panels + summary.
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
    with build_weekly_model(pipe, op=op, y_obs=None) as prior_model:
        # PyMC 5: use samples= and pull from prior/prior_predictive
        prior_idata = pm.sample_prior_predictive(samples=50, random_seed=123,
                                                 var_names=["weekly_pred", "mods"])

    weekly_vals = _idata_get(prior_idata, "weekly_pred")
    weekly = _stack_samples(weekly_vals)  # (S, A, W, L)
    mods_vals = _idata_get(prior_idata, "mods")
    if mods_vals.ndim == 2:
        mods_samples = mods_vals  # (S, M)
    else:
        c, d, M = mods_vals.shape
        mods_samples = mods_vals.reshape(c * d, M)

    S, A, Wm, Lm = weekly.shape
    assert Wm == W and Lm == L, "Weekly shape mismatch between Op and data alignment."

    # Numeric variance check to ensure injection happened
    locs_to_plot = [0, min(L - 1, max(0, L // 2))]
    for loc in locs_to_plot:
        series = weekly[:, :, :, loc].sum(axis=1)  # (S, W)
        var_per_week = series.var(axis=0)
        assert float(var_per_week.max()) > 0.0, (
            "No variation in prior-predictive weekly trajectories — modifiers may not be injected."
        )

    # Spaghetti plots per selected location (sum over ages)
    weeks = np.arange(W)
    for loc in locs_to_plot:
        series = weekly[:, :, :, loc].sum(axis=1)  # (S, W)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
        for s in range(min(50, S)):
            ax.plot(weeks, series[s], alpha=0.25, linewidth=1.0)
        ax.set_title(f"Prior predictive trajectories — sum over ages, location {loc}")
        ax.set_xlabel("MMWR week")
        ax.set_ylabel("Weekly hospitalizations (sum over age)")
        ax.grid(True, alpha=0.3)
        outpng = outdir / f"prior_trajs_loc{loc}.png"
        fig.tight_layout()
        fig.savefig(outpng, bbox_inches="tight")
        plt.close(fig)

    # Quick modifiers scatter (first ~12)
    M_show = min(12, mods_samples.shape[1])
    fig, axes = plt.subplots(M_show, 1, figsize=(10, 1.6 * M_show), dpi=120, sharex=True)
    if M_show == 1:
        axes = [axes]
    for i in range(M_show):
        ax = axes[i]
        ax.plot(mods_samples[:min(50, mods_samples.shape[0]), i], ".", alpha=0.7, markersize=4)
        ax.set_ylabel(f"mod[{i}]")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("sample index (prior)")
    fig.suptitle("Modifier samples (subset) — should vary under prior", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "prior_modifiers_subset.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- Build model WITH likelihood on observed weeks ----------------
    with build_weekly_model(pipe, op=op, y_obs=None) as model:
        weekly_sum = model["weekly_pred_sum_age"]  # dims: (week, location)

        # Slice to observed weeks for likelihood
        mu_obs = weekly_sum[obs_weeks, :]

        # Poisson likelihood on observed weeks only (NB is easy to swap back in)
        pm.Poisson("y", mu=mu_obs, observed=y_full[obs_weeks, :].astype(np.float64))

        # --------------- Sampling (moderate for CI/runtime) ---------------
        # make sure autotune is in effect (already called in fixture)
        ncores = min(4, os.cpu_count() or 1)
        idata = pm.sample(
            draws=2000,
            tune=3000,
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

    # ---------------- Basic sanity assertions ----------------
    assert "weekly_pred_sum_age" in idata.posterior_predictive
    assert "weekly_pred" in idata.posterior_predictive

    # Match current model variable names (no legacy sigma_age/sigma_loc)
    summ = az.summary(
        idata,
        var_names=[
            "mods",
            "mu_R_logit",
            "sigma_R_loc",
            "lambda_ext_mu_log",
            "lambda_ext_sigma_log",
            "obs_loc_scale",
        ],
        kind="stats",
        extend=True,
    )
    assert np.isfinite(summ["mean"].values).all()

    # ---------------- Save artifacts ----------------
    # compact trace plot (subset of variables)
    try:
        az.plot_trace(
            idata,
            var_names=["mods", "lambda_ext_loc", "obs_loc_scale"],
            compact=True,
            figsize=(12, 6),
        )
        plt.tight_layout()
        plt.savefig(outdir / "trace_compact.png", bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # Per-location panels (aggregated vs age-strat)
    _panel_per_location(idata, y_full, obs_weeks, loc_names, pipe, outdir)

    # Persist a small text summary
    (outdir / "summary.txt").write_text(summ.to_string())
    print("[artifact] wrote:", outdir / "summary.txt")
    print("[artifact] panels in:", outdir)
