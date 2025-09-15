# Fast-mode + no-mobility calibration that LOOPS over each state present in the
# empirical calibration CSV and writes ONE NetCDF PER STATE (single-location traces),
# using the FOURIER r0(t) prior/scale path.
#
# Expected artifacts in a single directory (defaults to ./model_output_all_states/):
#   - inference_idata_<STATE>.nc            (one per state)
#   - prior_spaghetti_<STATE>.png           (for 4 random states)
#   - triad_<STATE>.png                     (for 4 random states)
#   - ppc_panel_00_<STATE>.png              (for ALL states)

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
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm
import re
import confuse

from gempyor.vectorization_experiments import autotune_all, get_autotune_config
from gempyor.hosp_weekly_pipeline import build_pipeline_from_config, WeeklyHospPipeline
from gempyor.pymc_weekly_op import (
    WeeklyHospAndFinalSOp,
    build_weekly_model,
    _yaml_defaults_in_leaf_order,
)
from gempyor.vectorized_modifiers import compile_seir_modifiers

# ModelInfo for discovering permissible codes / structure
from gempyor.model_info import ModelInfo  # adjust path if your repo layout differs


# ============================= helpers =============================

def _materialize_structured_example(tmp_path_factory) -> Path:
    """Copy Structured_Example.yml & inputs into a temp root with absolute paths (seeding ON)."""
    tmp_root = tmp_path_factory.mktemp("weekly_infer_realdata_all_states_fourier")

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
    """Run autotune once; keep threads within NUMBA_NUM_THREADS if needed."""
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
    CSV columns expected: 'date', 'source', 'incidH'. `source` are state names.
    If `subset_sources` is provided, only those sources are considered and columns are ordered exactly as given.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"date", "source", "incidH"}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.copy()
    df["incidH"] = pd.to_numeric(df["incidH"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if subset_sources:
        df = df[df["source"].isin(subset_sources)]
        if df.empty:
            raise ValueError(f"No rows left after filtering to {subset_sources}")
        df["source"] = pd.Categorical(df["source"], categories=list(subset_sources), ordered=True)
        loc_names = tuple(subset_sources)
    else:
        loc_names = None

    # Daily totals per (state_name, date)
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

    # Convert category back to string
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
    For each location (here L=1):
      - Top: aggregated posterior predictive of y (mean + 95% HDI) vs observed.
      - Bottom: one subplot per age group (mean + 95% HDI) in ascending age-bin order.
    """
    ages = tuple(pipe.age_labels)
    A = len(ages)
    W = y_obs_full.shape[0]

    order = np.argsort([_age_lower_bound(a) for a in ages])
    ages_sorted = [ages[i] for i in order]

    # Prefer observation-level predictive if present
    if "y" in idata.posterior_predictive:
        agg_ppc = idata.posterior_predictive["y"].values  # (chain, draw, W, L)
        mean_label = "Posterior mean (obs model)"
    else:
        if "weekly_pred_sum_age_shifted" in idata.posterior_predictive:
            agg_ppc = idata.posterior_predictive["weekly_pred_sum_age_shifted"].values
        else:
            agg_ppc = idata.posterior_predictive["weekly_pred_sum_age"].values
        mean_label = "Posterior mean (process)"

    age_ppc = idata.posterior_predictive["weekly_pred"].values  # (chain, draw, A, W, L)
    weeks = np.arange(W)

    L = y_obs_full.shape[1]
    for loc_idx in range(L):
        loc_name = str(loc_names[loc_idx])
        fig_width = min(30, max(14, 3.2 * A))
        fig = plt.figure(figsize=(fig_width, 7.5), dpi=120)

        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.3, 1.0], hspace=0.35)
        ax_agg = fig.add_subplot(gs[0, 0])
        bottom = gs[1].subgridspec(1, A, wspace=0.25)

        # Aggregated
        samples = agg_ppc[:, :, :, loc_idx]            # (chain, draw, W)
        mean = samples.mean(axis=(0, 1))               # (W,)
        hdi = az.hdi(samples, hdi_prob=0.95)           # (W, 2)
        ax_agg.fill_between(weeks, hdi[:, 0], hdi[:, 1], alpha=0.25, step="mid", label="95% HDI")
        ax_agg.plot(weeks, mean, linewidth=1.5, label=mean_label)
        y_loc = y_obs_full[:, loc_idx]
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


# ---------- r0-effective helpers (match Op transforms) ----------

def _week_centers_from_assign(assign: np.ndarray) -> np.ndarray:
    if assign.size == 0:
        return np.zeros(0, dtype=float)
    W = int(assign.max()) + 1
    centers = np.zeros(W, dtype=float)
    for w in range(W):
        idx = np.flatnonzero(assign == w)
        centers[w] = 0.5 * (float(idx[0]) + float(idx[-1])) if idx.size else ((centers[w - 1] + 7.0) if w > 0 else 0.0)
    return centers

def _interp_weekly_to_daily(scale_w: np.ndarray, centers: np.ndarray, T_days: int) -> np.ndarray:
    """
    Interpolate a weekly scale (length W) to daily. If centers length != W
    (e.g., off-by-one due to horizon alignment), truncate/pad centers to match W.
    """
    t = np.arange(T_days, dtype=float)
    W = len(scale_w)
    if len(centers) != W:
        if len(centers) > W:
            centers = centers[:W]
        else:
            # Pad centers forward with ~weekly spacing
            if len(centers) >= 2:
                step = centers[-1] - centers[-2]
                if not np.isfinite(step) or abs(step) < 1e-9:
                    step = 7.0
            else:
                step = 7.0
            pad = centers[-1] + step * np.arange(1, W - len(centers) + 1)
            centers = np.concatenate([centers, pad])
    return np.interp(t, centers, scale_w.astype(float),
                     left=float(scale_w[0]), right=float(scale_w[-1]))

def _log_hann_smooth(x_daily: np.ndarray, N: int) -> np.ndarray:
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
    injected = applier.apply_to_params(base_params, leaf_value_array=leaf_eff, scenario="none")
    r0 = injected[r0_idx, :, loc].astype(float).copy()
    if scale_w is not None:
        centers = _week_centers_from_assign(day_to_week)
        scale_d = _interp_weekly_to_daily(scale_w, centers, r0.shape[0])
        r0 *= scale_d
    r0 = _log_hann_smooth(r0, smooth_N)
    return r0


# ---------- fast mode + disable mobility toggles ----------

def _disable_mobility_in_precomputed(pc: dict) -> None:
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
    for obj in (pipe, op, getattr(op, "_rhs", None)):
        try: setattr(obj, "fast_mode", True)
        except Exception: pass
        try: setattr(obj, "disable_mobility", True)
        except Exception: pass
    try:
        setattr(op, "_disable_mobility", True)
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


# ============================= NEW helpers (ModelInfo + config patch + concat) =============================

def _make_confuse_from_yaml(yaml_path: Path) -> confuse.Configuration:
    cfg = confuse.Configuration("StructuredExampleAllStates", __name__)
    cfg.set_file(str(yaml_path))
    return cfg

def _discover_permissible_codes_and_build_mapping(cfg_path: Path, csv_path: Path) -> tuple[list[str], list[str]]:
    """
    Build a ModelInfo WITHOUT 'selected' (inflates all locations) to read `subpop_names`
    -> model codes in order. Read CSV unique sources (state names) IN ORDER as they first
    appear; we assume CSV order matches ModelInfo order (per user). Returns:
        model_codes: [code0, code1, ...]
        csv_states:  [state0, state1, ...]  (same length/order as model_codes)
    """
    conf = _make_confuse_from_yaml(cfg_path)
    mi = ModelInfo(config=conf)
    model_codes = list(map(str, mi.subpop_struct.subpop_names))

    df = pd.read_csv(csv_path, usecols=["source"])
    csv_states = list(pd.unique(df["source"]))
    if len(csv_states) != len(model_codes):
        raise AssertionError(
            f"CSV has {len(csv_states)} unique sources but ModelInfo has {len(model_codes)} subpops. "
            "Expected equal counts with the same order."
        )
    return model_codes, csv_states

def _patch_config_selected_block(
    base_cfg_path: Path,
    out_cfg_path: Path,
    *,
    selected_code: str,
) -> None:
    """
    Write a patched YAML to `out_cfg_path` that is identical to base but ensures, within
    `subpop_setup:`:
        - `selected: ["<selected_code>"]` is present (added or overwritten).
        - If `state_level:` is missing, add `state_level: TRUE` (do not override existing).
    Preserve existing absolute `geodata`/`mobility` values.
    """
    text = base_cfg_path.read_text().splitlines()
    out = []
    in_sub = False
    sub_indent = ""
    saw_state_level = False
    wrote_selected = False

    def _emit_missing(indent: str):
        nonlocal saw_state_level, wrote_selected
        if not saw_state_level:
            out.append(f"{indent}state_level: TRUE")
        if not wrote_selected:
            out.append(f'{indent}selected: ["{selected_code}"]')

    for line in text:
        if not in_sub:
            out.append(line)
            if re.match(r"^\s*subpop_setup\s*:\s*$", line):
                in_sub = True
                sub_indent = None
            continue

        # inside subpop_setup
        if sub_indent is None and line.strip():
            sub_indent = re.match(r"^(\s*)", line).group(1)
        if sub_indent is None:
            sub_indent = "  "

        stripped = line.strip()
        key = stripped.split(":")[0] if ":" in stripped else ""

        if key == "state_level":
            saw_state_level = True
            out.append(line)  # keep as-is
            continue
        if key == "selected":
            out.append(f'{sub_indent}selected: ["{selected_code}"]')
            wrote_selected = True
            continue

        # End of block? (next top-level key)
        if re.match(r"^\S", line) and not line.startswith(" "):
            _emit_missing(sub_indent)
            out.append(line)
            in_sub = False
            continue

        out.append(line)

    if in_sub:
        # file ended while in subpop_setup
        _emit_missing(sub_indent if sub_indent is not None else "  ")

    out_cfg_path.write_text("\n".join(out) + "\n")


def _concat_idatas_along_location(idatas: list[az.InferenceData]) -> az.InferenceData:
    """
    (Unused in this per-state writer, kept for convenience.)
    Concatenate multiple single-location InferenceData objects into one with a
    multi-location 'location' dimension by xarray-concatenating group datasets
    along 'location'.
    """
    out = az.InferenceData()
    group_names = set().union(*[idata.groups() for idata in idatas])

    for grp in sorted(group_names):
        dsets = [getattr(idata, grp) for idata in idatas if getattr(idata, grp, None) is not None]
        if not dsets:
            continue

        # Which vars have a 'location' dim?
        loc_vars = [vn for vn, da in dsets[0].data_vars.items() if "location" in da.dims]
        noloc_vars = [vn for vn, da in dsets[0].data_vars.items() if "location" not in da.dims]

        ds_loc_cat = None
        if loc_vars:
            parts = [ds[loc_vars] for ds in dsets]  # each has location coord length 1 with state name
            ds_loc_cat = xr.concat(parts, dim="location")

        ds_noloc = dsets[0][noloc_vars] if noloc_vars else None

        if ds_loc_cat is not None and ds_noloc is not None:
            ds_comb = xr.merge([ds_loc_cat, ds_noloc], combine_attrs="override")
        elif ds_loc_cat is not None:
            ds_comb = ds_loc_cat
        else:
            ds_comb = ds_noloc

        setattr(out, grp, ds_comb)

    return out


# ============================= TRIAD support (only for 4 random states) =============================

def _precompute_sr_mass0(pipe: WeeklyHospPipeline) -> np.ndarray:
    """sr_mass0[a, l] = (total mass in age a, loc l) minus (non S/R mass) at t0."""
    initial = pipe.initial_array
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

    sr = np.zeros((len(ages), pipe.NL), dtype=np.float64)
    for a, tok in enumerate(age_tokens):
        m_age = (comp_age_norm == tok)
        if not m_age.any():
            m_age = np.ones(NC, dtype=bool)
        m_s = m_age & is_S
        m_r = m_age & is_R
        m_other = m_age & (~m_s) & (~m_r)
        N_tot = initial[m_age, :].sum(axis=0)
        N_other = initial[m_other, :].sum(axis=0)
        sr[a, :] = np.maximum(N_tot - N_other, 0.0)
    return sr


def _triad_plot_for_state(outdir: Path,
                          state_name: str,
                          pipe: WeeklyHospPipeline,
                          op: WeeklyHospAndFinalSOp,
                          idata_state: az.InferenceData,
                          y_full_state: np.ndarray):
    """Make the 3-panel 'triad' figure for a SINGLE-location run."""
    post = idata_state.posterior
    chains = post.dims["chain"]
    draws = post.dims["draw"]

    rng = np.random.default_rng(20240831)
    flat_ix = rng.choice(chains * draws, size=2, replace=False)
    sample_pairs = [(ix // draws, ix % draws) for ix in flat_ix]

    mu_shifted_name = "weekly_pred_sum_age_shifted"
    if mu_shifted_name not in post:
        raise AssertionError(f"'{mu_shifted_name}' not found in posterior.")

    mu_shifted = post[mu_shifted_name].values            # (chain, draw, W, 1)
    beta0_loc = post["beta0_loc"].values                 # (chain, draw, 1)
    delta_week = post["delta_week"].values               # (chain, draw, W, 1)
    alpha_nb_loc = post["alpha_nb_loc"].values if "alpha_nb_loc" in post else None

    sr_mass0 = _precompute_sr_mass0(pipe)

    # r0-effective prep
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

    mods_loc = post["mods_loc"].values                  # (chain,draw,modifier,1)
    mods = post["mods"].values if "mods" in post else None
    pR = post["pR"].values
    S_final = post["S_final"].values
    r0_weekly_scale_post = post["r0_weekly_scale"].values if "r0_weekly_scale" in post else None

    # Seasonal prior-mean leaf set (monthly/holiday)
    leaf_names = tuple(pipe.modifier_order())
    defaults = _yaml_defaults_in_leaf_order(pipe.config_path, leaf_names)
    mh_idx = [i for i, nm in enumerate(leaf_names) if ("month" in nm.lower()) or ("holi" in nm.lower())]

    colors = ["C0", "C1", "C2"]
    BASELINE_R0_COLOR = "0.5"   # gray

    fig = plt.figure(figsize=(12, 9), dpi=130)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.0, 1.2], hspace=0.28)

    # Panel 1: r0 baseline vs effective injected (after weekly interp; Fourier)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_days, base[r0_idx, :, 0], label="baseline r0", linewidth=1.6, color=BASELINE_R0_COLOR)
    s0_points, sT_points = [], []

    if mh_idx:
        leaf_eff_prior = np.ones(len(leaf_names), dtype=float)
        leaf_eff_prior[mh_idx] = defaults[mh_idx]
        smooth_N = int(getattr(op, "_smooth_r0_days", 0) or 0)
        r0_seasonal = _effective_r0_daily(
            base, applier, leaf_eff_prior, r0_idx, 0,
            scale_w=None, day_to_week=day_to_week, smooth_N=smooth_N
        )
        ax1.plot(t_days, r0_seasonal, linestyle=":", linewidth=1.8, color="black",
                 label="seasonal prior-mean")

    smooth_N = int(getattr(op, "_smooth_r0_days", 0) or 0)
    for k, (c, d) in enumerate(sample_pairs):
        if mods is not None:
            mods_vec_sh = np.asarray(mods[c, d, :], dtype=float)
        else:
            mods_vec_sh = np.ones(mods_loc.shape[2], dtype=float)
        mods_loc_vec = np.asarray(mods_loc[c, d, :, 0], dtype=float)
        leaf_eff = mods_vec_sh * mods_loc_vec

        scale_w = None
        if r0_weekly_scale_post is not None:
            scale_w = np.array(r0_weekly_scale_post[c, d, :, 0], dtype=float)

        r0_eff = _effective_r0_daily(
            base, applier, leaf_eff, r0_idx, 0,
            scale_w=scale_w, day_to_week=day_to_week, smooth_N=smooth_N
        )
        ax1.plot(t_days, r0_eff, linestyle="--", linewidth=1.4, color=colors[k],
                 label=f"sample {k+1} (eff r0 via Fourier)")

        pR_samp = pR[c, d, :, 0]
        S0_agg = np.sum((1.0 - pR_samp) * sr_mass0[:, 0]); s0_points.append(S0_agg)
        Sfinal_agg = np.sum(S_final[c, d, :, 0]); sT_points.append(Sfinal_agg)

    ax1.set_title(f"{state_name} — r0 baseline vs effective injected (Fourier scale, unit-mean)")
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
    ax2.set_title(f"{state_name} — S0 vs S(T)"); ax2.grid(True, alpha=0.3); ax2.legend(loc="best")

    # Panel 3: Weekly hosp with 50% noise bands + target
    ax3 = fig.add_subplot(gs[2, 0])
    W = y_full_state.shape[0]
    w = np.arange(W)

    for k, (c, d) in enumerate(sample_pairs):
        mu_t = np.asarray(mu_shifted[c, d, :, 0], dtype=float) * np.exp(
            float(beta0_loc[c, d, 0])
        ) * np.exp(np.asarray(delta_week[c, d, :, 0], dtype=float))
        if alpha_nb_loc is not None:
            alpha_loc = float(alpha_nb_loc[c, d, 0])
            var_t = mu_t + (mu_t ** 2) / max(alpha_loc, 1e-12)
        else:
            var_t = mu_t
        sd_t = np.sqrt(np.maximum(var_t, 1e-12))
        z50 = 0.67448975
        lo50 = np.clip(mu_t - z50 * sd_t, 0.0, np.inf)
        hi50 = mu_t + z50 * sd_t

        ax3.fill_between(w, lo50, hi50, alpha=0.20, step="mid", color=colors[k],
                         label=f"sample {k+1} 50% band")
        ax3.plot(w, mu_t, linewidth=1.6, color=colors[k], label=f"sample {k+1} mean")

    _plot_weekly_targets(ax3, y_full_state[:, 0].astype(float), label="Target (data)", color="0.1")
    ax3.set_xlabel("Week"); ax3.set_ylabel("Hosp (sum over age)")
    ax3.set_title(f"{state_name} — Weekly hosp (samples + 50% noise band vs target)")
    ax3.grid(True, alpha=0.3); ax3.legend(loc="best")

    fig.suptitle(f"Three-panel summary — {state_name}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile = outdir / f"triad_{state_name}.png"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[artifact] saved: {outfile}")


# ============================= NetCDF sanity check =============================

def _assert_netcdf_has_posterior(nc_path: Path) -> None:
    """Open a freshly-written NetCDF and assert it contains a non-empty posterior group."""
    if (not nc_path.exists()) or nc_path.stat().st_size == 0:
        raise RuntimeError(f"NetCDF appears empty or missing: {nc_path}")
    try:
        idata = az.from_netcdf(nc_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read NetCDF ({nc_path}): {e}") from e

    if not hasattr(idata, "posterior") or idata.posterior is None:
        raise RuntimeError(f"No 'posterior' group found in NetCDF: {nc_path}")

    ds = idata.posterior
    for dim in ("chain", "draw"):
        if dim not in ds.dims or int(ds.dims[dim]) <= 0:
            raise RuntimeError(f"'posterior' has non-positive {dim} in NetCDF: {nc_path}")

    expected_any = ["weekly_pred", "pR", "mods_loc", "beta0_loc"]
    if not any(v in ds.data_vars for v in expected_any):
        raise RuntimeError(f"'posterior' missing expected vars {expected_any} in NetCDF: {nc_path}")


# ============================= main test =============================

def _locate_csv_or_skip() -> Path:
    csv_env = os.environ.get("REALDATA_CSV") or "/Users/josh/Documents/test_data_alt.csv"
    csv_path = Path(csv_env)
    if not csv_path.exists():
        pytest.skip(f"REALDATA_CSV not found at {csv_path}; skipping real-data test.")
    return csv_path


@pytest.mark.slow
def test_pymc_weekly_inference_all_states_per_file_fourier(tmp_path_factory):
    """
    Full run over ALL states in the CSV (aligned to ModelInfo order).
    For EACH state, run single-location inference with the FOURIER r0-scale prior and write ONE NetCDF per state:
        inference_idata_<STATE>.nc
    Also write PPC and triad plots per state.
    After the FIRST state's NetCDF is written, re-open it and verify it's non-empty.
    """
    # ---------- quick knobs ----------
    PRIOR_SAMPLES = int(os.environ.get("PRIOR_SAMPLES", "10"))
    TUNE = int(os.environ.get("TUNE", "700"))
    DRAWS = int(os.environ.get("DRAWS", "300"))
    CHAINS = int(os.environ.get("CHAINS", "2"))
    CORES = min(CHAINS, max(1, os.cpu_count() or 1))
    RNG_SEED = int(os.environ.get("STATE_SAMPLE_SEED", "20240901"))
    FOURIER_SCALE = bool(int(os.environ.get("FOURIER_SCALE", "1")))  
    FOURIER_HARMONICS = int(os.environ.get("FOURIER_HARMONICS", "16"))          # K
    FOURIER_PERIOD_DAYS = float(os.environ.get("FOURIER_PERIOD_DAYS", "365.25"))
    PROGRESS_BAR = bool(int(os.environ.get("PROGRESS_BAR", "1")))
    USE_NB = bool(int(os.environ.get("USE_NB", "0")))
    # ----------------------------------

    base_cfg_path = _materialize_structured_example(tmp_path_factory)
    csv_path = _locate_csv_or_skip()

    # Autotune ONCE
    _safe_autotune()

    # Discover (model_code, state_name) ordered pairs
    model_codes, csv_states = _discover_permissible_codes_and_build_mapping(base_cfg_path, csv_path)
    L_total = len(model_codes)
    assert L_total == len(csv_states) and L_total >= 1

    # Choose 4 random states (or fewer if <4 total) for prior spaghetti + triad plots
    rng = np.random.default_rng(RNG_SEED)
    four_idx = set(rng.choice(L_total, size=min(4, L_total), replace=False).tolist())

    # Output directory (single)
    outdir = Path(os.environ.get("E2E_OUTDIR", "") or (Path.cwd() / "model_output_all_states"))
    outdir.mkdir(parents=True, exist_ok=True)

    first_nc_checked = False

    for i in range(L_total):
        code_i = model_codes[i]   # e.g., "01000"
        state_i = csv_states[i]   # e.g., "Alabama"
        print(f"\n=== [{i+1}/{L_total}] State={state_i} (code {code_i}) — Fourier K={FOURIER_HARMONICS}, P={FOURIER_PERIOD_DAYS} ===")

        # Patch config for this state into a temp file
        cfg_state = base_cfg_path.with_name(f"{base_cfg_path.stem}__{code_i}.yml")
        _patch_config_selected_block(base_cfg_path, cfg_state, selected_code=code_i)

        # Build single-location pipeline/op
        try:
            pipe = build_pipeline_from_config(cfg_state, dt_days=0.5, fast_mode=True, disable_mobility=True)
        except TypeError:
            pipe = build_pipeline_from_config(cfg_state, dt_days=0.5)
        op = WeeklyHospAndFinalSOp(pipe, fast_mode=True, disable_mobility=True)
        _enable_fast_and_disable_mobility(pipe, op)

        # Align CSV to THIS single-location model (subset to just this state name)
        y_full, obs_weeks, loc_names = _load_and_align_csv_to_weeks(
            csv_path, pipe, op, subset_sources=(state_i,)
        )
        assert y_full.shape[1] == 1 and op.locations == 1, "Single-location run expected."

        # Optional PRIOR spaghetti ONLY for selected 4 states (Fourier prior)
        if i in four_idx:
            with build_weekly_model(
                pipe, op=op, y_obs=None,
                force_r0_fourier_scale=FOURIER_SCALE,
                fourier_harmonics=FOURIER_HARMONICS,
                fourier_period_days=FOURIER_PERIOD_DAYS,
            ) as prior_model:
                present = set(prior_model.named_vars.keys())
                requested = ["weekly_pred", "mods_loc", "mods_mu_log_loc", "r0_weekly_scale"]
                prior_vars = [v for v in requested if v in present]
                prior_idata = pm.sample_prior_predictive(
                    samples=PRIOR_SAMPLES,
                    random_seed=123 + i,
                    var_names=prior_vars,
                )
            # spaghetti: sum over age
            weekly_vals = _idata_get(prior_idata, "weekly_pred")
            weekly = _stack_samples(weekly_vals)  # (S, A, W, 1)
            series = weekly.sum(axis=1)[:, :, 0]  # (S, W)
            weeks = np.arange(series.shape[1])
            fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=120)
            for s in range(min(PRIOR_SAMPLES, series.shape[0])):
                ax.plot(weeks, series[s], alpha=0.25, linewidth=1.0)
            ax.set_title(f"Prior predictive (Fourier r0-scale) — sum over ages, {state_i}")
            ax.set_xlabel("Week"); ax.set_ylabel("Weekly hospitalizations (sum over age)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(outdir / f"prior_spaghetti_{state_i}.png", bbox_inches="tight")
            plt.close(fig)

        # ---- POSTERIOR for this state (Fourier path) ----
        with build_weekly_model(
            pipe, op=op, y_obs=y_full, use_nb=USE_NB,
            force_r0_fourier_scale=FOURIER_SCALE,
            fourier_harmonics=FOURIER_HARMONICS,
            fourier_period_days=FOURIER_PERIOD_DAYS,
        ) as model:
            idata = pm.sample(
                draws=DRAWS,
                tune=TUNE,
                chains=CHAINS,
                cores=CORES,
                step=pm.DEMetropolisZ(tune_interval=100),
                random_seed=777 + i,
                progressbar=PROGRESS_BAR,
            )
            ppc = pm.sample_posterior_predictive(
                idata,
                var_names=["y", "weekly_pred_sum_age_shifted", "weekly_pred"],
                random_seed=888 + i,
                progressbar=PROGRESS_BAR,
            )
        idata.extend(ppc)

        # Tag with *state name* so saved tensors have a human-readable location coord
        idata = idata.copy()
        for group_name in idata.groups():
            ds = getattr(idata, group_name)
            if ds is None:
                continue
            if "location" in ds.dims or "location" in ds.coords:
                setattr(idata, group_name, ds.assign_coords(location=[str(state_i)]))

        # Per-state PPC + (optionally) TRIAD plots
        _panel_per_location(idata, y_full, np.arange(y_full.shape[0]), (state_i,), pipe, outdir)
        if i in four_idx:
            _triad_plot_for_state(outdir, state_i, pipe, op, idata, y_full)

        # ---- SAVE *ONE FILE PER STATE* ----
        nc_path = outdir / f"inference_idata_{state_i}.nc"
        az.to_netcdf(idata, nc_path)
        print(f"[artifact] saved per-state idata:", nc_path)

        # After FIRST state's file is written, check it's not empty
        if not first_nc_checked:
            _assert_netcdf_has_posterior(nc_path)
            print(f"[sanity] verified non-empty posterior in: {nc_path}")
            first_nc_checked = True

    print("\n[done] Wrote one NetCDF per state + plots to:", outdir)
