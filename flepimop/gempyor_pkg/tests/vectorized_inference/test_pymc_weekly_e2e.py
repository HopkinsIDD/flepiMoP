# tests/vectorized_inference/test_pymc_weekly_e2e.py
# End-to-end PyMC test using the Structured_Example config, our WeeklyHospPipeline,
# and the WeeklyHospAndFinalSOp Op. We generate synthetic data directly from the Op
# (which uses real transition amounts + YAML probabilities + delays), then run a short
# calibration and write prior/posterior predictive and trace plots.
#
# NOTE: This is a slow test (full 51 locations). Marked as @pytest.mark.slow.
# TIP: run with -s to see tqdm progress bars in the terminal:
#   E2E_OUTDIR=tests/vectorized_inference/_artifacts pytest -s tests/vectorized_inference/test_pymc_weekly_e2e.py

import os
import platform
from ctypes.util import find_library

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Force PyMC/tqdm progressbars in CLI
os.environ.setdefault("PYMC_PROGRESSBAR", "1")

# --- Choose a numba threading layer *before* importing any numba users ---
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
# 4 chains -> reserve up to 4 threads by default (adjust if fewer cores)
_default_threads = str(min(4, max(1, os.cpu_count() or 1)))
os.environ.setdefault("NUMBA_NUM_THREADS", _default_threads)

# ---------------------------------------------------------------------------------
# Regular imports (SAFE now that env is set)
# ---------------------------------------------------------------------------------
from pathlib import Path
import shutil
import numpy as np
import pytest
import confuse
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymc as pm

from gempyor.vectorization_experiments import autotune_all, get_autotune_config
from gempyor.hosp_weekly_pipeline import build_pipeline_from_config
from gempyor.pymc_weekly_op import (
    WeeklyHospAndFinalSOp,
    build_weekly_model,
)

# ----------------------------- helpers ---------------------------------
def _materialize_structured_example(tmp_path_factory) -> Path:
    """Copy examples/tutorials/Structured_Example.yml & inputs into a temp root with absolute paths."""
    tmp_root = tmp_path_factory.mktemp("weekly_infer_case")

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


def _yaml_defaults_in_leaf_order(config_path: Path, leaf_order: tuple[str, ...]) -> np.ndarray:
    """Robustly extract default replacement multipliers in the pipeline’s leaf order."""
    conf = confuse.Configuration("WeeklyE2EDefaults", __name__)
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


def _normalize_age_token(s: str) -> str:
    s = str(s).lower().replace("to", "-").replace("–", "-").replace("—", "-")
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(ch)
        elif ch in "-_":
            out.append("_")
        elif ch == "+":
            out.append("p")
    key = []
    for c in out:
        if not (key and key[-1] == "_" and c == "_"):
            key.append(c)
    return "".join(key).strip("_")


def _pR_from_initial_sr_share(pipe, op) -> np.ndarray:
    """
    Build baseline pR per (age, location) from the initial state BUT using the Op's
    S/R-redistribution semantics: pR is the fraction of R within (S+R), not within total.
    """
    df = pipe.model.compartments.compartments
    is_S = df["infection_stage"].astype(str).str.startswith("S").values
    is_R = df["infection_stage"].astype(str).str.startswith("R").values

    A = len(pipe.age_labels)
    L = pipe.NL
    pR = np.zeros((A, L), dtype=np.float64)
    for a_idx, (m_s, m_tot, m_r) in enumerate(
        zip(op.age_masks.s_mask_by_age, op.age_masks.total_mask_by_age, op.age_masks.r_mask_by_age)
    ):
        m_other = m_tot & (~m_s) & (~m_r)
        N = pipe.initial_array
        N_total = N[m_tot, :].sum(axis=0)
        N_other = N[m_other, :].sum(axis=0)
        N_sr = np.maximum(N_total - N_other, 1e-9)
        R_counts = N[m_r, :].sum(axis=0)
        pR[a_idx, :] = np.clip(R_counts / N_sr, 1e-6, 1 - 1e-6)
    return pR


def _run_op_forward(op: WeeklyHospAndFinalSOp, mods: np.ndarray, pR: np.ndarray, lambda_ext_loc: np.ndarray):
    """Call the Op directly to obtain (weekly, S_final) as numpy arrays."""
    out_w = [None]
    out_S = [None]
    op.perform(None, [mods, pR, lambda_ext_loc], [out_w, out_S])
    weekly = np.asarray(out_w[0], dtype=np.float64)
    S_final = np.asarray(out_S[0], dtype=np.float64)
    return weekly, S_final


def _make_synthetic_hosp_from_op(pipe, op, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic weekly hospitalization data shaped (A,W,L) by:
      1) Using YAML-default replacement modifiers in pipeline order,
      2) Running the Op forward (true transition amounts + YAML incidence→hosp transforms),
      3) Drawing Negative Binomial noise around the weekly mean.
    """
    defaults = _yaml_defaults_in_leaf_order(pipe.config_path, pipe.modifier_order())
    pR0 = _pR_from_initial_sr_share(pipe, op)
    lambda1 = np.ones(pipe.NL, dtype=np.float64)

    weekly_mean, _ = _run_op_forward(op, defaults, pR0, lambda1)  # (A,W,L)
    mu = np.maximum(weekly_mean, 0.0) + 1e-6

    alpha = 10.0
    rate = alpha / mu
    lam = rng.gamma(shape=alpha, scale=1.0 / rate)  # Gamma-Poisson mixture
    y = rng.poisson(lam)
    return y.astype(np.int64, copy=False)


def _save_trace_plot(idata, outpng: Path):
    """Save a compact trace plot for a subset of parameters."""
    try:
        az.plot_trace(
            idata,
            var_names=["mods", "mu_R_logit", "sigma_age", "sigma_loc", "alpha_nb"],
            compact=True,
            figsize=(12, 6)
        )
        plt.tight_layout()
        plt.savefig(outpng, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def _posterior_predictive_panels_two_locations(idata, y_obs, pipe, outdir: Path):
    """
    Create panel figures for two locations. For each location, plot posterior predictive
    envelopes (mean + 95% HDI) vs observed for the first K age groups (K<=4) across weeks.
    Uses ArviZ across (chain, draw) to avoid shape warnings.
    """
    A = y_obs.shape[0]
    W = y_obs.shape[1]
    ages = np.array(pipe.age_labels, dtype=object)
    weeks = np.arange(W)

    L = y_obs.shape[2]
    locs = [0, min(L - 1, L // 2)]
    K = min(4, A)

    ppc = idata.posterior_predictive["y"].values  # (chain, draw, A, W, L)

    for loc in locs:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=120, sharex=True)
        axes = axes.ravel()
        for a_idx in range(K):
            ax = axes[a_idx]
            y_loc = y_obs[a_idx, :, loc]
            # samples has shape (chain, draw, W)
            samples = ppc[:, :, a_idx, :, loc]
            mean = samples.mean(axis=(0, 1))  # (W,)
            # HDI across (chain, draw), returns (W, 2)
            hdi = az.hdi(samples, hdi_prob=0.95)

            ax.fill_between(weeks, hdi[:, 0], hdi[:, 1], alpha=0.25, step="mid", label="95% HDI")
            ax.plot(weeks, mean, linewidth=1.5, label="Posterior mean")
            ax.step(weeks, y_loc, where="mid", linewidth=1.2, label="Observed")
            ax.set_title(f"Loc {loc} — {ages[a_idx]}")
            ax.grid(True, alpha=0.3)
            if a_idx in (2, 3):
                ax.set_xlabel("MMWR week")
            if a_idx in (0, 2):
                ax.set_ylabel("Hosp")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Posterior predictive — Location {loc} (first {K} ages)", y=0.98)
        fig.tight_layout(rect=[0, 0, 0.96, 0.95])

        outpng = outdir / f"ppc_loc{loc}.png"
        fig.savefig(outpng, bbox_inches="tight")
        plt.close(fig)


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


# ----------------------------- tests -----------------------------------

@pytest.fixture(scope="module")
def pipeline_and_op(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)
    pipe = build_pipeline_from_config(cfg_path)

    # Perform hardware optimization after resetting Numba params.
    _safe_autotune()

    op = WeeklyHospAndFinalSOp(pipe)
    return pipe, op


@pytest.mark.slow
def test_pymc_weekly_e2e_inference(pipeline_and_op, tmp_path):
    """
    End-to-end test with visible CLI progress:
      1) Build model with coords.
      2) Generate synthetic weekly hospitalization data from the Op (true transition amounts).
      3) Validate λ_ext scaling and pR override invariants.
      4) Prior predictive.
      5) Run short posterior.
      6) Posterior predictive; save summary + plots.
    """
    pipe, op = pipeline_and_op
    rng = np.random.default_rng(123)

    # --- Synthetic observations from the *Op* path (uses true transition amounts) ---
    y_obs = _make_synthetic_hosp_from_op(pipe, op, rng)
    A, W, L = y_obs.shape
    assert (A, W, L) == op.weekly_shape

    # --- Sanity checks specific to recent fixes ---

    # (1) λ_ext scaling: Only enforce an increase if the underlying lambda_ext time series is non-zero.
    defaults = _yaml_defaults_in_leaf_order(pipe.config_path, pipe.modifier_order())
    pR0 = _pR_from_initial_sr_share(pipe, op)
    weekly_base, _ = _run_op_forward(op, defaults, pR0, np.ones(L))
    weekly_scaled, _ = _run_op_forward(op, defaults, pR0, np.full(L, 2.0))
    tot_base = float(weekly_base.sum())
    tot_scaled = float(weekly_scaled.sum())
    assert np.isfinite(tot_base) and np.isfinite(tot_scaled)

    # Examine lambda_ext in the pipeline
    pmap = getattr(pipe, "param_name_to_idx", None) or getattr(pipe.mod_applier, "param_name_to_idx", {})
    if "lambda_ext" in pmap:
        pidx = int(pmap["lambda_ext"])
        lam_base = np.asarray(pipe.base_params[pidx, :, :], dtype=np.float64)
        if np.allclose(lam_base, 0.0, rtol=0.0, atol=1e-14):
            # No exogenous force configured -> scaling should do nothing
            np.testing.assert_allclose(tot_scaled, tot_base, rtol=0.0, atol=1e-9)
        else:
            # With nonzero exogenous force, totals should increase noticeably
            assert tot_scaled > 1.01 * tot_base
    else:
        # No such parameter -> scaling should do nothing
        np.testing.assert_allclose(tot_scaled, tot_base, rtol=0.0, atol=1e-9)

    # (2) pR override preserves non-(S|R) mass (e.g., E/I seeding) per (age, location)
    df = pipe.model.compartments.compartments
    is_E_or_I = df["infection_stage"].astype(str).str.startswith(("E", "I")).values
    y0_orig = pipe.initial_array
    # Push pR to an extreme to stress the override
    pR_extreme = np.clip(pR0 * 0 + 0.9, 1e-6, 1 - 1e-6)
    y0_new = op._override_ic_with_pR_preserve_others(y0_orig, pR_extreme)
    for a_idx, m_tot in enumerate(op.age_masks.total_mask_by_age):
        m_ei = m_tot & is_E_or_I
        before = y0_orig[m_ei, :].sum(axis=0)
        after = y0_new[m_ei, :].sum(axis=0)
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-9)

    # --- Build model (uses Negative Binomial likelihood if y_obs is provided) ---
    with build_weekly_model(pipe, op=op, y_obs=y_obs) as model:
        # Prior predictive (show progress)
        prior_idata = pm.sample_prior_predictive(draws=500, random_seed=123)

        # Posterior — 4 chains across 4 cores (show progress)
        ncores = min(4, os.cpu_count() or 1)
        idata = pm.sample(
            draws=200,
            tune=200,
            chains=ncores,
            cores=ncores,
            step=pm.DEMetropolisZ(),  # derivative-free black-box
            random_seed=123,
            progressbar=True,         # <-- ensure CLI tqdm bars
        )

        # Posterior predictive (show progress)
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["y", "weekly_pred", "S_final"], random_seed=123, progressbar=True
        )

    # Merge for ArviZ convenience
    idata.extend(prior_idata)
    idata.extend(ppc)

    # Basic sanity assertions
    assert "weekly_pred" in idata.posterior
    assert "S_final" in idata.posterior
    assert "y" in idata.posterior_predictive
    posterior_vars = set(idata.posterior.data_vars)
    # New external-force parameter should be present (allow either naming pattern)
    assert any(v in posterior_vars for v in ["lambda_ext_loc", "lambda_ext_log", "lambda_ext_log_loc"])

    # Choose output dir:
    outdir_env = os.environ.get("E2E_OUTDIR", "").strip()
    outdir = Path(outdir_env) if outdir_env else Path(tmp_path)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    def _save_trace_plot(idata, outpng: Path):
        try:
            az.plot_trace(
                idata,
                var_names=["mods", "mu_R_logit", "sigma_age", "sigma_loc", "alpha_nb"],
                compact=True,
                figsize=(12, 6)
            )
            plt.tight_layout()
            plt.savefig(outpng, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    _save_trace_plot(idata, outdir / "trace_compact.png")
    _posterior_predictive_panels_two_locations(idata, y_obs, pipe, outdir)

    # Prior predictive quick check
    if "y" in idata.prior_predictive:
        prior_y = idata.prior_predictive["y"].values  # (chain, draw, A, W, L)
        mu = prior_y.mean(axis=(0, 1))
        hdi = az.hdi(prior_y, hdi_prob=0.95)          # (A, W, L, 2)
        assert np.isfinite(mu).all()
        assert np.isfinite(hdi).all()

    # Quick posterior check on S_final deterministics (finite, nonnegative)
    S_final = idata.posterior["S_final"].values  # (chain, draw, A, L)
    assert np.isfinite(S_final).all()
    assert (S_final >= 0.0).all()

    # Persist a small text summary (useful in CI logs)
    core_vars = ["mods", "mu_R_logit", "sigma_age", "sigma_loc", "alpha_nb"]
    core_vars += [v for v in ["lambda_ext_loc", "lambda_ext_log", "lambda_ext_log_loc"] if v in posterior_vars]
    summ = az.summary(idata, var_names=core_vars)
    (outdir / "summary.txt").write_text(summ.to_string())
    print("[artifact] wrote:", outdir / "summary.txt")
