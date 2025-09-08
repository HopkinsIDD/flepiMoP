# src/gempyor/pymc_weekly_op.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import datetime as _dt
import confuse
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

from gempyor.hosp_weekly_pipeline import (
    WeeklyHospPipeline,
    build_pipeline_from_config,
    _compile_outcomes_from_config,
    _param_slice_step,
    _compute_amounts_for_step,
    _steps_to_weeks_assign,
)
from gempyor.vectorization_experiments import RHSfactory


# ------------------------------ utilities ------------------------------

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


def _yaml_defaults_in_leaf_order(config_path: Path, leaf_order: tuple[str, ...]) -> np.ndarray:
    conf = confuse.Configuration("WeeklyHospPipelineDefaults", __name__)
    conf.set_file(str(config_path))
    sm = conf["seir_modifiers"].get()
    mods = sm["modifiers"]
    return np.asarray([_extract_yaml_value(mods[nm]) for nm in leaf_order], dtype=np.float64)


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


@dataclass(frozen=True)
class AgeMasks:
    age_labels: tuple[str, ...]
    s_mask_by_age: tuple[np.ndarray, ...]
    r_mask_by_age: tuple[np.ndarray, ...]
    total_mask_by_age: tuple[np.ndarray, ...]


def _build_age_masks(pipeline: WeeklyHospPipeline) -> AgeMasks:
    df = pipeline.model.compartments.compartments
    NC = pipeline.NC
    comp_age_norm = df["age_strata"].astype(str).map(_normalize_age_token).values
    comp_stage = df["infection_stage"].astype(str).values
    is_S = np.fromiter((str(st).startswith("S") for st in comp_stage), dtype=bool, count=NC)
    is_R = np.fromiter((str(st).startswith("R") for st in comp_stage), dtype=bool, count=NC)

    masks_s, masks_r, masks_tot = [], [], []
    for age_token in pipeline.age_labels:
        key = _normalize_age_token(age_token.replace("age", "", 1))
        m_age = (comp_age_norm == key)
        if not m_age.any():
            m_age = np.ones(NC, dtype=bool)
        masks_s.append(m_age & is_S)
        masks_r.append(m_age & is_R)
        masks_tot.append(m_age)
    return AgeMasks(tuple(pipeline.age_labels), tuple(masks_s), tuple(masks_r), tuple(masks_tot))


def _mmwr_assign_from(start_date: np.datetime64 | object, T: int) -> tuple[np.ndarray, int]:
    if T <= 0:
        return np.zeros(0, dtype=np.int64), 0
    if isinstance(start_date, np.datetime64):
        start = _dt.date.fromtimestamp((start_date - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"))
    else:
        start = start_date
    offset_to_sun = (6 - start.weekday()) % 7
    first_len = 7 if offset_to_sun == 0 else offset_to_sun
    first_len = min(first_len, T)
    assign = np.empty(T, dtype=np.int64)
    assign[:first_len] = 0
    if T > first_len:
        rest = T - first_len
        assign[first_len:] = 1 + (np.arange(rest, dtype=np.int64) // 7)
    n_weeks = int(assign.max()) + 1
    return assign, n_weeks


# ------------------------------ Op ------------------------------

class WeeklyHospAndFinalSOp(Op):
    """
    Runs the simulator once and returns:
      • weekly   : (age, week, location) accumulated outcomes from config
      • S_final  : (age, location) final susceptible counts

    Inputs (flexible):
      mods            (1D): shared leaf values in pipeline.leaf order
      pR              (2D): (age, location)
      lambda_ext_loc  (1D, optional): per-location scale if 'lambda_ext' exists
      mods_loc        (2D, optional): (modifier, location)
      r0_weekly_scale (2D, optional): (week, location) multiplicative scale for r0
                                      ONLY used when **no r0 modifiers** exist in config.

    If 'lambda_ext' is absent in the config, any provided lambda_ext_loc is ignored.
    If 'r0' is not a target of any modifiers in the config, you may pass r0_weekly_scale.
    """

    def __init__(self, pipeline: WeeklyHospPipeline):
        super().__init__()
        self.pipe = pipeline
        self.leaf_order = pipeline.modifier_order()
        self.age_masks = _build_age_masks(pipeline)

        # Probe shapes (fixes A, W, L)
        defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, self.leaf_order)
        wk, _, _ = pipeline.evaluate(defaults)
        self._weekly_shape = wk.shape
        self._A, self._W, self._L = self._weekly_shape
        self._age_to_names = dict(pipeline.age_to_names)
        self._age_labels = tuple(pipeline.age_labels)

        # Steps→weeks assignment (for mapping weekly r0 scale to days)
        self._day_to_week, self._n_weeks = _mmwr_assign_from(pipeline.start_date, pipeline.T)

        # Outcomes compiled at the pipeline step width
        outcomes_cfg = (
            pipeline.outcomes_cfg
            if hasattr(pipeline, "outcomes_cfg")
            else pipeline.model.outcomes_config["outcomes"].get()
        )
        (
            self._resolve_map,
            self._prob_map,
            self._delay_steps_map,
            self._sum_map,
            self._out_order,
        ) = _compile_outcomes_from_config(
            outcomes_cfg,
            pipeline.model.compartments.compartments,
            pipeline.transitions,
            pipeline.NL,
            pipeline.dt,
        )

        # Direct-indexing RHS (Pattern #2)
        self._rhs = RHSfactory(precomputed=self.pipe.precomputed, param_time_mode="step")

        # Optional mapping (base param names → row indices)
        self._param_name_to_idx: dict[str, int] = {}
        if hasattr(self.pipe, "param_name_to_idx"):
            self._param_name_to_idx = dict(self.pipe.param_name_to_idx)
        elif hasattr(self.pipe, "mod_applier"):
            self._param_name_to_idx = dict(getattr(self.pipe.mod_applier, "param_name_to_idx", {}))

        # Presence flags
        self._has_lambda_ext = ("lambda_ext" in self._param_name_to_idx)
        self._has_r0 = ("r0" in self._param_name_to_idx)

        # Detect if any config modifiers target r0
        self._has_r0_modifiers = self._detect_r0_modifiers()

    # ---- helpers -------------------------------------------------------

    def _detect_r0_modifiers(self) -> bool:
        # Best effort: inspect config; fall back to leaf names if needed.
        try:
            sm = self.pipe.config["seir_modifiers"].get()
            mods = sm.get("modifiers", {})
            for name, spec in mods.items():
                if isinstance(spec, dict) and str(spec.get("parameter", "")).lower() == "r0":
                    return True
        except Exception:
            pass
        # Heuristic fallback: look at applier metadata if present
        try:
            applier = getattr(self.pipe, "mod_applier", None)
            targets = getattr(applier, "leaf_targets", None)
            if isinstance(targets, dict):
                for t in targets.values():
                    if isinstance(t, str) and t.lower() == "r0":
                        return True
                    if isinstance(t, (tuple, list)) and any(str(x).lower() == "r0" for x in t):
                        return True
        except Exception:
            pass
        # Last resort: leaf names that mention r0
        return any("r0" in str(nm).lower() for nm in self.leaf_order)

    def _get_param_row(self, name: str) -> int:
        try:
            return int(self._param_name_to_idx[name])
        except Exception as e:
            raise KeyError(f"Parameter row '{name}' not found in pipeline mapping.") from e

    def _scale_lambda_ext(self, params_base: np.ndarray, lambda_ext_loc: np.ndarray | None) -> None:
        if (not self._has_lambda_ext) or (lambda_ext_loc is None):
            return
        pidx = self._get_param_row("lambda_ext")
        params_base[pidx, :, :] *= lambda_ext_loc[None, :]

    def _scale_r0_weekly(self, params_base: np.ndarray, r0_weekly_scale: np.ndarray | None) -> None:
        """Multiply r0(t,ℓ) by weekly scale (week,ℓ) mapped to days. No-op unless allowed and provided."""
        if (not self._has_r0) or self._has_r0_modifiers or (r0_weekly_scale is None):
            return
        pidx = self._get_param_row("r0")
        W = int(self._n_weeks)
        expected = (W, self._L)
        if r0_weekly_scale.shape != expected:
            raise ValueError(f"r0_weekly_scale shape {r0_weekly_scale.shape} != {expected}")
        # Map days → week index; params_base is (P, T_days, L)
        T_days = params_base.shape[1]
        for d in range(T_days):
            w = int(self._day_to_week[d]) if d < len(self._day_to_week) else min(W - 1, W - 1)
            params_base[pidx, d, :] *= r0_weekly_scale[w, :]

    def _override_ic_with_pR_preserve_others(self, y0_in: np.ndarray, pR: np.ndarray) -> np.ndarray:
        y0 = np.array(y0_in, dtype=np.float64, copy=True)
        for a, (m_s, m_r, m_tot) in enumerate(
            zip(self.age_masks.s_mask_by_age, self.age_masks.r_mask_by_age, self.age_masks.total_mask_by_age)
        ):
            m_other = m_tot & (~m_s) & (~m_r)
            N_tot = y0[m_tot, :].sum(axis=0)
            N_other = y0[m_other, :].sum(axis=0)
            sr_mass = np.maximum(N_tot - N_other, 0.0)

            pR_clip = np.clip(pR[a, :], 1e-6, 1 - 1e-6)
            R_counts = pR_clip * sr_mass
            S_counts = sr_mass - R_counts

            y0[m_s, :] = 0.0
            y0[m_r, :] = 0.0
            n_s, n_r = int(m_s.sum()), int(m_r.sum())
            if n_s > 0:
                y0[m_s, :] += (S_counts[None, :] / n_s)
            if n_r > 0:
                y0[m_r, :] += (R_counts[None, :] / n_r)
        return y0

    def _apply_modifiers_locationwise(
        self,
        mods_shared: np.ndarray,
        mods_loc: np.ndarray | None,
    ) -> np.ndarray:
        mods_shared = np.asarray(mods_shared, dtype=np.float64).reshape(-1)
        L = self._L
        if mods_loc is None:
            eff = np.tile(mods_shared[:, None], (1, L))
        else:
            mods_loc = np.asarray(mods_loc, dtype=np.float64)
            if mods_loc.shape != (mods_shared.shape[0], L):
                raise ValueError(f"mods_loc shape {mods_loc.shape} != {(mods_shared.shape[0], L)}")
            eff = mods_shared[:, None] * mods_loc

        out = np.array(self.pipe.base_params, dtype=np.float64, copy=True)
        for j in range(L):
            params_j = self.pipe.mod_applier.apply_to_params(
                self.pipe.base_params, leaf_value_array=eff[:, j], scenario="none"
            )
            out[:, :, j] = params_j[:, :, j]
        return out

    # ---- Op API --------------------------------------------------------

    def make_node(self, mods, pR, lambda_ext_loc=None, mods_loc=None, r0_weekly_scale=None):
        """
        Accepts 2–5 inputs. If 'r0' has no modifiers in the config, you may pass r0_weekly_scale (W,L).
        If 'lambda_ext' is absent, any provided lambda_ext_loc is ignored.
        """
        mods = pt.as_tensor_variable(mods)
        pR = pt.as_tensor_variable(pR)
        if mods.ndim != 1:
            raise TypeError("mods must be 1D")
        if pR.ndim != 2:
            raise TypeError("pR must be 2D (age, location)")

        inputs = [mods, pR]

        # Optional lambda_ext_loc
        if (lambda_ext_loc is not None) and self._has_lambda_ext:
            lam_node = pt.as_tensor_variable(lambda_ext_loc)
            if lam_node.ndim != 1:
                raise TypeError("lambda_ext_loc must be 1D (location,)")
            inputs.append(lam_node)

        # Optional mods_loc
        if mods_loc is not None:
            mods_loc = pt.as_tensor_variable(mods_loc)
            if mods_loc.ndim != 2:
                raise TypeError("mods_loc must be 2D (modifier, location)")
            inputs.append(mods_loc)

        # Optional r0_weekly_scale (only meaningful if no r0 modifiers)
        if r0_weekly_scale is not None and (not self._has_r0_modifiers):
            r0_weekly_scale = pt.as_tensor_variable(r0_weekly_scale)
            if r0_weekly_scale.ndim != 2:
                raise TypeError("r0_weekly_scale must be 2D (week, location)")
            inputs.append(r0_weekly_scale)

        return Apply(self, inputs, [pt.dtensor3(), pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        # Unpack flexible inputs:
        # [mods, pR] (+ optional lambda_ext_loc, mods_loc, r0_weekly_scale)
        mods = inputs[0]
        pR = inputs[1]
        lambda_ext_loc = None
        mods_loc = None
        r0_weekly_scale = None

        # Remaining inputs can be 0–3 items in any of these shapes:
        # 1D (L,) -> lambda_ext_loc (only if has_lambda_ext)
        # 2D (M,L) -> mods_loc
        # 2D (W,L) -> r0_weekly_scale (only if not has_r0_modifiers)
        for extra in inputs[2:]:
            shp = np.shape(extra)
            if extra.ndim == 1:
                lambda_ext_loc = extra
            elif extra.ndim == 2:
                if shp == (len(self.leaf_order), self._L):
                    mods_loc = extra
                elif shp == (self._W, self._L):
                    r0_weekly_scale = extra
                else:
                    # Prefer mods_loc if the first dim matches; else, assume r0_weekly_scale
                    if shp[0] == len(self.leaf_order):
                        mods_loc = extra
                    else:
                        r0_weekly_scale = extra
            else:
                raise TypeError("Unexpected extra input rank.")

        # Validate shapes
        if mods.shape[0] != len(self.leaf_order):
            raise ValueError(f"mods length {mods.shape[0]} != {len(self.leaf_order)}")
        if pR.shape != (self._A, self._L):
            raise ValueError(f"pR shape {pR.shape} != {(self._A, self._L)}")
        if (lambda_ext_loc is not None) and (np.shape(lambda_ext_loc) != (self._L,)):
            raise ValueError(f"lambda_ext_loc shape {np.shape(lambda_ext_loc)} != {(self._L,)}")
        if (mods_loc is not None) and (np.shape(mods_loc) != (len(self.leaf_order), self._L)):
            raise ValueError(f"mods_loc shape {np.shape(mods_loc)} != {(len(self.leaf_order), self._L)}")
        if (r0_weekly_scale is not None) and (np.shape(r0_weekly_scale) != (self._W, self._L)):
            raise ValueError(f"r0_weekly_scale shape {np.shape(r0_weekly_scale)} != {(self._W, self._L)}")

        # 1) Apply modifiers to BASE params (location-wise if mods_loc provided)
        params_base = self._apply_modifiers_locationwise(mods, mods_loc)

        # 1a) Optional weekly r0 scaling (only if r0 has no config modifiers)
        if (r0_weekly_scale is not None) and (not self._has_r0_modifiers):
            self._scale_r0_weekly(params_base, np.asarray(r0_weekly_scale, dtype=np.float64))

        # 1b) Inject exogenous FOI scale (if available & provided)
        lambda_vec = None
        if self._has_lambda_ext and (lambda_ext_loc is not None):
            lambda_vec = np.asarray(lambda_ext_loc, dtype=np.float64)
        self._scale_lambda_ext(params_base, lambda_vec)

        # 1c) BASE → UNIQUE
        params_unique = self.pipe.model.compartments.parse_parameters(
            params_base, self.pipe.param_defs, self.pipe.unique_strings
        )

        # 2) Override initial conditions (S/R only)
        y0 = self._override_ic_with_pR_preserve_others(self.pipe.initial_array, np.asarray(pR, dtype=np.float64))

        # 3) Integrate on the step grid
        total_days = float(self.pipe.T - 1)
        n_steps = max(1, int(round(total_days / self.pipe.dt)))
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)
        res = self._rhs.solve(
            y0=y0.ravel(),
            parameters=params_unique,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-3, atol=1e-6,
        )
        if not res.success:
            raise RuntimeError(f"Integration failed: {res.message}")
        states = res.y.T.reshape(len(t_eval), self.pipe.NC, self._L)

        # 4) Reconstruct transitions and outcomes
        series = {name: np.zeros((n_steps, self._L), dtype=np.float64) for name in self._out_order}
        pc = self.pipe.precomputed
        for i in range(n_steps):
            t0 = t_eval[i]
            param_t_slice = _param_slice_step(params_unique, t0)
            amounts, _ = _compute_amounts_for_step(
                states_current=states[i],
                transitions=self.pipe.transitions,
                proportion_info=self.pipe.proportion_info,
                transition_sum_compartments=self.pipe.transition_sum_compartments,
                param_t_slice=param_t_slice,
                percent_day_away=pc["percent_day_away"],
                prop_who_move=pc["proportion_who_move"],
                mobility_data=pc["mobility_data"],
                mobility_indptr=pc["mobility_data_indices"],
                mobility_indices=pc["mobility_row_indices"],
                population=pc["population"],
                param_expr_lookup=None,
                param_name_to_row=None,
            )
            for name, rows in self._resolve_map.items():
                if rows.size == 0:
                    continue
                inc_vec = amounts[rows, :].sum(axis=0)
                p_vec = self._prob_map.get(name, np.ones(self._L))
                shift = int(self._delay_steps_map.get(name, 0))
                j = i + shift
                if j < n_steps:
                    series[name][j, :] += inc_vec * p_vec

        # Resolve sums/aliases
        unresolved = set(self._sum_map.keys())
        guard = 0
        while unresolved and guard < 10000:
            guard += 1
            progressed = False
            for name in list(unresolved):
                kids = self._sum_map[name]
                if not all(k in series for k in kids):
                    continue
                combined = sum(series[k] for k in kids)
                p_vec = self._prob_map.get(name, np.ones(self._L))
                shift = int(self._delay_steps_map.get(name, 0))
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
        if unresolved:
            raise RuntimeError(f"Unresolved outcomes remain: {unresolved}")

        # 5) Steps → weeks & age aggregation
        assign_steps, n_weeks = _steps_to_weeks_assign(
            start=self.pipe.start_date, T_days=self.pipe.T - 1, T_steps=n_steps, dt_days=float(self.pipe.dt)
        )
        weekly_age = np.zeros((len(self._age_labels), n_weeks, self._L), dtype=np.float64)
        for a_idx, age in enumerate(self._age_labels):
            step_sum = None
            for out_name in self._age_to_names[age]:
                if out_name in series:
                    arr = series[out_name]
                    step_sum = arr if step_sum is None else (step_sum + arr)
            if step_sum is None:
                step_sum = np.zeros((n_steps, self._L), dtype=np.float64)
            W = np.zeros((n_weeks, self._L), dtype=np.float64)
            for w in range(n_weeks):
                mask = (assign_steps == w)
                if mask.any():
                    W[w, :] = step_sum[mask, :].sum(axis=0)
            weekly_age[a_idx, :, :] = W

        # 6) Final S
        last = states[-1]
        S_final = np.zeros((len(self._age_labels), self._L), dtype=np.float64)
        for a_idx, m_s in enumerate(self.age_masks.s_mask_by_age):
            S_final[a_idx, :] = last[m_s, :].sum(axis=0)

        outputs[0][0] = weekly_age
        outputs[1][0] = S_final

    # ---- metadata ------------------------------------------------------

    @property
    def weekly_shape(self) -> tuple[int, int, int]:
        return self._weekly_shape

    @property
    def n_weeks(self) -> int:
        return self._W

    @property
    def age_labels(self) -> tuple[str, ...]:
        return self._age_labels

    @property
    def locations(self) -> int:
        return self._L

    @property
    def has_lambda_ext(self) -> bool:
        return self._has_lambda_ext

    @property
    def has_r0_modifiers(self) -> bool:
        return self._has_r0_modifiers


# ------------------------------ model builder ------------------------------

def build_weekly_model(
    pipeline: WeeklyHospPipeline,
    op: WeeklyHospAndFinalSOp | None = None,
    *,
    y_obs: np.ndarray | None = None,
    use_nb: bool = False,
) -> pm.Model:
    """
    Statistical model (edited to your spec):
      • Per-location modifiers are independent (no pooling): LogNormal for each (modifier, location),
        with expectation equal to the config default.
      • Only one global r0 modifier (shared across locations & weeks). If the config has NO r0
        modifiers, we apply it via a constant (week, location) scale into the Op.
      • pR, lambda_ext_loc, and likelihood unchanged.
    """
    if op is None:
        op = WeeklyHospAndFinalSOp(pipeline)

    A, W, L = op.weekly_shape
    mod_names = tuple(pipeline.modifier_order())
    coords = {
        "age": np.array(op.age_labels, dtype=object),
        "week": np.arange(W),
        "location": np.arange(L),
        "modifier": np.array(mod_names, dtype=object),
    }

    defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, mod_names)

    with pm.Model(coords=coords) as m:
        # ---- MODIFIERS: per-location independent LogNormals (no pooling)
        # Choose a fixed log-sigma; set mu so that E[LogNormal]=default.
        sigma_mod_log = 0.25  # adjust if you want tighter/wider priors
        mu_log_adjusted = np.log(defaults + 1e-12) - 0.5 * (sigma_mod_log**2)
        mods_loc = pm.LogNormal(
            "mods_loc",
            mu=mu_log_adjusted[:, None],
            sigma=sigma_mod_log,
            dims=("modifier", "location"),
        )
        # Shared-vector input to Op is identity (no global sharing in leaves)
        mods_vec = pm.Deterministic("mods", pt.ones((len(mod_names),)), dims=("modifier",))

        # ---- pR (unchanged)
        mu_center = pm.Uniform("mu_center", lower=-1.6, upper=-0.35)
        mu_R_logit = pm.Normal("mu_R_logit", mu=mu_center, sigma=0.5)
        sigma_R_age = pm.HalfNormal("sigma_R_age", sigma=0.5)
        z_age = pm.Normal("z_R_age", 0.0, 1.0, dims=("age",))
        delta_age = pm.Deterministic("delta_R_age", z_age * sigma_R_age, dims=("age",))
        sigma_R_loc = pm.HalfNormal("sigma_R_loc", sigma=0.3)
        z_loc = pm.Normal("z_R_loc", 0.0, 1.0, dims=("location",))
        delta_loc = pm.Deterministic("delta_R_loc", z_loc * sigma_R_loc, dims=("location",))
        sigma_R_resid = pm.HalfNormal("sigma_R_resid", sigma=0.15)
        z_resid = pm.Normal("z_R_resid", 0.0, 1.0, dims=("age", "location"))
        delta_resid = pm.Deterministic("delta_R_resid", z_resid * sigma_R_resid, dims=("age", "location"))
        pR = pm.Deterministic(
            "pR",
            pt.sigmoid(mu_R_logit + delta_age[:, None] + delta_loc[None, :] + delta_resid),
            dims=("age", "location"),
        )

        # ---- lambda_ext_loc (conditional, unchanged)
        lambda_ext_loc = None
        if op.has_lambda_ext:
            mu_lex = pm.Normal("lambda_ext_mu_log", mu=0.0, sigma=0.1)
            sd_lex = pm.HalfNormal("lambda_ext_sigma_log", sigma=0.1)
            lambda_ext_loc = pm.LogNormal("lambda_ext_loc", mu=mu_lex, sigma=sd_lex, dims=("location",))
            pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))

        # ---- Single GLOBAL r0 multiplicative modifier (shared across all seasons & locations)
        # Prior mean 1.0; ~99% mass in [0.8, 1.2] using a scaled Beta with moderate concentration.
        u = pm.Beta("r0_global_unit", alpha=5.0, beta=5.0)
        r0_global = pm.Deterministic("r0_global", 0.8 + 0.4 * u)

        # If the config has NO r0 modifiers, apply r0_global as a constant via Op's r0_weekly_scale.
        r0_weekly_scale_full = None
        if not op.has_r0_modifiers:
            r0_weekly_scale_full = pm.Deterministic(
                "r0_weekly_scale",
                r0_global * pt.ones((W, L)),
                dims=("week", "location"),
            )

        # ---- forward model
        if op.has_lambda_ext and (r0_weekly_scale_full is not None):
            weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc, mods_loc, r0_weekly_scale_full)
        elif op.has_lambda_ext and (r0_weekly_scale_full is None):
            weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc, mods_loc)
        elif (not op.has_lambda_ext) and (r0_weekly_scale_full is not None):
            weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc, r0_weekly_scale_full)
        else:
            weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc)

        weekly = pm.Deterministic("weekly_pred", weekly_pred_t, dims=("age", "week", "location"))
        weekly_sum_age = pm.Deterministic("weekly_pred_sum_age", weekly.sum(axis=0), dims=("week", "location"))
        pm.Deterministic("S_final", S_final_t, dims=("age", "location"))

        # ---- Observation model (unchanged)
        beta0 = pm.Normal("beta0", 0.0, 1.0)
        u_loc = pm.Normal("u_loc", 0.0, 0.25, dims=("location",))
        sigma_week = pm.HalfNormal("sigma_week", 0.2)
        z_week = pm.Normal("z_week", 0.0, 1.0, dims=("week",))
        delta_week = pm.Deterministic("delta_week", z_week * sigma_week - pt.mean(z_week * sigma_week), dims=("week",))
        log_mu_obs = pt.log(weekly_sum_age + 1e-12) + beta0 + u_loc[None, :] + delta_week[:, None]
        mu_obs = pt.exp(log_mu_obs)

        # ---- Likelihood
        if y_obs is not None:
            y_obs = np.asarray(y_obs, dtype=np.float64)
            assert y_obs.shape == (W, L), f"y_obs must be (W, L); got {y_obs.shape}"
            if use_nb:
                alpha_bar = pm.HalfNormal("alpha_bar", 1.0)
                alpha_sd = pm.HalfNormal("alpha_sd", 0.5)
                z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims=("location",))
                alpha_loc = pm.Deterministic("alpha_nb_loc", alpha_bar * pt.exp(alpha_sd * z_alpha), dims=("location",))
                pm.NegativeBinomial("y", mu=mu_obs, alpha=alpha_loc, observed=y_obs, dims=("week", "location"))
            else:
                pm.Poisson("y", mu=mu_obs, observed=y_obs, dims=("week", "location"))
    return m


# ------------------------------ convenience ------------------------------

def build_op_and_model_from_config(
    config_path: str | Path,
    *,
    y_obs: np.ndarray | None = None,
) -> tuple[WeeklyHospPipeline, WeeklyHospAndFinalSOp, pm.Model]:
    pipe = build_pipeline_from_config(config_path, dt_days=0.05)
    op = WeeklyHospAndFinalSOp(pipe)
    model = build_weekly_model(pipe, op, y_obs=y_obs)
    return pipe, op, model
