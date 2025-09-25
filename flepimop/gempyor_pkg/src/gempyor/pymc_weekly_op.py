# src/gempyor/pymc_weekly_op.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
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
        m_age = (m_age := (comp_age_norm == key))
        if not m_age.any():
            m_age = np.ones(NC, dtype=bool)
        masks_s.append(m_age & is_S)
        masks_r.append(m_age & is_R)
        masks_tot.append(m_age)
    return AgeMasks(tuple(pipeline.age_labels), tuple(masks_s), tuple(masks_r), tuple(masks_tot))


# ------------------------------ Op ------------------------------

class WeeklyHospAndFinalSOp(Op):
    """
    Simulator Op returning:
      • weekly  : (age, week, location)
      • S_final : (age, location)

    Robustness:
      • Single fast path only (RK45). If RK45 fails or returns non-finite
        values, the Op returns arrays of NaN with the correct shapes.
    """

    def __init__(
        self,
        pipeline: WeeklyHospPipeline,
        *,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        allow_r0_weekly_scale_with_modifiers: bool = False,
    ):
        super().__init__()
        self.pipe = pipeline
        self.leaf_order = pipeline.modifier_order()
        self.age_masks = _build_age_masks(pipeline)

        self._rtol = float(rtol)
        self._atol = float(atol)
        self._allow_r0_weekly_scale_with_modifiers = bool(allow_r0_weekly_scale_with_modifiers)

        defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, self.leaf_order)
        wk, _, _ = pipeline.evaluate(defaults)
        self._weekly_shape = wk.shape
        self._A, self._W, self._L = self._weekly_shape
        self._age_to_names = dict(pipeline.age_to_names)
        self._age_labels = tuple(pipeline.age_labels)

        # Simple evenly spaced week centers in days (used by model builder)
        self._week_centers = np.arange(self._W, dtype=float) * 7.0

        # Prefer outcome maps from pipeline
        if all(
            hasattr(pipeline, attr)
            for attr in ("_out_resolve_rows", "_out_prob_map", "_out_delay_steps", "_out_sum_map", "_out_all_names")
        ):
            self._resolve_map     = pipeline._out_resolve_rows
            self._prob_map        = pipeline._out_prob_map
            self._delay_steps_map = pipeline._out_delay_steps
            self._sum_map         = pipeline._out_sum_map
            self._out_order       = pipeline._out_all_names
        else:
            import warnings
            warnings.warn(
                "WeeklyHospAndFinalSOp: falling back to local outcome compilation. "
                "Per-subpopulation relative_probability weights from the outcomes "
                "config file will NOT be applied unless the pipeline precompiled them.",
                RuntimeWarning,
            )
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

        self._rhs = RHSfactory(precomputed=self.pipe.precomputed, param_time_mode="step")

        self._param_name_to_idx: dict[str, int] = {}
        if hasattr(self.pipe, "param_name_to_idx"):
            self._param_name_to_idx = dict(self.pipe.param_name_to_idx)
        elif hasattr(self.pipe, "mod_applier"):
            self._param_name_to_idx = dict(getattr(self.pipe, "mod_applier").param_name_to_idx)

        self._has_lambda_ext = ("lambda_ext" in self._param_name_to_idx)
        self._has_r0 = ("r0" in self._param_name_to_idx)

        self._r0_modifier_leaves: tuple[str, ...] = tuple()
        self._has_r0_modifiers = self._detect_r0_modifiers_authoritative()
        if not self._r0_modifier_leaves:
            try:
                y_names = self._find_r0_modifier_names_from_yaml()
                if y_names:
                    self._r0_modifier_leaves = tuple(sorted(set(y_names)))
            except Exception:
                pass

    # ---- helpers --------------------------------------------------------

    def _leaf_rows_map(self):
        applier = getattr(self.pipe, "mod_applier", None)
        if applier is None:
            return None, None
        l2r = getattr(applier, "leaf_to_param_rows", None)
        pmap = getattr(self.pipe, "param_name_to_idx", None)
        if l2r is None or pmap is None:
            return None, None
        if not isinstance(l2r, dict) or not isinstance(pmap, dict):
            return None, None
        return l2r, pmap

    def _detect_r0_modifiers_authoritative(self) -> bool:
        try:
            l2r, pmap = self._leaf_rows_map()
            if l2r is None or pmap is None or "r0" not in pmap:
                return True
            r0_row = int(pmap["r0"])
            hit: list[str] = []
            for leaf, rows in l2r.items():
                try:
                    rows_list = list(rows) if not isinstance(rows, (list, tuple, set)) else list(rows)
                except Exception:
                    rows_list = []
                if r0_row in rows_list:
                    hit.append(str(leaf))
            if hit:
                self._r0_modifier_leaves = tuple(sorted(set(hit)))
                return True
            return False
        except Exception:
            return True

    def _find_r0_modifier_names_from_yaml(self) -> list[str]:
        names: list[str] = []
        conf = confuse.Configuration("WeeklyHospPipelineDefaults", __name__)
        conf.set_file(str(self.pipe.config_path))
        mods = conf["seir_modifiers"]["modifiers"].get()

        def _walk(name: str, spec) -> None:
            if not isinstance(spec, dict):
                return
            method = str(spec.get("method", "")).lower()
            if method == "stackedmodifier":
                for ch in spec.get("modifiers", []) or []:
                    csp = mods.get(ch, {})
                    _walk(str(ch), csp)
            else:
                param = str(spec.get("parameter", "")).lower()
                if param == "r0":
                    names.append(name)

        for nm, sp in dict(mods).items():
            _walk(str(nm), sp)
        return names

    def _get_param_row(self, name: str) -> int:
        try:
            return int(self._param_name_to_idx[name])
        except Exception as e:
            raise KeyError(f"Parameter row '{name}' not found in pipeline mapping.") from e

    def _interp_weeks_to_days(self, weekly_scale: np.ndarray, T_days: int) -> np.ndarray:
        W, L = weekly_scale.shape
        if W <= 0:
            return np.ones((T_days, L), dtype=np.float64)
        x = self._week_centers[:W]
        t = np.arange(T_days, dtype=np.float64)
        daily = np.empty((T_days, L), dtype=np.float64)
        for j in range(L):
            y = np.asarray(weekly_scale[:, j], dtype=np.float64)
            daily[:, j] = np.interp(t, x, y, left=float(y[0]), right=float(y[-1]))
        return daily

    def _scale_lambda_ext(self, params_base: np.ndarray, lambda_ext_loc: np.ndarray | None) -> None:
        if (not self._has_lambda_ext) or (lambda_ext_loc is None):
            return
        pidx = self._get_param_row("lambda_ext")
        params_base[pidx, :, :] *= lambda_ext_loc[None, :]

    def _scale_r0_weekly(self, params_base: np.ndarray, r0_weekly_scale: np.ndarray | None) -> None:
        if (not self._has_r0) or (r0_weekly_scale is None):
            return
        if self._has_r0_modifiers and (not self._allow_r0_weekly_scale_with_modifiers):
            return
        pidx = self._get_param_row("r0")
        W = int(self._W)
        expected = (W, self._L)
        if r0_weekly_scale.shape != expected:
            raise ValueError(f"r0_weekly_scale shape {r0_weekly_scale.shape} != {expected}")
        T_days = params_base.shape[1]
        daily_scale = self._interp_weeks_to_days(np.asarray(r0_weekly_scale, dtype=np.float64), T_days)
        params_base[pidx, :, :] *= daily_scale

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

    def _apply_modifiers_locationwise(self, mods_shared: np.ndarray, mods_loc: np.ndarray | None) -> np.ndarray:
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

    def _precheck(self, params_unique: np.ndarray, y0: np.ndarray) -> bool:
        if (params_unique is None) or (y0 is None):
            return False
        if not np.isfinite(params_unique).all():
            return False
        if not np.isfinite(y0).all():
            return False
        return True

    def _solve_fast(self, y0_vec: np.ndarray, params_unique: np.ndarray, t_eval: np.ndarray):
        res = self._rhs.solve(
            y0=y0_vec,
            parameters=params_unique,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            t_eval=t_eval,
            method="RK45",
            rtol=self._rtol,
            atol=self._atol,
        )
        ok = bool(getattr(res, "success", False)) and np.isfinite(res.y).all()
        return res if ok else None

    # ---- Op API --------------------------------------------------------

    def make_node(self, mods, pR, lambda_ext_loc=None, mods_loc=None, r0_weekly_scale=None):
        mods = pt.as_tensor_variable(mods)
        pR = pt.as_tensor_variable(pR)
        if mods.ndim != 1:
            raise TypeError("mods must be 1D")
        if pR.ndim != 2:
            raise TypeError("pR must be 2D (age, location)")

        inputs = [mods, pR]

        if (lambda_ext_loc is not None) and self._has_lambda_ext:
            lam_node = pt.as_tensor_variable(lambda_ext_loc)
            if lam_node.ndim != 1:
                raise TypeError("lambda_ext_loc must be 1D (location,)")
            inputs.append(lam_node)

        if mods_loc is not None:
            mods_loc = pt.as_tensor_variable(mods_loc)
            if mods_loc.ndim != 2:
                raise TypeError("mods_loc must be 2D (modifier, location)")
            inputs.append(mods_loc)

        if r0_weekly_scale is not None and ((not self._has_r0_modifiers) or self._allow_r0_weekly_scale_with_modifiers):
            r0_weekly_scale = pt.as_tensor_variable(r0_weekly_scale)
            if r0_weekly_scale.ndim != 2:
                raise TypeError("r0_weekly_scale must be 2D (week, location)")
            inputs.append(r0_weekly_scale)

        return Apply(self, inputs, [pt.dtensor3(), pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        mods = inputs[0]
        pR_in = inputs[1]  # shape (A, L)
        lambda_ext_loc = None
        mods_loc = None
        r0_weekly_scale = None

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
                    if shp[0] == len(self.leaf_order):
                        mods_loc = extra
                    else:
                        r0_weekly_scale = extra
            else:
                raise TypeError("Unexpected extra input rank.")

        if mods.shape[0] != len(self.leaf_order):
            raise ValueError(f"mods length {mods.shape[0]} != {len(self.leaf_order)}")

        pR_slice = np.asarray(pR_in, dtype=np.float64)
        if pR_slice.shape != (self._A, self._L):
            raise ValueError(f"pR shape {pR_slice.shape} != {(self._A, self._L)}")

        if (lambda_ext_loc is not None) and (np.shape(lambda_ext_loc) != (self._L,)):
            raise ValueError(f"lambda_ext_loc shape {np.shape(lambda_ext_loc)} != {(self._L,)}")
        if (mods_loc is not None) and (np.shape(mods_loc) != (len(self.leaf_order), self._L)):
            raise ValueError(f"mods_loc shape {np.shape(mods_loc)} != {(len(self.leaf_order), self._L)}")
        if (r0_weekly_scale is not None) and (np.shape(r0_weekly_scale) != (self._W, self._L)):
            raise ValueError(f"r0_weekly_scale shape {np.shape(r0_weekly_scale)} != {(self._W, self._L)}")

        # -------- Shared precomputation --------
        params_base = self._apply_modifiers_locationwise(mods, mods_loc)

        if r0_weekly_scale is not None:
            self._scale_r0_weekly(params_base, np.asarray(r0_weekly_scale, dtype=np.float64))

        if self._has_lambda_ext and (lambda_ext_loc is not None):
            self._scale_lambda_ext(params_base, np.asarray(lambda_ext_loc, dtype=np.float64))

        params_unique = self.pipe.model.compartments.parse_parameters(
            params_base, self.pipe.param_defs, self.pipe.unique_strings
        )

        total_days = float(self.pipe.T - 1)
        n_steps = max(1, int(round(total_days / self.pipe.dt)))
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

        assign_steps, n_weeks = _steps_to_weeks_assign(
            start=self.pipe.start_date, T_days=self.pipe.T - 1, T_steps=n_steps, dt_days=float(self.pipe.dt)
        )

        pc = self.pipe.precomputed
        mob_data = pc["mobility_data"]
        mob_indptr = pc["mobility_data_indices"]
        mob_indices = pc["mobility_row_indices"]

        # Single scenario
        y0 = self._override_ic_with_pR_preserve_others(self.pipe.initial_array, pR_slice)

        if not self._precheck(params_unique, y0):
            weekly_age_nan = np.full((len(self._age_labels), n_weeks, self._L), np.nan, dtype=np.float64)
            S_final_nan = np.full((len(self._age_labels), self._L), np.nan, dtype=np.float64)
            outputs[0][0] = weekly_age_nan
            outputs[1][0] = S_final_nan
            return

        res = self._solve_fast(y0.ravel(), params_unique, t_eval)
        if res is None:
            weekly_age_nan = np.full((len(self._age_labels), n_weeks, self._L), np.nan, dtype=np.float64)
            S_final_nan = np.full((len(self._age_labels), self._L), np.nan, dtype=np.float64)
            outputs[0][0] = weekly_age_nan
            outputs[1][0] = S_final_nan
            return

        states = res.y.T.reshape(len(t_eval), self.pipe.NC, self._L)

        # Accumulate flow series
        series = {name: np.zeros((n_steps, self._L), dtype=np.float64) for name in self._out_order}
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
                mobility_data=mob_data,
                mobility_indptr=mob_indptr,
                mobility_indices=mob_indices,
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

        # Resolve sums with delays
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
            weekly_age_nan = np.full((len(self._age_labels), n_weeks, self._L), np.nan, dtype=np.float64)
            S_final_nan = np.full((len(self._age_labels), self._L), np.nan, dtype=np.float64)
            outputs[0][0] = weekly_age_nan
            outputs[1][0] = S_final_nan
            return

        # Weekly aggregation per age
        weekly_age = np.zeros((len(self._age_labels), n_weeks, self._L), dtype=np.float64)
        for a_idx, age in enumerate(self._age_labels):
            step_sum = None
            for out_name in self._age_to_names[age]:
                if out_name in series:
                    arr = series[out_name]
                    step_sum = arr if step_sum is None else (step_sum + arr)
            if step_sum is None:
                step_sum = np.zeros((n_steps, self._L), dtype=np.float64)
            Wk = np.zeros((n_weeks, self._L), dtype=np.float64)
            for w in range(n_weeks):
                mask = (assign_steps == w)
                if mask.any():
                    Wk[w, :] = step_sum[mask, :].sum(axis=0)
            weekly_age[a_idx, :, :] = Wk

        # Final S by age
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

    @property
    def r0_modifier_leaves(self) -> tuple[str, ...]:
        return self._r0_modifier_leaves


# ------------------------------ Fourier helpers ------------------------------

def _fourier_design(t: np.ndarray, K: int, period_days: float) -> np.ndarray:
    """
    Build design matrix for times t (days) with columns [1, cos(2πk t/P), sin(2πk t/P)] for k=1..K.
    Returns shape (len(t), 1 + 2K).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    M = 1 + 2 * K
    X = np.empty((t.size, M), dtype=float)
    X[:, 0] = 1.0
    if K == 0:
        return X
    w = (2.0 * np.pi / float(period_days))
    col = 1
    for k in range(1, K + 1):
        ang = w * k * t
        X[:, col] = np.cos(ang); col += 1
        X[:, col] = np.sin(ang); col += 1
    return X


def _ls_init_fourier(
    r0_daily: np.ndarray,
    t_daily: np.ndarray,
    K: int,
    period_days: float,
) -> tuple[np.ndarray, float]:
    """
    Quick LS fit on log(r0_daily) to get theta_hat and residual std.
    Returns (theta_hat[M], resid_std).
    """
    y = np.log(np.clip(np.asarray(r0_daily, dtype=float).reshape(-1), 1e-12, np.inf))
    X = _fourier_design(np.asarray(t_daily, dtype=float), K, period_days)
    theta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta_hat
    resid_std = float(np.sqrt(np.maximum(np.mean(resid**2), 1e-12)))
    return theta_hat, resid_std


# ------------------------------ model builder ------------------------------

def build_weekly_model(
    pipeline: WeeklyHospPipeline,
    op: WeeklyHospAndFinalSOp | None = None,
    *,
    y_obs: np.ndarray | None = None,
    use_nb: bool = False,
    fourier_harmonics: int = 6,
    fourier_period_days: float = 365.25,
    obs_weeks: np.ndarray | None = None,
) -> pm.Model:
    """
    Simplified weekly hospitalization model:

      • r0(t) = exp( X_fourier @ theta - mean_w(X@theta) ), weekly (Fourier-only)
      • r0 modifiers are neutralized (others remain active)
      • No weekly random walk, no lag kernel, no censoring/endpoint penalties
      • Likelihood on weekly totals (Poisson by default; NB if use_nb=True)
    """
    if op is None:
        op = WeeklyHospAndFinalSOp(pipeline)

    # Allow r0 weekly scaling to coexist with modifiers
    try:
        setattr(op, "_allow_r0_weekly_scale_with_modifiers", True)
    except Exception:
        pass

    A, W, L = op.weekly_shape
    mod_names = tuple(pipeline.modifier_order())

    # ---- Coordinates
    coeff_names = ["c0"] + [f"{s}{k}" for k in range(1, max(0, fourier_harmonics) + 1) for s in ("cos", "sin")]
    coords = {
        "age": np.array(op.age_labels, dtype=object),
        "week": np.arange(W),
        "location": np.arange(L),
        "modifier": np.array(mod_names, dtype=object),
        "fourier_coeff": np.array(coeff_names, dtype=object),
    }

    # ---- Optional observed-week slicing
    obs_weeks_arr = None
    if obs_weeks is not None:
        obs_weeks_arr = np.asarray(obs_weeks, dtype=int).reshape(-1)
        if obs_weeks_arr.size == 0:
            obs_weeks_arr = None
        else:
            if np.any((obs_weeks_arr < 0) | (obs_weeks_arr >= W)):
                raise ValueError(f"obs_weeks contains out-of-range indices for W={W}: {obs_weeks_arr}")
            coords["obs_week"] = obs_weeks_arr

    # ---- Population (only for sanity/scale if needed)
    try:
        pop_arr = np.asarray(pipeline.precomputed.get("population", None))
        pop_loc = pop_arr if pop_arr is not None and pop_arr.ndim == 1 else pop_arr.sum(axis=0)
    except Exception:
        pop_loc = None
    if pop_loc is None:
        try:
            pop_loc = np.asarray(pipeline.initial_array, dtype=np.float64).sum(axis=0)
        except Exception:
            pop_loc = np.ones(L, dtype=np.float64)
    pop_loc = np.clip(np.asarray(pop_loc, dtype=np.float64).reshape(L), 1.0, np.inf)

    # ---- YAML defaults for modifiers (center of priors)
    defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, mod_names)

    # ---- Precompute Fourier LS init per location for r0(t)
    K = int(max(0, fourier_harmonics))
    period_days = float(fourier_period_days)

    # Locate r0 row in BASE params to build LS initializations
    param_names = np.array(list(pipeline.param_defs.keys()))
    if "r0" in param_names:
        r0_idx = int(np.where(param_names == "r0")[0][0])
    else:
        aliases = ["R0", "r_0", "basic_reproduction_number"]
        found = [nm for nm in aliases if nm in param_names]
        if not found:
            raise RuntimeError("Could not locate 'r0' parameter row in base params.")
        r0_idx = int(np.where(param_names == found[0])[0][0])

    base = pipeline.base_params  # (P, T_days, L)
    T_days = base.shape[1]
    t_days = np.arange(T_days, dtype=float)
    week_centers = np.asarray(getattr(op, "_week_centers", np.arange(W) * 7.0), dtype=float)[:W]
    Xw_shared = _fourier_design(week_centers, K, period_days)  # (W, 1+2K)

    M = 1 + 2 * K
    theta_hat_loc = np.zeros((M, L), dtype=float)
    resid_std_loc = np.zeros((L,), dtype=float)
    for l in range(L):
        r0_daily = base[r0_idx, :, l].astype(float)
        th, rs = _ls_init_fourier(r0_daily, t_days, K, period_days)
        theta_hat_loc[:, l] = th
        resid_std_loc[l] = rs

    # ---- Identify which modifier leaves touch r0; we'll neutralize them
    r0_leaves = set(op.r0_modifier_leaves or ())
    r0_mask_vec = np.array([nm in r0_leaves for nm in mod_names], dtype=bool)  # (M_mod,)
    r0_mask_mat = np.tile(r0_mask_vec[:, None], (1, L))  # (M_mod, L)

    with pm.Model(coords=coords) as m:
        # ---------- Non-r0 Modifiers (per location), centered at YAML defaults ----------
        sigma_mod_log = 0.25
        mu_log_base = np.log(defaults + 1e-12) - 0.5 * (sigma_mod_log ** 2)

        if L == 1:
            mods_mu_log_loc = pm.Normal("mods_mu_log_loc", mu=mu_log_base[:, None], sigma=0.20, dims=("modifier", "location"))
            mods_loc_raw = pm.LogNormal("mods_loc_raw", mu=mods_mu_log_loc, sigma=sigma_mod_log, dims=("modifier", "location"))
        else:
            mods_mu_log_ls, mods_loc_ls = [], []
            for l in range(L):
                mods_mu_log_l = pm.Normal(f"mods_mu_log_loc_l{l}", mu=mu_log_base, sigma=0.20, dims=("modifier",))
                mods_l = pm.LogNormal(f"mods_loc_l{l}", mu=mods_mu_log_l, sigma=sigma_mod_log, dims=("modifier",))
                mods_mu_log_ls.append(mods_mu_log_l)
                mods_loc_ls.append(mods_l)
            pm.Deterministic("mods_mu_log_loc", pt.stack(mods_mu_log_ls, axis=1), dims=("modifier", "location"))
            mods_loc_raw = pm.Deterministic("mods_loc_raw", pt.stack(mods_loc_ls, axis=1), dims=("modifier", "location"))

        # Neutralize r0-related leaves to 1.0; keep others as RVs
        mask = pt.as_tensor_variable(r0_mask_mat)  # (M_mod, L)
        ones = pt.ones_like(mods_loc_raw)
        mods_loc = pm.Deterministic("mods_loc",
                                    pt.where(mask, ones, mods_loc_raw),
                                    dims=("modifier", "location"))

        # Shared scalar multiplier vector (kept at ones)
        mods_vec = pm.Deterministic("mods", pt.ones((len(mod_names),)), dims=("modifier",))

        # ---------- lambda_ext per location (if present) ----------
        if op.has_lambda_ext:
            if L == 1:
                lambda_ext_loc = pm.LogNormal("lambda_ext_loc", mu=0.0, sigma=0.18, dims=("location",))
                pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))
            else:
                lambda_ls = [pm.LogNormal(f"lambda_ext_loc_l{l}", mu=0.0, sigma=0.18) for l in range(L)]
                lambda_ext_loc = pm.Deterministic("lambda_ext_loc", pt.stack(lambda_ls, axis=0), dims=("location",))
                pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))
        else:
            lambda_ext_loc = None

        # ---------- Age scaling ----------
        if L == 1:
            sigma_age_loc = pm.HalfNormal("sigma_age_loc", 0.10, dims=("location",))
            z_age_loc = pm.Normal("z_age_loc", 0.0, 1.0, dims=("age", "location"))
            theta_age_loc = pm.Deterministic("theta_age_loc", z_age_loc * sigma_age_loc[None, :], dims=("age", "location"))
        else:
            sigma_age_ls = [pm.HalfNormal(f"sigma_age_loc_l{l}", 0.10) for l in range(L)]
            z_age_ls = [pm.Normal(f"z_age_loc_l{l}", 0.0, 1.0, dims=("age",)) for l in range(L)]
            theta_age_loc = pm.Deterministic("theta_age_loc",
                                             pt.stack([z_age_ls[l] * sigma_age_ls[l] for l in range(L)], axis=1),
                                             dims=("age", "location"))
        age_scale = pm.Deterministic("age_scale", pt.exp(theta_age_loc), dims=("age", "location"))

        # ---------- Intercept ----------
        if L == 1:
            beta0_loc = pm.Normal("beta0_loc", 0.0, 0.2, dims=("location",))
        else:
            beta0_ls = [pm.Normal(f"beta0_loc_l{l}", 0.0, 0.2) for l in range(L)]
            beta0_loc = pm.Deterministic("beta0_loc", pt.stack(beta0_ls, axis=0), dims=("location",))

        # ---------- Initial immunity (per age, per location) ----------
        if L == 1:
            pR = pm.Beta("pR", alpha=1.0, beta=1.0, dims=("age", "location"))
        else:
            pR_ls = [pm.Beta(f"pR_l{l}", alpha=1.0, beta=1.0, dims=("age",)) for l in range(L)]
            pR = pm.Deterministic("pR", pt.stack(pR_ls, axis=1), dims=("age", "location"))

        # ---------- Fourier r0 weekly scale ----------
        tau = pm.HalfNormal("r0_fourier_tau", sigma=0.7)
        Xw = pt.as_tensor_variable(Xw_shared)  # (W, M)

        if L == 1:
            base_sigma = float(np.minimum(np.maximum(resid_std_loc[0], 0.05), 0.25))
            theta = pm.Normal("r0_fourier_coef_loc",
                              mu=theta_hat_loc[:, 0],
                              sigma=base_sigma * tau,
                              dims=("fourier_coeff",))
            f_w = Xw @ theta
            f_w_center = f_w - pt.mean(f_w)
            r0_weekly_scale = pm.Deterministic("r0_weekly_scale", pt.exp(f_w_center)[:, None], dims=("week", "location"))
        else:
            theta_ls = []
            for l in range(L):
                base_sigma = float(np.minimum(np.maximum(resid_std_loc[l], 0.05), 0.25))
                theta_ls.append(pm.Normal(f"r0_fourier_coef_loc_l{l}",
                                          mu=theta_hat_loc[:, l],
                                          sigma=base_sigma * tau,
                                          dims=("fourier_coeff",)))
            f_w = pt.stack([Xw @ th for th in theta_ls], axis=1)   # (W,L)
            f_w_center = f_w - pt.mean(f_w, axis=0, keepdims=True)
            r0_weekly_scale = pm.Deterministic("r0_weekly_scale", pt.exp(f_w_center), dims=("week", "location"))

        # ---------- Forward model (no lag/RW) ----------
        if op.has_lambda_ext:
            weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc=lambda_ext_loc,
                                          mods_loc=mods_loc, r0_weekly_scale=r0_weekly_scale)
        else:
            weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc=mods_loc, r0_weekly_scale=r0_weekly_scale)

        weekly = pm.Deterministic("weekly_pred", weekly_pred_t, dims=("age", "week", "location"))
        weekly_scaled = pm.Deterministic("weekly_pred_scaled", weekly * age_scale[:, None, :],
                                         dims=("age", "week", "location"))
        weekly_sum_age = pm.Deterministic("weekly_pred_sum_age", weekly_scaled.sum(axis=0),
                                          dims=("week", "location"))
        pm.Deterministic("S_final", S_final_t, dims=("age", "location"))

        # ---------- Linear predictor ----------
        mu_full = pt.exp(beta0_loc[None, :]) * weekly_sum_age  # (W, L)

        # ---------- Likelihood ----------
        if y_obs is not None:
            y_np = np.asarray(y_obs, dtype=np.float64)
            if obs_weeks_arr is None:
                assert y_np.shape == (W, L), f"y_obs must be shape (W, L); got {y_np.shape}"
                if use_nb:
                    alpha_nb_loc = pm.LogNormal("alpha_nb_loc", mu=np.log(20.0), sigma=0.3, dims=("location",))
                    pm.NegativeBinomial("y", mu=mu_full, alpha=alpha_nb_loc, observed=y_np, dims=("week", "location"))
                else:
                    pm.Poisson("y", mu=mu_full, observed=y_np, dims=("week", "location"))
            else:
                W_obs = int(obs_weeks_arr.size)
                assert y_np.shape == (W_obs, L), f"y_obs must be shape (len(obs_weeks), L); got {y_np.shape}"
                mu_obs_slice = pt.take(mu_full, obs_weeks_arr, axis=0)
                if use_nb:
                    alpha_nb_loc = pm.LogNormal("alpha_nb_loc", mu=np.log(20.0), sigma=0.3, dims=("location",))
                    pm.NegativeBinomial("y", mu=mu_obs_slice, alpha=alpha_nb_loc,
                                        observed=y_np, dims=("obs_week", "location"))
                else:
                    pm.Poisson("y", mu=mu_obs_slice, observed=y_np, dims=("obs_week", "location"))

        assert isinstance(m, pm.Model)
        return m


# ------------------------------ convenience ------------------------------

def build_op_and_model_from_config(
    config_path: str | Path,
    *,
    y_obs: np.ndarray | None = None,
) -> tuple[WeeklyHospPipeline, WeeklyHospAndFinalSOp, pm.Model]:
    pipe = build_pipeline_from_config(config_path, dt_days=1.0)
    op = WeeklyHospAndFinalSOp(pipe)
    model = build_weekly_model(pipe, op, y_obs=y_obs)
    return pipe, op, model
