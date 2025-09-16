# src/gempyor/pymc_weekly_op.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Iterable, List, Dict, Any
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
        m_age = (m_age := (comp_age_norm == key))
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

    Robustness:
      • Single fast path only (RK45). If RK45 fails or returns non-finite
        values, the Op returns arrays of NaN with the correct shapes.
    """

    def __init__(
        self,
        pipeline: WeeklyHospPipeline,
        *,
        fast_mode: bool = False,
        disable_mobility: bool = False,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        allow_r0_weekly_scale_with_modifiers: bool = False,
        smooth_r0_days: int = 5,
    ):
        super().__init__()
        self.pipe = pipeline
        self.leaf_order = pipeline.modifier_order()
        self.age_masks = _build_age_masks(pipeline)

        self._fast_mode = bool(fast_mode)
        self._disable_mobility = bool(disable_mobility)
        self._rtol = 5e-3 if fast_mode else float(rtol)
        self._atol = 5e-5 if fast_mode else float(atol)
        self._allow_r0_weekly_scale_with_modifiers = bool(allow_r0_weekly_scale_with_modifiers)
        self._smooth_r0_days = int(max(0, smooth_r0_days))

        defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, self.leaf_order)
        wk, _, _ = pipeline.evaluate(defaults)
        self._weekly_shape = wk.shape
        self._A, self._W, self._L = self._weekly_shape
        self._age_to_names = dict(pipeline.age_to_names)
        self._age_labels = tuple(pipeline.age_labels)

        self._day_to_week, self._n_weeks = _mmwr_assign_from(pipeline.start_date, pipeline.T)
        self._week_centers = self._compute_week_centers(self._day_to_week)

        # Align to model's weekly array
        self._n_weeks = self._W

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
            self._param_name_to_idx = dict(getattr(self.pipe.mod_applier, "param_name_to_idx", {}))

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

    def _compute_week_centers(self, day_to_week: np.ndarray) -> np.ndarray:
        W = int(day_to_week.max()) + 1 if day_to_week.size else 0
        centers = np.zeros(W, dtype=np.float64)
        for w in range(W):
            idx = np.flatnonzero(day_to_week == w)
            if idx.size == 0:
                centers[w] = (0.0 if w == 0 else centers[w - 1] + 7.0)
            else:
                centers[w] = 0.5 * (float(idx[0]) + float(idx[-1]))
        return centers

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

    def _apply_log_hann_smoothing_to_r0(self, params_base: np.ndarray) -> None:
        if (not self._has_r0) or (self._smooth_r0_days < 3):
            return
        N = int(self._smooth_r0_days)
        if N < 3:
            return
        k = np.hanning(N).astype(np.float64)
        if not np.isfinite(k).all() or k.sum() <= 0:
            return
        k /= k.sum()
        pidx = self._get_param_row("r0")
        T_days = params_base.shape[1]
        ones = np.ones(T_days, dtype=np.float64)
        denom = np.convolve(ones, k, mode="same")
        denom = np.maximum(denom, 1e-12)
        for j in range(self._L):
            r = np.asarray(params_base[pidx, :, j], dtype=np.float64)
            r = np.clip(r, 1e-12, np.inf)
            lr = np.log(r)
            num = np.convolve(lr, k, mode="same")
            sm = np.exp(num / denom)
            params_base[pidx, :, j] = sm

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
        pR = inputs[1]
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
        if pR.shape != (self._A, self._L):
            raise ValueError(f"pR shape {pR.shape} != {(self._A, self._L)}")
        if (lambda_ext_loc is not None) and (np.shape(lambda_ext_loc) != (self._L,)):
            raise ValueError(f"lambda_ext_loc shape {np.shape(lambda_ext_loc)} != {(self._L,)}")
        if (mods_loc is not None) and (np.shape(mods_loc) != (len(self.leaf_order), self._L)):
            raise ValueError(f"mods_loc shape {np.shape(mods_loc)} != {(len(self.leaf_order), self._L)}")
        if (r0_weekly_scale is not None) and (np.shape(r0_weekly_scale) != (self._W, self._L)):
            raise ValueError(f"r0_weekly_scale shape {np.shape(r0_weekly_scale)} != {(self._W, self._L)}")

        params_base = self._apply_modifiers_locationwise(mods, mods_loc)

        if r0_weekly_scale is not None:
            self._scale_r0_weekly(params_base, np.asarray(r0_weekly_scale, dtype=np.float64))

        if self._has_lambda_ext and (lambda_ext_loc is not None):
            self._scale_lambda_ext(params_base, np.asarray(lambda_ext_loc, dtype=np.float64))

        self._apply_log_hann_smoothing_to_r0(params_base)

        params_unique = self.pipe.model.compartments.parse_parameters(
            params_base, self.pipe.param_defs, self.pipe.unique_strings
        )

        y0 = self._override_ic_with_pR_preserve_others(self.pipe.initial_array, np.asarray(pR, dtype=np.float64))

        total_days = float(self.pipe.T - 1)
        n_steps = max(1, int(round(total_days / self.pipe.dt)))
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

        if not self._precheck(params_unique, y0):
            assign_steps, n_weeks = _steps_to_weeks_assign(
                start=self.pipe.start_date, T_days=self.pipe.T - 1, T_steps=n_steps, dt_days=float(self.pipe.dt)
            )
            weekly_age = np.full((len(self._age_labels), n_weeks, self._L), np.nan, dtype=np.float64)
            S_final = np.full((len(self._age_labels), self._L), np.nan, dtype=np.float64)
            outputs[0][0] = weekly_age
            outputs[1][0] = S_final
            return

        res = self._solve_fast(y0.ravel(), params_unique, t_eval)
        if res is None:
            assign_steps, n_weeks = _steps_to_weeks_assign(
                start=self.pipe.start_date, T_days=self.pipe.T - 1, T_steps=n_steps, dt_days=float(self.pipe.dt)
            )
            weekly_age = np.full((len(self._age_labels), n_weeks, self._L), np.nan, dtype=np.float64)
            S_final = np.full((len(self._age_labels), self._L), np.nan, dtype=np.float64)
            outputs[0][0] = weekly_age
            outputs[1][0] = S_final
            return

        states = res.y.T.reshape(len(t_eval), self.pipe.NC, self._L)

        series = {name: np.zeros((n_steps, self._L), dtype=np.float64) for name in self._out_order}
        pc = self.pipe.precomputed

        if self._disable_mobility:
            mob_data = np.zeros_like(pc["mobility_data"])
            mob_indptr = pc["mobility_data_indices"]
            mob_indices = pc["mobility_row_indices"]
        else:
            mob_data = pc["mobility_data"]
            mob_indptr = pc["mobility_data_indices"]
            mob_indices = pc["mobility_row_indices"]

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
            assign_steps, n_weeks = _steps_to_weeks_assign(
                start=self.pipe.start_date, T_days=self.pipe.T - 1, T_steps=n_steps, dt_days=float(self.pipe.dt)
            )
            weekly_age = np.full((len(self._age_labels), n_weeks, self._L), np.nan, dtype=np.float64)
            S_final = np.full((len(self._age_labels), self._L), np.nan, dtype=np.float64)
            outputs[0][0] = weekly_age
            outputs[1][0] = S_final
            return

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


# ------------------------------ Fourier helpers (new) ------------------------------

def _fourier_names(K: int) -> list[str]:
    names = ["c0"]
    for k in range(1, K + 1):
        names.append(f"cos{k}")
        names.append(f"sin{k}")
    return names

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
    # Least squares
    theta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta_hat
    resid_std = float(np.sqrt(np.maximum(np.mean(resid**2), 1e-12)))
    return theta_hat, resid_std

def _weekly_from_theta(theta: np.ndarray, week_centers: np.ndarray, K: int, period_days: float) -> np.ndarray:
    """
    Evaluate weekly f(w) = Xw @ theta, then return exp(f - mean(f)) as scale (unit geometric mean).
    """
    Xw = _fourier_design(np.asarray(week_centers, dtype=float), K, period_days)
    fw = Xw @ theta.reshape(-1)
    fw_center = fw - np.mean(fw)
    return np.exp(fw_center).astype(float)  # (W,)

# ------------------------------ model builder ------------------------------

def build_weekly_model(
    pipeline: WeeklyHospPipeline,
    op: WeeklyHospAndFinalSOp | None = None,
    *,
    y_obs: np.ndarray | None = None,
    use_nb: bool = False,
    force_r0_weekly_scale: bool = False,
    # ---- NEW Fourier knobs ----
    force_r0_fourier_scale: bool = False,
    fourier_harmonics: int = 3,
    fourier_period_days: float = 365.25,
) -> pm.Model:
    """
    If L == 1: build the original vectorized model.

    If L > 1: create *per-location* random variables (names end with `_l{idx}`),
    then stack them back into tensors and expose the usual Deterministic names.

    New:
      - `force_r0_fourier_scale`: use a Fourier series to generate r0 weekly multiplicative scale.
        Mutually exclusive with `force_r0_weekly_scale`.
    """
    if op is None:
        op = WeeklyHospAndFinalSOp(pipeline)

    # ---- Mutual exclusion guard ----
    if force_r0_weekly_scale and force_r0_fourier_scale:
        raise ValueError("Choose at most one of {force_r0_weekly_scale, force_r0_fourier_scale}.")

    # If either path is chosen, make sure Op allows weekly scaling alongside modifiers
    if force_r0_weekly_scale or force_r0_fourier_scale:
        try:
            setattr(op, "_allow_r0_weekly_scale_with_modifiers", True)
        except Exception:
            pass

    # Disable Op smoothing when using Fourier (to avoid double-smoothing)
    if force_r0_fourier_scale:
        try:
            setattr(op, "_smooth_r0_days", 0)
        except Exception:
            pass

    A, W, L = op.weekly_shape
    mod_names = tuple(pipeline.modifier_order())

    # coords
    coords = {
        "age": np.array(op.age_labels, dtype=object),
        "week": np.arange(W),
        "location": np.arange(L),
        "modifier": np.array(mod_names, dtype=object),
        "lag": np.arange(5),
    }

    # If Fourier: add coeff coord (1 + 2K)
    if force_r0_fourier_scale:
        K = int(max(0, fourier_harmonics))
        coeff_names = _fourier_names(K)
        coords["fourier_coeff"] = np.array(coeff_names, dtype=object)

    # population exposure
    pop_loc = None
    try:
        pop_arr = np.asarray(pipeline.precomputed.get("population", None))
        if pop_arr is not None:
            pop_loc = pop_arr if pop_arr.ndim == 1 else pop_arr.sum(axis=0)
    except Exception:
        pop_loc = None
    if pop_loc is None:
        try:
            pop_loc = np.asarray(pipeline.initial_array, dtype=np.float64).sum(axis=0)
        except Exception:
            pop_loc = np.ones(L, dtype=np.float64)
    pop_loc = np.clip(np.asarray(pop_loc, dtype=np.float64).reshape(L), 1.0, np.inf)

    defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, mod_names)

    # ---------- Precompute Fourier LS in NumPy (for priors), if requested ----------
    K = int(max(0, fourier_harmonics))
    period_days = float(fourier_period_days)
    theta_hat_loc = None
    resid_std_loc = None
    Xw_shared = None

    if force_r0_fourier_scale:
        # Access baseline r0 daily from base_params
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

        # week centers from Op (authoritative W)
        week_centers = np.asarray(getattr(op, "_week_centers", np.arange(W) * 7.0), dtype=float)[:W]
        Xw_shared = _fourier_design(week_centers, K, period_days)  # (W, M)

        M = 1 + 2 * K
        theta_hat_loc = np.zeros((M, L), dtype=float)
        resid_std_loc = np.zeros((L,), dtype=float)

        for l in range(L):
            r0_daily = base[r0_idx, :, l].astype(float)
            th, rs = _ls_init_fourier(r0_daily, t_days, K, period_days)
            theta_hat_loc[:, l] = th
            resid_std_loc[l] = rs

    with pm.Model(coords=coords) as m:
        sigma_mod_log = 0.35
        mu_log_base = np.log(defaults + 1e-12) - 0.5 * (sigma_mod_log ** 2)
        mods_vec = pm.Deterministic("mods", pt.ones((len(mod_names),)), dims=("modifier",))

        # ===== SINGLE-LOCATION =====
        if L == 1:
            mods_mu_log_loc = pm.Normal(
                "mods_mu_log_loc", mu=mu_log_base[:, None], sigma=0.30, dims=("modifier", "location")
            )
            mods_loc = pm.LogNormal("mods_loc", mu=mods_mu_log_loc, sigma=sigma_mod_log, dims=("modifier", "location"))
            pR = pm.Beta("pR", alpha=8.0, beta=12.0, dims=("age", "location"))

            lambda_ext_loc = None
            if op.has_lambda_ext:
                lambda_ext_loc = pm.LogNormal("lambda_ext_loc", mu=0.0, sigma=0.25, dims=("location",))
                pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))

            # ---------- r0 scale (RW2 vs Fourier vs none) ----------
            r0_weekly_scale_full = None
            base_scale = pt.ones((W, L))

            if not op.has_r0_modifiers:
                u_loc = pm.Beta("r0_global_unit_loc", alpha=5.0, beta=5.0, dims=("location",))
                r0_global_loc = pm.Deterministic("r0_global_loc", 0.8 + 0.4 * u_loc, dims=("location",))
                base_scale = r0_global_loc[None, :] * base_scale

            if force_r0_fourier_scale:
                # Priors around LS init with moderate shrinkage (per-harmonic decay) and bounded amplitude
                M = 1 + 2 * K
                th0 = theta_hat_loc[:, 0]
                # Base per-location scale from residuals, capped
                base_sigma = float(np.clip(resid_std_loc[0], 0.05, 0.8))
                # Decay ~ 1/k for cos/sin blocks; leave intercept less shrunk
                decay = np.ones(M, dtype=float)
                if K > 0:
                    dk = np.repeat(1.0 / np.arange(1, K + 1), 2)  # [1,1, 1/2,1/2, ...]
                    decay[1:] = dk
                # If user sets very large K, damp overall variance to keep total wiggle modest
                if K > 10:
                    decay *= np.sqrt(10.0 / K)
                sigma_vec = base_sigma * decay
                theta = pm.Normal("r0_fourier_coef_loc", mu=th0, sigma=sigma_vec, dims=("fourier_coeff",))
                Xw = pt.as_tensor_variable(Xw_shared)  # (W, M)
                f_w = Xw @ theta  # (W,)
                f_w_center = f_w - pt.mean(f_w)
                # Bounded amplitude via tanh ⇒ exp(±amp) envelope on weekly scale
                amp = pm.HalfNormal("r0_fourier_amp_loc", sigma=0.7)  # moderate amplitude
                r0_weekly_scale_full = pm.Deterministic(
                    "r0_weekly_scale",
                    base_scale[:, 0:1] * pt.exp(amp * pt.tanh(f_w_center))[:, None],  # (W,1)
                    dims=("week", "location"),
                )

            elif force_r0_weekly_scale or (not op.has_r0_modifiers):
                r0_sigma_loc = pm.HalfNormal("r0_weekly_sigma_loc", 0.03, dims=("location",))
                eps2 = pm.Normal("r0_rw2_eps", 0.0, r0_sigma_loc[None, :], dims=("week", "location"))
                eps1 = pt.cumsum(eps2, axis=0)
                eps = pt.cumsum(eps1, axis=0)
                eps_center = eps - pt.mean(eps, axis=0, keepdims=True)
                r0_weekly_scale_full = pm.Deterministic(
                    "r0_weekly_scale", base_scale * pt.exp(eps_center), dims=("week", "location")
                )

            # forward
            if op.has_lambda_ext and (r0_weekly_scale_full is not None):
                weekly_pred_t, S_final_t = op(mods_vec, pR,
                                              lambda_ext_loc=lambda_ext_loc, mods_loc=mods_loc,
                                              r0_weekly_scale=r0_weekly_scale_full)
            elif op.has_lambda_ext:
                weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc=lambda_ext_loc, mods_loc=mods_loc)
            elif r0_weekly_scale_full is not None:
                weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc=mods_loc, r0_weekly_scale=r0_weekly_scale_full)
            else:
                weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc=mods_loc)

            weekly = pm.Deterministic("weekly_pred", weekly_pred_t, dims=("age", "week", "location"))

            # Age scaling
            sigma_age_loc = pm.HalfNormal("sigma_age_loc", 0.15, dims=("location",))
            z_age_loc = pm.Normal("z_age_loc", 0.0, 1.0, dims=("age", "location"))
            theta_age_loc = pm.Deterministic("theta_age_loc", z_age_loc * sigma_age_loc[None, :], dims=("age", "location"))
            scale_raw = pt.exp(theta_age_loc)

            w = 1.0 - pR
            w_sum = pt.sum(w, axis=0, keepdims=True)
            w_norm = w / (w_sum + 1e-12)
            norm = pt.sum(w_norm * scale_raw, axis=0, keepdims=True)
            age_scale = pm.Deterministic("age_scale", scale_raw / (norm + 1e-12), dims=("age", "location"))

            weekly_scaled = pm.Deterministic("weekly_pred_scaled", weekly * age_scale[:, None, :],
                                             dims=("age", "week", "location"))
            weekly_sum_age = pm.Deterministic("weekly_pred_sum_age", weekly_scaled.sum(axis=0),
                                              dims=("week", "location"))
            pm.Deterministic("S_final", S_final_t, dims=("age", "location"))

            sigma_week_loc = pm.HalfNormal("sigma_week_loc", 0.35, dims=("location",))
            z_week = pm.Normal("z_week", 0.0, 1.0, dims=("week", "location"))
            delta_week_cum = pt.cumsum(z_week * sigma_week_loc[None, :], axis=0)
            delta_week = pm.Deterministic("delta_week",
                                          delta_week_cum - pt.mean(delta_week_cum, axis=0, keepdims=True),
                                          dims=("week", "location"))

            S_lag = 5
            pads = S_lag // 2
            shifts = list(range(-pads, pads + 1))
            a0 = np.array([0.25, 0.5, 98.5, 0.5, 0.25], dtype=np.float64)
            a = np.tile(a0, (L, 1))
            lag_weights = pm.Dirichlet("lag_weights", a=a, dims=("location", "lag"))

            top_pad = pt.repeat(weekly_sum_age[:1, :], pads, axis=0)
            bot_pad = pt.repeat(weekly_sum_age[-1:, :], pads, axis=0)
            padded = pt.concatenate([top_pad, weekly_sum_age, bot_pad], axis=0)
            shifted_sum = pt.zeros_like(weekly_sum_age)
            for k, s in enumerate(shifts):
                start = pads + s
                Yk = padded[start:start + W, :]
                wk = lag_weights[:, k]
                shifted_sum = shifted_sum + Yk * wk[None, :]
            weekly_sum_age_shifted = pm.Deterministic("weekly_pred_sum_age_shifted", shifted_sum,
                                                      dims=("week", "location"))

            # ---- Intercept (wider prior)
            beta0_loc = pm.Normal("beta0_loc", 0.0, 2.0, dims=("location",))

            pop = pt.as_tensor_variable(pop_loc)
            rate_pred = weekly_sum_age_shifted / (pop[None, :] + 1e-12)
            log_rate = pt.log(rate_pred + 1e-12) + beta0_loc[None, :] + delta_week
            mu_obs = pt.exp(log_rate) * pop[None, :]

            if y_obs is not None:
                y_obs = np.asarray(y_obs, dtype=np.float64)
                assert y_obs.shape == (W, L)
                if use_nb:
                    # slightly looser dispersion prior
                    alpha_nb_loc = pm.LogNormal("alpha_nb_loc", mu=np.log(25.0), sigma=0.7, dims=("location",))
                    pm.NegativeBinomial("y", mu=mu_obs, alpha=alpha_nb_loc, observed=y_obs, dims=("week", "location"))
                else:
                    pm.Poisson("y", mu=mu_obs, observed=y_obs, dims=("week", "location"))
            return m

        # ===== MULTI-LOCATION =====
        mods_mu_log_ls, mods_loc_ls, pR_ls = [], [], []
        lambda_ls, r0_scale_ls = [], []

        sigma_age_ls, z_age_ls = [], []
        sigma_week_ls, z_week_ls = [], []
        beta0_ls, lag_w_ls = [], []
        alpha_nb_ls = []  # optional

        base_scale = pt.ones((W, L))
        r0_global_loc = None
        if not op.has_r0_modifiers:
            u_loc_ls = [pm.Beta(f"r0_global_unit_loc_l{l}", alpha=5.0, beta=5.0) for l in range(L)]
            r0_global_loc = pt.stack([0.8 + 0.4 * u for u in u_loc_ls], axis=0)
            pm.Deterministic("r0_global_loc", r0_global_loc, dims=("location",))
            base_scale = r0_global_loc[None, :] * base_scale

        # Prepare shared Xw if Fourier
        if force_r0_fourier_scale:
            Xw = pt.as_tensor_variable(Xw_shared)  # (W, M)
            M = Xw_shared.shape[1]

        for l in range(L):
            mods_mu_log_l = pm.Normal(f"mods_mu_log_loc_l{l}", mu=mu_log_base, sigma=0.30, dims=("modifier",))
            mods_l = pm.LogNormal(f"mods_loc_l{l}", mu=mods_mu_log_l, sigma=sigma_mod_log, dims=("modifier",))
            mods_mu_log_ls.append(mods_mu_log_l); mods_loc_ls.append(mods_l)

            pR_l = pm.Beta(f"pR_l{l}", alpha=8.0, beta=12.0, dims=("age",))
            pR_ls.append(pR_l)

            if op.has_lambda_ext:
                lam_l = pm.LogNormal(f"lambda_ext_loc_l{l}", mu=0.0, sigma=0.25)
                lambda_ls.append(lam_l)

            r0_scale_l = None
            if force_r0_fourier_scale:
                th0 = theta_hat_loc[:, l]
                base_sigma = float(np.clip(resid_std_loc[l], 0.05, 0.8))
                decay = np.ones(M, dtype=float)
                if K > 0:
                    dk = np.repeat(1.0 / np.arange(1, K + 1), 2)
                    decay[1:] = dk
                if K > 10:
                    decay *= np.sqrt(10.0 / K)
                sigma_vec = base_sigma * decay
                theta_l = pm.Normal(f"r0_fourier_coef_loc_l{l}", mu=th0, sigma=sigma_vec, dims=("fourier_coeff",))
                f_w_l = Xw @ theta_l  # (W,)
                f_w_l_center = f_w_l - pt.mean(f_w_l)
                amp_l = pm.HalfNormal(f"r0_fourier_amp_loc_l{l}", sigma=0.7)
                r0_scale_l = pt.exp(amp_l * pt.tanh(f_w_l_center)) * (
                    r0_global_loc[l] if r0_global_loc is not None else 1.0
                )

            elif force_r0_weekly_scale or (not op.has_r0_modifiers):
                r0_sigma_l = pm.HalfNormal(f"r0_weekly_sigma_loc_l{l}", 0.03)
                eps2_l = pm.Normal(f"r0_rw2_eps_l{l}", 0.0, r0_sigma_l, dims=("week",))
                eps1_l = pt.cumsum(eps2_l)
                eps_l = pt.cumsum(eps1_l)
                eps_center_l = eps_l - pt.mean(eps_l)
                r0_scale_l = pt.exp(eps_center_l) * (r0_global_loc[l] if r0_global_loc is not None else 1.0)

            r0_scale_ls.append(r0_scale_l)

            sigma_age_l = pm.HalfNormal(f"sigma_age_loc_l{l}", 0.15)
            z_age_l = pm.Normal(f"z_age_loc_l{l}", 0.0, 1.0, dims=("age",))
            sigma_age_ls.append(sigma_age_l); z_age_ls.append(z_age_l)

            sigma_week_l = pm.HalfNormal(f"sigma_week_loc_l{l}", 0.35)
            z_week_l = pm.Normal(f"z_week_l{l}", 0.0, 1.0, dims=("week",))
            sigma_week_ls.append(sigma_week_l); z_week_ls.append(z_week_l)

            beta0_ls.append(pm.Normal(f"beta0_loc_l{l}", 0.0, 1.0))
            lag_w_ls.append(pm.Dirichlet(f"lag_weights_l{l}", a=np.array([0.25, 0.5, 98.5, 0.5, 0.25])))

            if use_nb:
                alpha_nb_ls.append(pm.LogNormal(f"alpha_nb_loc_l{l}", mu=np.log(25.0), sigma=0.7))

        # Stack back
        mods_loc = pm.Deterministic("mods_loc", pt.stack(mods_loc_ls, axis=1), dims=("modifier", "location"))
        pm.Deterministic("mods_mu_log_loc", pt.stack(mods_mu_log_ls, axis=1), dims=("modifier", "location"))
        pR = pm.Deterministic("pR", pt.stack(pR_ls, axis=1), dims=("age", "location"))
        if op.has_lambda_ext:
            lambda_ext_loc = pm.Deterministic("lambda_ext_loc", pt.stack(lambda_ls, axis=0), dims=("location",))
            pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))
        else:
            lambda_ext_loc = None

        if force_r0_fourier_scale:
            r0_weekly_scale_full = pm.Deterministic(
                "r0_weekly_scale",
                pt.stack(
                    [ (r if r is not None else pt.ones((W,))) for r in r0_scale_ls ],
                    axis=1
                ) * (base_scale),
                dims=("week", "location"),
            )
        elif force_r0_weekly_scale or (not op.has_r0_modifiers):
            r0_weekly_scale_full = pm.Deterministic(
                "r0_weekly_scale",
                pt.stack([r if r is not None else pt.ones((W,)) for r in r0_scale_ls], axis=1),
                dims=("week", "location"),
            )
        else:
            r0_weekly_scale_full = None

        # forward pass
        if op.has_lambda_ext and (r0_weekly_scale_full is not None):
            weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc=lambda_ext_loc,
                                          mods_loc=mods_loc, r0_weekly_scale=r0_weekly_scale_full)
        elif op.has_lambda_ext:
            weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc=lambda_ext_loc, mods_loc=mods_loc)
        elif r0_weekly_scale_full is not None:
            weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc=mods_loc, r0_weekly_scale=r0_weekly_scale_full)
        else:
            weekly_pred_t, S_final_t = op(mods_vec, pR, mods_loc=mods_loc)

        weekly = pm.Deterministic("weekly_pred", weekly_pred_t, dims=("age", "week", "location"))

        # Age scaling
        theta_age = pt.stack([z_age_ls[l] * sigma_age_ls[l] for l in range(L)], axis=1)
        theta_age_loc = pm.Deterministic("theta_age_loc", theta_age, dims=("age", "location"))
        scale_raw = pt.exp(theta_age_loc)
        w = 1.0 - pR
        w_sum = pt.sum(w, axis=0, keepdims=True)
        w_norm = w / (w_sum + 1e-12)
        norm = pt.sum(w_norm * scale_raw, axis=0, keepdims=True)
        age_scale = pm.Deterministic("age_scale", scale_raw / (norm + 1e-12), dims=("age", "location"))

        weekly_scaled = pm.Deterministic("weekly_pred_scaled", weekly * age_scale[:, None, :],
                                         dims=("age", "week", "location"))
        weekly_sum_age = pm.Deterministic("weekly_pred_sum_age", weekly_scaled.sum(axis=0),
                                          dims=("week", "location"))
        pm.Deterministic("S_final", S_final_t, dims=("age", "location"))

        # Weekly residuals + lag
        delta_week_cum = pt.stack([pt.cumsum(z_week_ls[l] * sigma_week_ls[l]) for l in range(L)], axis=1)
        delta_week = pm.Deterministic("delta_week",
                                      delta_week_cum - pt.mean(delta_week_cum, axis=0, keepdims=True),
                                      dims=("week", "location"))

        S_lag = 5
        pads = S_lag // 2
        shifts = list(range(-pads, pads + 1))
        lag_weights = pm.Deterministic("lag_weights", pt.stack(lag_w_ls, axis=0), dims=("location", "lag"))

        top_pad = pt.repeat(weekly_sum_age[:1, :], pads, axis=0)
        bot_pad = pt.repeat(weekly_sum_age[-1:, :], pads, axis=0)
        padded = pt.concatenate([top_pad, weekly_sum_age, bot_pad], axis=0)
        shifted_sum = pt.zeros_like(weekly_sum_age)
        for k, s in enumerate(shifts):
            start = pads + s
            Yk = padded[start:start + W, :]
            wk = lag_weights[:, k]
            shifted_sum = shifted_sum + Yk * wk[None, :]
        weekly_sum_age_shifted = pm.Deterministic("weekly_pred_sum_age_shifted", shifted_sum,
                                                  dims=("week", "location"))

        # ---- Intercept: hierarchical across locations
        mu_beta0 = pm.Normal("mu_beta0", 0.0, 2.0)
        sigma_beta0 = pm.HalfNormal("sigma_beta0", 1.0)
        beta0_loc = pm.Deterministic(
            "beta0_loc",
            pt.stack([pm.Normal(f"beta0_loc_l{l}", mu_beta0, sigma_beta0) for l in range(L)], axis=0),
            dims=("location",),
        )

        pop = pt.as_tensor_variable(pop_loc)
        rate_pred = weekly_sum_age_shifted / (pop[None, :] + 1e-12)
        log_rate = pt.log(rate_pred + 1e-12) + beta0_loc[None, :] + delta_week
        mu_obs = pt.exp(log_rate) * pop[None, :]

        if y_obs is not None:
            y_obs = np.asarray(y_obs, dtype=np.float64)
            assert y_obs.shape == (W, L), f"y_obs must be (W, L); got {y_obs.shape}"
            if use_nb:
                alpha_nb_loc = pm.Deterministic("alpha_nb_loc", pt.stack(alpha_nb_ls, axis=0), dims=("location",))
                pm.NegativeBinomial("y", mu=mu_obs, alpha=alpha_nb_loc, observed=y_obs, dims=("week", "location"))
            else:
                pm.Poisson("y", mu=mu_obs, observed=y_obs, dims=("week", "location"))

    return m

# ------------------------------ blocked step helper ------------------------------

def make_location_blocked_step(
    model: pm.Model,
    step_cls=pm.DEMetropolisZ,
    **kwargs,
) -> pm.CompoundStep | pm.ArrayStep:
    try:
        from pymc.util import get_untransformed_name, is_transformed_name
    except Exception:
        def is_transformed_name(name: str) -> bool:
            return name.endswith("__")
        def get_untransformed_name(name: str) -> str:
            if not is_transformed_name(name):
                return name
            base = name[:-2]
            if "_" in base:
                base = base[: base.rfind("_")]
            return base

    free = list(model.free_RVs)
    if not free:
        return step_cls(vars=[], **kwargs)

    import os, re
    loc_re = re.compile(r"_l(?P<idx>\d+)(?:$|_)")

    by_loc: dict[int, list] = {}
    globals_group: list = []

    for rv in free:
        tname = str(rv.name)
        base = get_untransformed_name(tname) if is_transformed_name(tname) else tname
        m = loc_re.search(base)
        if m:
            l = int(m.group("idx"))
            by_loc.setdefault(l, []).append(rv)
        else:
            globals_group.append(rv)

    if os.environ.get("WEEKLY_BLOCK_DEBUG", "").strip() == "1":
        try:
            print("[make_location_blocked_step] groups:",
                  {k: len(v) for k, v in sorted(by_loc.items())},
                  "globals:", len(globals_group))
        except Exception:
            pass

    steps: list[pm.ArrayStep] = []
    for l in sorted(by_loc):
        group = by_loc[l]
        if group:
            steps.append(step_cls(vars=group, **kwargs))

    if globals_group:
        steps.append(step_cls(vars=globals_group, **kwargs))

    if not steps:
        return step_cls(vars=free, **kwargs)
    if len(steps) == 1:
        return steps[0]
    return pm.CompoundStep(steps)


def make_location_blocked_step_factory(step_cls=pm.DEMetropolisZ, **kwargs):
    def _factory(model: pm.Model | None = None):
        mdl = model if model is not None else pm.modelcontext(model)
        return make_location_blocked_step(mdl, step_cls=step_cls, **kwargs)
    return _factory

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
