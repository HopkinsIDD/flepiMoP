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
    """Extract a float 'value' from a flexible YAML spec dict; fallback to 1.0."""
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
    """Read seir_modifiers defaults in the pipeline’s leaf order."""
    conf = confuse.Configuration("WeeklyHospPipelineDefaults", __name__)
    conf.set_file(str(config_path))
    sm = conf["seir_modifiers"].get()
    mods = sm["modifiers"]
    return np.asarray([_extract_yaml_value(mods[nm]) for nm in leaf_order], dtype=np.float64)


def _normalize_age_token(s: str) -> str:
    """Normalize tokens like 'age5to17' / '5–17' / '5-17' / '5to17+' into a compact key."""
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
    """Masks to aggregate compartments by age & infection stage."""
    age_labels: tuple[str, ...]
    s_mask_by_age: tuple[np.ndarray, ...]      # per-age (NC,) bool
    r_mask_by_age: tuple[np.ndarray, ...]      # per-age (NC,) bool
    total_mask_by_age: tuple[np.ndarray, ...]  # per-age (NC,) bool (all stages)


def _build_age_masks(pipeline: WeeklyHospPipeline) -> AgeMasks:
    """Map pipeline.age_labels onto compartments dataframe columns."""
    df = pipeline.model.compartments.compartments  # pandas.DataFrame
    NC = pipeline.NC

    comp_age_norm = df["age_strata"].astype(str).map(_normalize_age_token).values
    comp_stage = df["infection_stage"].astype(str).values
    is_S = np.fromiter((str(st).startswith("S") for st in comp_stage), dtype=bool, count=NC)
    is_R = np.fromiter((str(st).startswith("R") for st in comp_stage), dtype=bool, count=NC)

    masks_s: list[np.ndarray] = []
    masks_r: list[np.ndarray] = []
    masks_tot: list[np.ndarray] = []
    for age_token in pipeline.age_labels:
        key = _normalize_age_token(age_token.replace("age", "", 1))
        m_age = (comp_age_norm == key)
        if not m_age.any():
            m_age = np.ones(NC, dtype=bool)
        m_s = m_age & is_S
        m_r = m_age & is_R
        masks_s.append(m_s)
        masks_r.append(m_r)
        masks_tot.append(m_age)
    return AgeMasks(tuple(pipeline.age_labels), tuple(masks_s), tuple(masks_r), tuple(masks_tot))


def _mmwr_assign_from(start_date: np.datetime64 | object, T: int) -> tuple[np.ndarray, int]:
    """Assign each day index in [0..T-1] to an MMWR week id, returning (assign, n_weeks)."""
    if T <= 0:
        return np.zeros(0, dtype=np.int64), 0
    if isinstance(start_date, np.datetime64):
        start = _dt.date.fromtimestamp((start_date - np.datetime64("1970-01-01")) / np.timedelta64(1, "s"))
    else:
        start = start_date  # assume datetime.date
    offset_to_sun = (6 - start.weekday()) % 7  # Monday=0..Sunday=6
    first_len = 7 if offset_to_sun == 0 else offset_to_sun
    first_len = min(first_len, T)

    assign = np.empty(T, dtype=np.int64)
    assign[:first_len] = 0
    if T > first_len:
        rest = T - first_len
        assign[first_len:] = 1 + (np.arange(rest, dtype=np.int64) // 7)
    n_weeks = int(assign.max()) + 1
    return assign, n_weeks


# ------------------------------ Op (Pattern #2 + location modifiers) ------------------------------

class WeeklyHospAndFinalSOp(Op):
    """
    Runs the simulator once and returns:
      • weekly   : (age, week, location) accumulated outcomes from config (e.g., hospitalizations)
      • S_final  : (age, location) final susceptible counts

    Inputs:
      mods            (1D): replacement modifier values (shared center) in pipeline.leaf order
      pR              (2D): fraction immune at t=0, shape (age, location)
      lambda_ext_loc  (1D): per-location exogenous force of infection (multiplicative scale)
      mods_loc  [opt] (2D): per-location replacement multipliers, shape (modifier, location)

    Semantics:
      If `mods_loc` is provided, the effective leaf vector for location ℓ is:
         mods_eff[:, ℓ] = mods * mods_loc[:, ℓ]
      We then apply the modifier applier **once per location** and stitch the results
      into a (P_base, T, L) tensor, preserving Pattern #2 thereafter.
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

    # ---- helpers -------------------------------------------------------

    def _get_param_row(self, name: str) -> int:
        try:
            return int(self._param_name_to_idx[name])
        except Exception as e:
            raise KeyError(f"Parameter row '{name}' not found in pipeline mapping.") from e

    def _scale_lambda_ext(self, params_base: np.ndarray, lambda_ext_loc: np.ndarray) -> None:
        """Multiply lambda_ext time series by a per-location scale (no-op if absent)."""
        if "lambda_ext" not in self._param_name_to_idx:
            return
        pidx = self._get_param_row("lambda_ext")
        params_base[pidx, :, :] *= lambda_ext_loc[None, :]

    def _override_ic_with_pR_preserve_others(self, y0_in: np.ndarray, pR: np.ndarray) -> np.ndarray:
        """Redistribute only S and R within each (age, location); preserve all other compartments."""
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
        mods_shared: np.ndarray,                 # (n_leaves,)
        mods_loc: np.ndarray | None,             # (n_leaves, L) or None
    ) -> np.ndarray:
        """
        Return a (P_base, T, L) parameter tensor with **per-location** leaf values applied.
        Implementation: call the applier once per location with the location's effective
        leaf vector, then copy out the single location's column into the output tensor.
        """
        mods_shared = np.asarray(mods_shared, dtype=np.float64).reshape(-1)
        L = self._L
        if mods_loc is None:
            eff = np.tile(mods_shared[:, None], (1, L))  # (n_leaves, L)
        else:
            mods_loc = np.asarray(mods_loc, dtype=np.float64)
            if mods_loc.shape != (mods_shared.shape[0], L):
                raise ValueError(f"mods_loc shape {mods_loc.shape} != {(mods_shared.shape[0], L)}")
            eff = mods_shared[:, None] * mods_loc  # (n_leaves, L)

        # Output initialized from base (we'll overwrite col-by-col)
        out = np.array(self.pipe.base_params, dtype=np.float64, copy=True)
        # For each location, apply leaf values and take only that column
        for j in range(L):
            params_j = self.pipe.mod_applier.apply_to_params(
                self.pipe.base_params, leaf_value_array=eff[:, j], scenario="none"
            )
            out[:, :, j] = params_j[:, :, j]
        return out

    # ---- Op API --------------------------------------------------------

    def make_node(self, mods, pR, lambda_ext_loc, mods_loc=None):
        """
        Accepts 3 or 4 inputs. If `mods_loc` is omitted, modifiers are shared across locations.
        """
        mods = pt.as_tensor_variable(mods)
        pR = pt.as_tensor_variable(pR)
        lambda_ext_loc = pt.as_tensor_variable(lambda_ext_loc)
        if mods.ndim != 1:
            raise TypeError("mods must be 1D")
        if pR.ndim != 2:
            raise TypeError("pR must be 2D (age, location)")
        if lambda_ext_loc.ndim != 1:
            raise TypeError("lambda_ext_loc must be 1D (location,)")

        inputs = [mods, pR, lambda_ext_loc]

        if mods_loc is not None:
            mods_loc = pt.as_tensor_variable(mods_loc)
            if mods_loc.ndim != 2:
                raise TypeError("mods_loc must be 2D (modifier, location)")
            inputs.append(mods_loc)

        # outputs: (weekly_age, S_final)
        return Apply(self, inputs, [pt.dtensor3(), pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        """
        Pattern #2:
          1) Apply modifiers (shared or per-location) to BASE params.
          2) Inject lambda_ext scaling (per location).
          3) Parse BASE → UNIQUE (rows align with transitions[2,*]).
          4) Integrate once on the step grid; reconstruct per-step transitions & apply outcome prob/delay.
          5) Aggregate steps → MMWR weeks; read S_final from same trajectory.
        """
        # Unpack (3 or 4 inputs)
        if len(inputs) == 3:
            mods, pR, lambda_ext_loc = inputs
            mods_loc = None
        elif len(inputs) == 4:
            mods, pR, lambda_ext_loc, mods_loc = inputs
        else:
            raise TypeError(f"Expected 3 or 4 inputs, got {len(inputs)}")

        if mods.shape[0] != len(self.leaf_order):
            raise ValueError(f"mods length {mods.shape[0]} != {len(self.leaf_order)}")
        if pR.shape != (self._A, self._L):
            raise ValueError(f"pR shape {pR.shape} != {(self._A, self._L)}")
        if lambda_ext_loc.shape != (self._L,):
            raise ValueError(f"lambda_ext_loc shape {lambda_ext_loc.shape} != {(self._L,)}")
        if mods_loc is not None and mods_loc.shape != (len(self.leaf_order), self._L):
            raise ValueError(
                f"mods_loc shape {mods_loc.shape} != {(len(self.leaf_order), self._L)}"
            )

        # 1) Apply modifiers to BASE params (location-wise if mods_loc provided)
        params_base = self._apply_modifiers_locationwise(mods, mods_loc)

        # 1b) Inject exogenous FOI scale (if available)
        self._scale_lambda_ext(params_base, np.asarray(lambda_ext_loc, dtype=np.float64))

        # 1c) BASE → UNIQUE (align with transitions[2,*])
        params_unique = self.pipe.model.compartments.parse_parameters(
            params_base, self.pipe.param_defs, self.pipe.unique_strings
        )

        # 2) Override initial conditions (S/R only; preserve others)
        y0 = self._override_ic_with_pR_preserve_others(self.pipe.initial_array, np.asarray(pR, dtype=np.float64))

        # 3) Integrate on the pipeline step grid
        total_days = float(self.pipe.T - 1)
        n_steps = max(1, int(round(total_days / self.pipe.dt)))
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)
        res = self._rhs.solve(
            y0=y0.ravel(),
            parameters=params_unique,  # UNIQUE tensor; RHS indexes pidx directly
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-3, atol=1e-6,
        )
        if not res.success:
            raise RuntimeError(f"Integration failed: {res.message}")
        states = res.y.T.reshape(len(t_eval), self.pipe.NC, self._L)

        # 4) Reconstruct transition amounts and accumulate outcomes (prob + delay)
        series = {name: np.zeros((n_steps, self._L), dtype=np.float64) for name in self._out_order}
        pc = self.pipe.precomputed
        for i in range(n_steps):
            t0 = t_eval[i]
            param_t_slice = _param_slice_step(params_unique, t0)  # (P_unique, L)
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
                # DIRECT indexing path (Pattern #2)
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

        # Resolve sums/aliases (with their own prob+delay)
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

        # 5) Steps → MMWR weeks and aggregate by age mapping
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

        # 6) Final S from the same trajectory
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


# ------------------------------ model builder ------------------------------

def build_weekly_model(
    pipeline: WeeklyHospPipeline,
    op: WeeklyHospAndFinalSOp | None = None,
    *,
    y_obs: np.ndarray | None = None,
    use_nb: bool = False,
) -> pm.Model:
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

    def _is_month_leaf(n: str) -> bool:
        s = n.lower()
        return any(m in s for m in ["seas_oct", "seas_nov", "seas_dec", "seas_jan", "seas_feb", "seas_mar", "seas_apr"])

    def _is_holiday_leaf(n: str) -> bool:
        s = n.lower()
        return ("winter" in s) or ("holiday" in s)

    with pm.Model(coords=coords) as m:
        # ---- global (shared) modifier scales (replacement semantics)
        shared_mods = []
        for i, name in enumerate(mod_names):
            mu0 = float(np.log(defaults[i] + 1e-12))
            if _is_month_leaf(name):
                sigma = 0.8
            elif _is_holiday_leaf(name):
                sigma = 0.8
            else:
                sigma = 0.5
            shared_mods.append(pm.LogNormal(name, mu=mu0, sigma=sigma))
        mods_vec = pm.Deterministic("mods", pt.stack(shared_mods), dims=("modifier",))

        # ---- pR: logit-hierarchical (shared across age here, replicated to A)
        mu0 = float(np.log(0.40 / (1.0 - 0.40)))
        mu_R = pm.Normal("mu_R_logit", mu=mu0, sigma=0.5)
        sd_loc = pm.HalfNormal("sigma_R_loc", sigma=0.3)
        b_loc = pm.Normal("b_loc", mu=0.0, sigma=sd_loc, dims=("location",))
        eta = mu_R + b_loc  # (L,)
        pR = pm.Deterministic("pR", pt.sigmoid(eta)[None, :].repeat(A, axis=0), dims=("age", "location"))

        # ---- exogenous FOI per location (log-normal around shared center)
        lam_center = 2.5e-5
        mu_lex = pm.Normal("lambda_ext_mu_log", mu=np.log(lam_center + 1e-16), sigma=0.75)
        sd_lex = pm.HalfNormal("lambda_ext_sigma_log", sigma=0.2)
        lambda_ext_loc = pm.LogNormal("lambda_ext_loc", mu=mu_lex, sigma=sd_lex, dims=("location",))
        pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))

        # ---- forward model (single solve through the Op)
        # Default path (3 inputs); you can pass a 4th argument mods_loc from a hierarchical model.
        weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc)
        weekly = pm.Deterministic("weekly_pred", weekly_pred_t, dims=("age", "week", "location"))
        weekly_sum_age = pm.Deterministic("weekly_pred_sum_age", weekly.sum(axis=0), dims=("week", "location"))
        pm.Deterministic("S_final", S_final_t, dims=("age", "location"))

        # ---- simple observation layer (aggregate over age to match typical data)
        obs_loc_scale = pm.LogNormal("obs_loc_scale", mu=0.0, sigma=0.5, dims=("location",))
        mu_obs = weekly_sum_age * obs_loc_scale[None, :]

        if y_obs is not None:
            y_obs = np.asarray(y_obs, dtype=np.float64)
            assert y_obs.shape == (W, L), f"y_obs must be (W, L); got {y_obs.shape}"
            if use_nb:
                alpha_loc = pm.HalfNormal("alpha_nb_loc", sigma=10.0, dims=("location",))
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
    """Create the pipeline, Op (weekly + S_final), and PyMC5 model with coords."""
    pipe = build_pipeline_from_config(config_path)
    op = WeeklyHospAndFinalSOp(pipe)
    model = build_weekly_model(pipe, op, y_obs=y_obs)
    return pipe, op, model
