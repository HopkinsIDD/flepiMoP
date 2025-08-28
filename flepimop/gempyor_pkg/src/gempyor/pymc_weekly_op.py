from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


# ------------------------------ Op ------------------------------

class WeeklyHospAndFinalSOp(Op):
    """
    PyTensor Op that runs the simulator once and returns:
      weekly      : (age, week, location) accumulated outcomes from config (e.g., hospitalizations)
      S_final     : (age, location) final susceptible counts

    Inputs:
      mods           (1D): replacement modifier values in pipeline.leaf order
      pR             (2D): fraction immune at t=0, shape (age, location)
      lambda_ext_loc (1D): per-location exogenous force of infection (multiplicative scale)

    Implementation detail:
      • We perform a single ODE integration on the pipeline’s step grid (Δt = pipeline.dt),
        reconstruct transition amounts per step, accumulate outcomes using the pipeline’s
        compiled outcome graph, and read the final state for S — *all from the same solve*.
    """
    itypes = [pt.dvector, pt.dmatrix, pt.dvector]
    otypes = [pt.dtensor3, pt.dmatrix]

    def __init__(self, pipeline: WeeklyHospPipeline):
        super().__init__()
        self.pipe = pipeline
        self.leaf_order = pipeline.modifier_order()
        self.age_masks = _build_age_masks(pipeline)

        # Probe shapes (fixes W etc.) — a single build-time run
        defaults = _yaml_defaults_in_leaf_order(pipeline.config_path, self.leaf_order)
        wk, _, _ = pipeline.evaluate(defaults)
        self._weekly_shape = wk.shape
        self._A, self._W, self._L = self._weekly_shape
        self._age_to_names = dict(pipeline.age_to_names)
        self._age_labels = tuple(pipeline.age_labels)

        # Outcomes compilation artifacts (resolver & prob/delay maps) at step width
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

        # Parameter row mapping helper (if exposed by the pipeline stack)
        self._param_name_to_idx: dict[str, int] = {}
        if hasattr(self.pipe, "param_name_to_idx"):
            self._param_name_to_idx = dict(self.pipe.param_name_to_idx)
        elif hasattr(self.pipe, "mod_applier"):
            self._param_name_to_idx = dict(getattr(self.pipe.mod_applier, "param_name_to_idx", {}))

    # ---- helpers -------------------------------------------------------

    def _get_param_row(self, name: str) -> int:
        """Return row index for a parameter by name; raise KeyError if missing."""
        try:
            return int(self._param_name_to_idx[name])
        except Exception as e:
            raise KeyError(f"Parameter row '{name}' not found in pipeline mapping.") from e

    def _scale_lambda_ext(self, params_mod: np.ndarray, lambda_ext_loc: np.ndarray) -> None:
        """Multiply lambda_ext time series by a per-location scale (no-op if absent)."""
        if "lambda_ext" not in self._param_name_to_idx:
            return
        pidx = self._get_param_row("lambda_ext")
        params_mod[pidx, :, :] *= lambda_ext_loc[None, :]

    def _override_ic_with_pR_preserve_others(self, y0_in: np.ndarray, pR: np.ndarray) -> np.ndarray:
        """Redistribute only S and R within each (age, location) to match pR; keep other comps untouched."""
        y0 = np.array(y0_in, dtype=np.float64, copy=True)
        for a, (m_s, m_r, m_tot) in enumerate(
            zip(self.age_masks.s_mask_by_age, self.age_masks.r_mask_by_age, self.age_masks.total_mask_by_age)
        ):
            m_other = m_tot & (~m_s) & (~m_r)
            N_tot = y0[m_tot, :].sum(axis=0)
            N_other = y0[m_other, :].sum(axis=0)
            sr_mass = np.maximum(N_tot - N_other, 0.0)

            R_counts = np.clip(pR[a, :], 1e-6, 1 - 1e-6) * sr_mass
            S_counts = sr_mass - R_counts

            y0[m_s, :] = 0.0
            y0[m_r, :] = 0.0
            n_s_rows = int(m_s.sum())
            n_r_rows = int(m_r.sum())
            if n_s_rows > 0:
                y0[m_s, :] += (S_counts[None, :] / n_s_rows)
            if n_r_rows > 0:
                y0[m_r, :] += (R_counts[None, :] / n_r_rows)
        return y0

    # ---- PyTensor op methods ------------------------------------------

    def make_node(self, mods, pR, lambda_ext_loc):
        mods = pt.as_tensor_variable(mods)
        pR = pt.as_tensor_variable(pR)
        lambda_ext_loc = pt.as_tensor_variable(lambda_ext_loc)
        if mods.ndim != 1:
            raise TypeError("mods must be 1D")
        if pR.ndim != 2:
            raise TypeError("pR must be 2D (age, location)")
        if lambda_ext_loc.ndim != 1:
            raise TypeError("lambda_ext_loc must be 1D (location,)")
        return Apply(self, [mods, pR, lambda_ext_loc], [pt.dtensor3(), pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        """
        Single-solve execution:
          1) Apply modifier replacements and lambda_ext scaling
          2) Override S/R by pR (preserving non-(S|R) mass)
          3) Integrate once on the pipeline step grid (Δt = pipeline.dt)
          4) Reconstruct per-step transition amounts; accumulate outcomes with prob+delay
          5) Aggregate to MMWR weeks by pipeline.start_date
          6) Read S_final from the same trajectory
        """
        mods, pR, lambda_ext_loc = inputs
        if mods.shape[0] != len(self.leaf_order):
            raise ValueError(f"mods length {mods.shape[0]} != {len(self.leaf_order)}")
        if pR.shape != (self._A, self._L):
            raise ValueError(f"pR shape {pR.shape} != {(self._A, self._L)}")
        if lambda_ext_loc.shape != (self._L,):
            raise ValueError(f"lambda_ext_loc shape {lambda_ext_loc.shape} != {(self._L,)}")

        # 1) modifiers -> params (replacement semantics) + lambda_ext scaling
        params_mod = self.pipe.mod_applier.apply_to_params(
            self.pipe.base_params, leaf_value_array=np.asarray(mods, dtype=np.float64), scenario="none"
        )
        self._scale_lambda_ext(params_mod, np.asarray(lambda_ext_loc, dtype=np.float64))

        # 2) initial conditions override (S/R only)
        y0 = self._override_ic_with_pR_preserve_others(self.pipe.initial_array, np.asarray(pR, dtype=np.float64))

        # 3) One integration on step grid
        total_days = float(self.pipe.T - 1)
        dt = float(self.pipe.dt)
        n_steps = max(1, int(round(total_days / dt)))
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

        res = self.pipe.factory.solve(
            y0=y0.ravel(),
            parameters=params_mod,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-3,
            atol=1e-6,
        )
        if not res.success:
            raise RuntimeError(f"Integration failed: {res.message}")

        states = res.y.T.reshape(len(t_eval), self.pipe.NC, self._L)  # (T_pts, C, L)

        # 4) Per-step transition amounts + outcomes accumulation on the same grid
        series = {name: np.zeros((n_steps, self._L), dtype=np.float64) for name in self._out_order}
        pc = self.pipe.precomputed

        for i in range(n_steps):
            t0 = t_eval[i]
            param_t_slice = _param_slice_step(params_mod, t0)  # (P, L)
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
            )  # (Tn, L)

            # Leaf incidence nodes (direct transition rows)
            for name, rows in self._resolve_map.items():
                if rows.size == 0:
                    continue
                inc_vec = amounts[rows, :].sum(axis=0)
                p_vec = self._prob_map.get(name, np.ones(self._L, dtype=np.float64))
                shift = int(self._delay_steps_map.get(name, 0))
                j = i + shift
                if j < n_steps:
                    series[name][j, :] += inc_vec * p_vec

        # Post-pass: resolve sums/aliases with own prob+delay (inline; no _apply_delay_prob)
        unresolved = set(self._sum_map.keys())
        guard = 0
        while unresolved and guard < 10000:
            guard += 1
            progress = False
            for name in list(unresolved):
                children = self._sum_map[name]
                if not all(ch in series for ch in children):
                    continue
                combined = np.zeros_like(series[name])
                for ch in children:
                    combined += series[ch]
                p_vec = self._prob_map.get(name, np.ones(self._L, dtype=np.float64))
                shift = int(self._delay_steps_map.get(name, 0))
                if shift:
                    out = np.zeros_like(combined)
                    if shift < combined.shape[0]:
                        out[shift:, :] = combined[:-shift, :] * p_vec
                else:
                    out = combined * p_vec
                series[name] = out
                unresolved.remove(name)
                progress = True
            if not progress:
                break
        if unresolved:
            raise RuntimeError(f"Unresolved sum outcomes remain: {unresolved}")

        # 5) Aggregate to MMWR weeks using step->week mapping
        assign_steps, n_weeks = _steps_to_weeks_assign(
            start=self.pipe.start_date, T_days=self.pipe.T - 1, T_steps=n_steps, dt_days=dt
        )
        weekly_age = np.zeros((len(self._age_labels), n_weeks, self._L), dtype=np.float64)
        for a_idx, age in enumerate(self._age_labels):
            step_sum = None
            for out_name in self._age_to_names[age]:
                if out_name in series:
                    arr = series[out_name]  # (n_steps, L)
                    step_sum = arr if step_sum is None else (step_sum + arr)
            if step_sum is None:
                step_sum = np.zeros((n_steps, self._L), dtype=np.float64)

            # sum by week id
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
    modifier_prior_specs: dict[str, dict[str, float]] | None = None,
) -> pm.Model:
    """
    Construct a PyMC model wrapping the Op; supports (A×W×L) or (W×L) observations.
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
    if modifier_prior_specs is None:
        modifier_prior_specs = {
            name: {"dist": "lognormal", "mu": float(np.log(val + 1e-12)), "sigma": 0.35}
            for name, val in zip(mod_names, defaults)
        }

    # Center for lambda_ext prior from base params (robust)
    try:
        pmap = getattr(pipeline, "param_name_to_idx", None) or getattr(pipeline.mod_applier, "param_name_to_idx", {})
        pidx_lex = int(pmap["lambda_ext"])
        lam_base = np.asarray(pipeline.base_params[pidx_lex, :, :], dtype=np.float64)  # (T, L)
        lambda_default = float(np.median(lam_base)) if np.isfinite(lam_base).any() else 2.5e-5
    except Exception:
        lambda_default = 2.5e-5

    with pm.Model(coords=coords) as m:
        # ----- Hierarchical modifiers
        mods_loc_list = []
        for i, name in enumerate(mod_names):
            spec = modifier_prior_specs.get(name, {})
            dist = str(spec.get("dist", "lognormal")).lower()
            if dist == "lognormal":
                mu_i = pm.Normal(f"{name}_mu", mu=float(spec.get("mu", np.log(defaults[i] + 1e-12))), sigma=0.5)
                sigma_i = pm.HalfNormal(f"{name}_sigma", sigma=float(spec.get("sigma", 0.35)))
                mods_i_loc = pm.LogNormal(f"{name}_loc", mu=mu_i, sigma=sigma_i, dims=("location",))
            elif dist == "gamma":
                mean0 = float(spec.get("mean", defaults[i]))
                sd0 = float(spec.get("sd", max(0.25 * defaults[i], 0.05)))
                mean_i = pm.HalfNormal(f"{name}_mean", sigma=2.0 * mean0)
                sd_i = pm.HalfNormal(f"{name}_sd", sigma=2.0 * sd0)
                alpha_i = pm.Deterministic(f"{name}_alpha", (mean_i / (sd_i + 1e-12)) ** 2)
                beta_i = pm.Deterministic(f"{name}_beta", mean_i / (sd_i ** 2 + 1e-12))
                mods_i_loc = pm.Gamma(f"{name}_loc", alpha=alpha_i, beta=beta_i, dims=("location",))
            else:
                raise ValueError(f"Unknown modifier prior dist for {name}: {dist}")
            mods_loc_list.append(mods_i_loc)

        mods_loc = pm.Deterministic(
            "mods_loc", pt.stack([pt.as_tensor_variable(v) for v in mods_loc_list], axis=0),
            dims=("modifier", "location"),
        )
        mods_vec = pm.Deterministic("mods", mods_loc.mean(axis=1), dims=("modifier",))

        # ----- Hierarchical logit-normal for pR
        mu0 = float(np.log(0.40 / (1.0 - 0.40)))
        mu = pm.Normal("mu_R_logit", mu=mu0, sigma=0.5)
        sigma_age = pm.HalfNormal("sigma_age", sigma=0.5)
        sigma_loc = pm.HalfNormal("sigma_loc", sigma=0.5)
        a_age = pm.Normal("a_age", mu=0.0, sigma=sigma_age, dims=("age",))
        b_loc = pm.Normal("b_loc", mu=0.0, sigma=sigma_loc, dims=("location",))
        eta = pm.Deterministic("eta_R_logit", mu + a_age[:, None] + b_loc[None, :], dims=("age", "location"))
        pR = pm.Deterministic("pR", pt.sigmoid(eta), dims=("age", "location"))

        # ----- Hierarchical log-normal for lambda_ext by location
        mu_lex = pm.Normal("lambda_ext_mu_log", mu=np.log(lambda_default + 1e-16), sigma=0.1)
        sd_lex = pm.HalfNormal("lambda_ext_sigma_log", sigma=0.1)
        lambda_ext_loc = pm.LogNormal("lambda_ext_loc", mu=mu_lex, sigma=sd_lex, dims=("location",))
        pm.Deterministic("inv_lambda_ext_loc", 1.0 / (lambda_ext_loc + 1e-16), dims=("location",))

        # ----- Forward op (single-solve simulator)
        weekly_pred_t, S_final_t = op(mods_vec, pR, lambda_ext_loc)
        weekly = pm.Deterministic("weekly_pred", weekly_pred_t, dims=("age", "week", "location"))
        weekly_sum_age = pm.Deterministic("weekly_pred_sum_age", weekly.sum(axis=0), dims=("week", "location"))
        S_final = pm.Deterministic("S_final", S_final_t, dims=("age", "location"))

        # ----- Likelihood (optional)
        if y_obs is not None:
            y_obs = np.asarray(y_obs, dtype=np.float64)
            alpha = pm.HalfNormal("alpha_nb", sigma=10.0)
            if y_obs.shape == (A, W, L):
                pm.NegativeBinomial("y", mu=weekly, alpha=alpha, observed=y_obs, dims=("age", "week", "location"))
            elif y_obs.shape == (W, L):
                pm.NegativeBinomial("y", mu=weekly_sum_age, alpha=alpha, observed=y_obs, dims=("week", "location"))
            else:
                raise ValueError(f"y_obs shape must be (A,W,L)=({A},{W},{L}) or (W,L)=({W},{L}); got {y_obs.shape}")
    return m


def build_op_and_model_from_config(
    config_path: str | Path,
    *,
    y_obs: np.ndarray | None = None,
    modifier_prior_specs: dict[str, dict[str, float]] | None = None,
) -> tuple[WeeklyHospPipeline, WeeklyHospAndFinalSOp, pm.Model]:
    """Convenience builder returning (pipeline, op, pymc.Model)."""
    pipe = build_pipeline_from_config(config_path)
    op = WeeklyHospAndFinalSOp(pipe)
    model = build_weekly_model(pipe, op, y_obs=y_obs, modifier_prior_specs=modifier_prior_specs)
    return pipe, op, model
