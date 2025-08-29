# src/gempyor/hosp_weekly_pipeline.py
from __future__ import annotations

import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import datetime as dt

import numpy as np
import confuse
from scipy.sparse import csr_matrix

from gempyor.model_info import ModelInfo
from gempyor.vectorization_experiments import (
    RHSfactory,
    # proportion + exponents
    _compute_proportion_sums_exponents_manual,
    _compute_proportion_sums_exponents_from_sums,  # imported but not used; kept for parity
    # mobility + param scaling
    _compute_transition_rates,
    # elementwise hotspot (autotuned inside factory)
    compute_transition_amounts_meta,
)

# ==========================================================
# Helpers copied/adapted from testing utilities (self-contained)
# ==========================================================

def _safe_param_expr_lookup(unique_strings: list[str]):
    name_to_row = {name: i for i, name in enumerate(unique_strings)}
    expr_lookup: dict[int, str] = {}
    for idx, s in enumerate(unique_strings):
        if "*" in s:
            terms = [t.strip() for t in s.split("*")]
            if all(term in name_to_row for term in terms):
                expr_lookup[idx] = s
    return (expr_lookup if expr_lookup else None, name_to_row)


def _make_incidence_resolver(compartments_df, transitions_arr):
    """
    Build a function that resolves a source.incidence filter into transition-row indices.
    We treat the source as the *destination* compartment(s) of a transition whose
    infection_stage (prefix match allowed), vaccination_stage, variant_type, age_strata match.
    """
    dst_row = transitions_arr[1, :].astype(np.int64)

    def _resolve(infection_stage=None, vaccination_stage=None, variant_type=None, age_strata=None) -> np.ndarray:
        mask = np.ones(len(compartments_df), dtype=bool)
        if infection_stage is not None:
            cs = compartments_df["infection_stage"].astype(str)
            # allow "I" to match "I1", "I2", etc.
            mask &= (cs == str(infection_stage)) | cs.str.startswith(str(infection_stage))
        if vaccination_stage is not None:
            mask &= compartments_df["vaccination_stage"].astype(str).eq(str(vaccination_stage))
        if variant_type is not None:
            mask &= compartments_df["variant_type"].astype(str).eq(str(variant_type))
        if age_strata is not None:
            mask &= compartments_df["age_strata"].astype(str).eq(str(age_strata))
        dst_comp_indices = compartments_df.index.values[mask]
        return np.nonzero(np.isin(dst_row, dst_comp_indices))[0].astype(np.int64, copy=False)

    return _resolve


def _extract_prob(spec: dict | float | int | None) -> float:
    """Pull a scalar probability from the YAML node; default 1.0."""
    if spec is None:
        return 1.0
    if isinstance(spec, (float, int)):
        return float(spec)
    # typical: {"value": {"distribution": "fixed", "value": X}}
    v = spec.get("value", None)
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, dict):
        inner = v.get("value", v.get("val", v.get("mult", None)))
        if isinstance(inner, (float, int)):
            return float(inner)
    return 1.0


def _extract_delay_days(spec: dict | float | int | None) -> float:
    """Pull a scalar delay in *days* from the YAML node; default 0."""
    if spec is None:
        return 0.0
    if isinstance(spec, (float, int)):
        return float(spec)
    v = spec.get("value", None)
    if isinstance(v, (float, int)):
        return float(v)
    if isinstance(v, dict):
        inner = v.get("value", v.get("val", v.get("days", None)))
        if isinstance(inner, (float, int)):
            return float(inner)
    return 0.0


def _compile_outcomes_from_config(
    outcomes_cfg: dict[str, dict],
    compartments_df,
    transitions: np.ndarray,
    n_nodes: int,
    bin_width_days: float,
) -> tuple[
    dict[str, np.ndarray],      # resolve_map: outcome -> rows (Tn,) (leaf-incidence only; others -> empty)
    dict[str, np.ndarray],      # prob_map  : outcome -> (L,) vector
    dict[str, int],             # delay_steps_map: outcome -> int shift in *steps*
    dict[str, list[str]],       # sum_map   : outcome -> list of child outcome names
    list[str],                  # order     : stable order of all names
]:
    """
    Parse the YAML 'outcomes' block into minimal runtime instructions:
    - Leaves with incidence sources map to explicit transition rows.
    - Nodes that alias another outcome via `source: <name>` are represented in sum_map.
    - Nodes that sum over a list keep their children in sum_map.
    - Every node records its own probability (as vector of length L) and delay in steps.
    """
    resolve_rows: dict[str, np.ndarray] = {}
    prob_map: dict[str, np.ndarray] = {}
    delay_steps: dict[str, int] = {}
    sum_map: dict[str, list[str]] = {}
    names_in_order: list[str] = []

    resolver = _make_incidence_resolver(compartments_df, transitions)

    for name, spec in outcomes_cfg.items():
        if not isinstance(spec, dict):
            continue
        names_in_order.append(name)

        # Probability vector (per node, per location; keep uniform unless config states otherwise)
        p = _extract_prob(spec.get("probability"))
        prob_map[name] = np.full(n_nodes, float(p), dtype=np.float64)

        # Delay in *steps*: convert days -> steps using the bin width
        d_days = _extract_delay_days(spec.get("delay"))
        shift = int(round(float(d_days) / float(bin_width_days))) if bin_width_days > 0 else 0
        delay_steps[name] = int(max(0, shift))

        # Resolve source
        if "sum" in spec:
            children = spec["sum"]
            if isinstance(children, (list, tuple)):
                sum_map[name] = [str(c) for c in children]
            else:
                raise ValueError(f"Outcome '{name}' has non-list 'sum'.")
            resolve_rows[name] = np.array([], dtype=np.int64)
            continue

        source = spec.get("source", None)
        if source is None:
            resolve_rows[name] = np.array([], dtype=np.int64)
            continue

        # Two cases: (a) a dict like {"incidence": {...filters...}}
        #            (b) a string: another outcome name (alias)
        if isinstance(source, str):
            # treat as alias -> a sum of one child
            sum_map[name] = [source]
            resolve_rows[name] = np.array([], dtype=np.int64)
            continue

        if isinstance(source, dict) and "incidence" in source:
            filt = source["incidence"]
            rows = resolver(
                infection_stage=filt.get("infection_stage"),
                vaccination_stage=filt.get("vaccination_stage"),
                variant_type=filt.get("variant_type"),
                age_strata=filt.get("age_strata"),
            )
            resolve_rows[name] = np.asarray(rows, dtype=np.int64)
            if resolve_rows[name].size == 0:
                # Don't crash; keep empty, but warn in logs if desired
                pass
            continue

        # If we reach here, format is unsupported; register as empty (will be caught later if referenced)
        resolve_rows[name] = np.array([], dtype=np.int64)

    # Ensure every sum target exists in dicts
    for name, children in list(sum_map.items()):
        for ch in children:
            if ch not in resolve_rows:
                # create a stub so topo pass can report cleanly if truly missing later
                resolve_rows[ch] = np.array([], dtype=np.int64)
                prob_map[ch] = np.ones(n_nodes, dtype=np.float64)
                delay_steps[ch] = 0

    return resolve_rows, prob_map, delay_steps, sum_map, names_in_order


def _param_slice_step(parameters: np.ndarray, t: float) -> np.ndarray:
    """
    Step-mode slice of parameters at floor(t) day.

    Supports shapes:
      (P, T, L) -> return (P, L)
      (P, T)    -> return (P, 1)  (broadcast at call site if needed)
    """
    if parameters.ndim == 3:
        P, T, L = parameters.shape
        i = int(np.clip(np.floor(t), 0, T - 1))
        return parameters[:, i, :]
    elif parameters.ndim == 2:
        P, T = parameters.shape
        i = int(np.clip(np.floor(t), 0, T - 1))
        return parameters[:, i : i + 1]
    else:
        raise ValueError(f"Unsupported parameter shape for step slicing: {parameters.shape}")


def _compute_amounts_for_step(
    *,
    states_current: np.ndarray,                 # (C, L)
    transitions: np.ndarray,                    # (5, Tn)
    proportion_info: np.ndarray,                # (3, Pk)
    transition_sum_compartments: np.ndarray,    # (S,)
    param_t_slice: np.ndarray,                  # (P, L)
    percent_day_away: float,
    prop_who_move: np.ndarray,                  # (L,)
    mobility_data: np.ndarray,                  # (nnz,)
    mobility_indptr: np.ndarray,                # (L+1,)
    mobility_indices: np.ndarray,               # (nnz,)
    population: np.ndarray,                     # (L,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    For a single step time t, compute:
      - transition amounts (Tn, L)
      - source_numbers (Tn, L)  (useful for diagnostics)
    Mirrors the factory's internals but avoids selector autotune for simplicity.
    """
    total_base, source_numbers, single_prop_mask = _compute_proportion_sums_exponents_manual(
        states_current, transitions, proportion_info, transition_sum_compartments, param_t_slice
    )
    total_rates = _compute_transition_rates(
        total_rates_base=total_base,
        source_numbers=source_numbers,
        transitions=transitions,
        param_t=param_t_slice,
        percent_day_away=float(percent_day_away),
        proportion_who_move=np.asarray(prop_who_move, dtype=np.float64),
        mobility_data=np.asarray(mobility_data, dtype=np.float64),
        mobility_data_indices=np.asarray(mobility_indptr, dtype=np.int64),
        mobility_row_indices=np.asarray(mobility_indices, dtype=np.int64),
        population=np.asarray(population, dtype=np.float64),
        single_prop_mask=np.asarray(single_prop_mask, dtype=np.uint8),
        param_expr_lookup=None,
        param_name_to_row=None,
    )
    amounts = compute_transition_amounts_meta(source_numbers, total_rates)
    return amounts, source_numbers


# ==========================================================
# Time/aggregation helpers (MMWR weeks)
# ==========================================================

def _mmwr_assign_from_daily(start: dt.date, T_days: int) -> tuple[np.ndarray, int]:
    """
    Build week assignments (0..W-1) for T_days consecutive *daily* bins starting at `start`,
    aligned to MMWR (Sunday starts).
    """
    if T_days <= 0:
        return np.zeros(0, dtype=np.int64), 0
    offset_to_sun = (6 - start.weekday()) % 7  # Monday=0..Sunday=6
    first_len = 7 if offset_to_sun == 0 else offset_to_sun
    first_len = min(first_len, T_days)

    assign = np.empty(T_days, dtype=np.int64)
    assign[:first_len] = 0
    if T_days > first_len:
        rest = T_days - first_len
        assign[first_len:] = 1 + (np.arange(rest, dtype=np.int64) // 7)
    n_weeks = int(assign.max()) + 1
    return assign, n_weeks


def _steps_to_weeks_assign(start: dt.date, T_days: int, T_steps: int, dt_days: float) -> tuple[np.ndarray, int]:
    """
    Map each *step interval* (length dt_days) to an MMWR week index by day-of-step-start.
    """
    assign_days, n_weeks = _mmwr_assign_from_daily(start, T_days)
    if T_steps <= 0:
        return np.zeros(0, dtype=np.int64), n_weeks
    day_idx = np.floor(np.arange(T_steps, dtype=np.float64) * float(dt_days)).astype(np.int64)
    day_idx = np.clip(day_idx, 0, max(0, T_days - 1))
    return assign_days[day_idx], n_weeks


# ==========================================================
# Age group extraction (used to aggregate outputs)
# ==========================================================

def _age_group_hosp_outcomes(outcomes_cfg: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, spec in outcomes_cfg.items():
        if not isinstance(name, str):
            continue
        if not name.startswith("incidH_"):
            continue
        parts = name.split("_")
        age_tokens = [p for p in parts if p.startswith("age")]
        if not age_tokens:
            continue
        age = age_tokens[-1]
        groups.setdefault(age, []).append(name)
    # stable order by age token
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


# ==========================================================
# Main pipeline
# ==========================================================

class WeeklyHospPipeline:
    """
    Build once; call evaluate(mod_values) -> (A, W, L) hospitalization incidence.

    A: # age groups parsed from config's incidH_* leaves
    W: # MMWR weeks covering the series length from the solver step grid
    L: # locations
    """

    def __init__(self, config_path: str | Path, *, dt_days: float = 0.1):
        self.config_path = Path(config_path)
        self.dt = float(dt_days)  # integration & outcome step (e.g., 0.1 day)

        conf = confuse.Configuration("WeeklyHospPipeline", __name__)
        conf.set_file(str(self.config_path))

        self.model = ModelInfo(
            config=conf,
            config_filepath=str(self.config_path),
            path_prefix=str(self.config_path.parent),
            setup_name=conf["setup_name"].get(str) if "setup_name" in conf else "Structured_Example",
            seir_modifiers_scenario="none",
        )

        # ---- Core model pieces
        self.initial_array = self.model.initial_conditions.get_from_config(sim_id=0, modinf=self.model)
        self.unique_strings, self.transitions, self.transition_sum_compartments, self.proportion_info = (
            self.model.compartments.get_transition_array()
        )
        self.NC, self.NL = self.initial_array.shape
        self.start_date: dt.date = self.model.ti
        self.end_date: dt.date = self.model.tf
        # number of *calendar days* (inclusive)
        self.T = (self.end_date - self.start_date).days + 1

        # Parameters (P, T_days, L)
        self.param_defs = conf["seir"]["parameters"].get()
        self.param_names = np.array(list(self.param_defs.keys()))
        base_params = self.model.parameters.parameters_quick_draw(self.T, self.NL)
        self.base_params = self.model.compartments.parse_parameters(
            base_params, self.param_defs, self.unique_strings
        )

        # Mobility / precomputed
        mob: csr_matrix = self.model.mobility
        self.population = self.model.subpop_pop
        self.mobility_data = mob.data
        self.mobility_indptr = mob.indptr
        self.mobility_indices = mob.indices
        prop_move = np.zeros(self.NL, dtype=np.float64)
        for i in range(self.NL):
            pop_i = float(self.population[i])
            total_flux = float(self.mobility_data[self.mobility_indptr[i] : self.mobility_indptr[i + 1]].sum())
            prop_move[i] = min(total_flux / pop_i, 1.0) if pop_i > 0 else 0.0
        self.prop_move = prop_move

        # ---------- Optional seeding wiring ----------
        # These attributes are useful for debugging/inspection.
        self.seeding_on = False
        self.seeding_amounts = None
        self.seeding_data = None

        # Try to pull seeding from the config/model; if available, add to precomputed.
        # Shapes follow the seeding tests: daily_incidence is (T_days, NC, NL).
        try:
            seeding_nb_dict, seeding_amounts = self.model.get_seeding_data(sim_id=0)
            # Convert numba-dicts to plain numpy arrays with safe dtypes
            def _to_py(d):
                return {str(k): np.ascontiguousarray(v) for k, v in d.items()}

            sd = _to_py(seeding_nb_dict)
            seeding_data = {
                "day_start_idx": np.ascontiguousarray(sd["day_start_idx"], dtype=np.int64),
                "seeding_subpops": np.ascontiguousarray(sd["seeding_subpops"], dtype=np.int64),
                "seeding_sources": np.ascontiguousarray(sd["seeding_sources"], dtype=np.int64),
                "seeding_destinations": np.ascontiguousarray(sd["seeding_destinations"], dtype=np.int64),
            }
            seeding_amounts = np.ascontiguousarray(seeding_amounts, dtype=np.float64)
            daily_incidence = np.zeros((self.T, self.NC, self.NL), dtype=np.float64)

            self.seeding_on = True
            self.seeding_amounts = seeding_amounts
            self.seeding_data = seeding_data
        except Exception:
            # No seeding configured; proceed without it.
            seeding_data = None
            seeding_amounts = None
            daily_incidence = None
            self.seeding_on = False

        # Base precomputed for RHSfactory
        self.precomputed = {
            "ncompartments": self.NC,
            "nspatial_nodes": self.NL,
            "transitions": self.transitions.astype(np.int64, copy=False),
            "proportion_info": self.proportion_info.astype(np.int64, copy=False),
            "transition_sum_compartments": self.transition_sum_compartments.astype(np.int64, copy=False),
            "percent_day_away": 0.5,
            "proportion_who_move": self.prop_move,
            "mobility_data": self.mobility_data,
            "mobility_data_indices": self.mobility_indptr.astype(np.int64, copy=False),
            "mobility_row_indices": self.mobility_indices.astype(np.int64, copy=False),
            "population": self.population,
        }
        # Inject seeding keys only if available
        if self.seeding_on:
            self.precomputed.update(
                {
                    "seeding_data": seeding_data,
                    "seeding_amounts": seeding_amounts,
                    "daily_incidence": daily_incidence,
                }
            )

        # Solver factory (autotune preserved)
        self.param_expr_lookup, self.param_name_to_row__unique = _safe_param_expr_lookup(self.unique_strings)
        self.factory = RHSfactory(
            precomputed=self.precomputed,
            param_expr_lookup=self.param_expr_lookup,
            param_name_to_row={k: int(v) for k, v in self.param_name_to_row__unique.items()},
            param_time_mode="step",
        )

        # ---- Outcomes: compile from ACTUAL config (leaf incidence + sums/aliases)
        self.outcomes_cfg = conf["outcomes"]["outcomes"].get()
        (
            self._out_resolve_rows,
            self._out_prob_map,
            self._out_delay_steps,
            self._out_sum_map,
            self._out_all_names,
        ) = _compile_outcomes_from_config(
            self.outcomes_cfg,
            self.model.compartments.compartments,
            self.transitions,
            self.NL,
            bin_width_days=self.dt,
        )

        # Age groups for aggregation
        self.age_to_names = _age_group_hosp_outcomes(self.outcomes_cfg)
        self.age_labels = list(self.age_to_names.keys())
        if not self.age_labels:
            raise ValueError("No age-specific hospitalization outcomes (incidH_*_age...) found.")

        # Modifiers
        from gempyor.vectorized_modifiers import compile_seir_modifiers  # local import to avoid cycles
        self.mod_applier = compile_seir_modifiers(
            seir_modifiers_cfg=conf["seir_modifiers"].get(),
            start_date=self.start_date,
            n_days=self.T,
            n_loc=self.NL,
            param_names=self.param_names,
        )
        self.leaf_order = self.mod_applier.list_leaf_modifiers()


    # -------------------------- public API --------------------------

    def modifier_order(self) -> tuple[str, ...]:
        return self.leaf_order

    def evaluate(
        self,
        mod_values: np.ndarray,
        *,
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ) -> tuple[np.ndarray, list[str], int]:
        """
        Apply *replacing* modifier values (aligned to self.modifier_order()), run model,
        and return age × weeks × locations hospitalization incidence.

        Notes:
            • Probability and delay are applied per outcome node.
            • Nodes with `source: 'other_name'` are handled as aliases via the sum pass.
            • Sums of many children are resolved after the leaf-incidence pass.
            • Weekly aggregation follows MMWR (Sunday starts) using step-start days.
        """
        mod_values = np.asarray(mod_values, dtype=np.float64)
        if mod_values.shape != (len(self.leaf_order),):
            raise ValueError(f"mod_values must be length {len(self.leaf_order)} in order {self.leaf_order}")

        # 1) Replace each leaf's multiplier with provided values
        params_mod = self.mod_applier.apply_to_params(
            self.base_params,
            leaf_value_array=mod_values,
            scenario="none",
        )

        # 2) Integrate on a sub-daily grid covering [0 .. T-1] days (open end)
        total_days = float(self.T - 1)
        n_steps = int(round(total_days / self.dt))
        n_steps = max(1, n_steps)
        t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

        res = self.factory.solve(
            y0=self.initial_array.ravel(),
            parameters=params_mod,
            t_span=(t_eval[0], t_eval[-1]),
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not res.success:
            raise RuntimeError(f"Integration failed: {res.message}")

        states = res.y.T.reshape(len(t_eval), self.NC, self.NL)  # (T_pts, C, L)

        # 3) Build only leaf incidence outcomes on the step grid, then resolve sums/aliases
        series = {name: np.zeros((n_steps, self.NL), dtype=np.float64) for name in self._out_all_names}

        for i in range(n_steps):
            t0 = t_eval[i]
            param_t_slice = _param_slice_step(params_mod, t0)  # (P, L)
            amounts, _src = _compute_amounts_for_step(
                states_current=states[i],
                transitions=self.transitions,
                proportion_info=self.proportion_info,
                transition_sum_compartments=self.transition_sum_compartments,
                param_t_slice=param_t_slice,
                percent_day_away=self.precomputed["percent_day_away"],
                prop_who_move=self.prop_move,
                mobility_data=self.mobility_data,
                mobility_indptr=self.mobility_indptr,
                mobility_indices=self.mobility_indices,
                population=self.population,
            )  # (Tn, L)

            for name, rows in self._out_resolve_rows.items():
                if rows.size == 0:
                    continue  # not a leaf-incidence node
                inc_vec = amounts[rows, :].sum(axis=0)  # (L,)
                p_vec = self._out_prob_map.get(name, np.ones(self.NL, dtype=np.float64))
                shift = int(self._out_delay_steps.get(name, 0))
                j = i + shift
                if j < n_steps:
                    series[name][j, :] += inc_vec * p_vec

        # 4) Resolve sums/aliases in a topological loop (children must exist first)
        unresolved = set(self._out_sum_map.keys())
        guard = 0
        while unresolved and guard < 10000:
            guard += 1
            progress = False
            for name in list(unresolved):
                children = self._out_sum_map[name]
                # Only proceed when ALL children arrays are materialized in 'series'
                if not all(ch in series for ch in children):
                    continue
                combined = np.zeros_like(series[name])
                for ch in children:
                    combined += series[ch]
                p_vec = self._out_prob_map.get(name, np.ones(self.NL, dtype=np.float64))
                shift = int(self._out_delay_steps.get(name, 0))
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

        # 5) Weekly aggregation per age group
        assign_steps, n_weeks = _steps_to_weeks_assign(
            start=self.start_date,
            T_days=self.T - 1,  # step-start days cover [0..T-2]
            T_steps=n_steps,
            dt_days=self.dt,
        )

        def _sum_to_weeks(step_arr: np.ndarray) -> np.ndarray:
            """Sum (n_steps, L) into (W, L) using assign_steps."""
            W = int(n_weeks)
            L = step_arr.shape[1]
            out = np.zeros((W, L), dtype=np.float64)
            for w in range(W):
                mask = (assign_steps == w)
                if mask.any():
                    out[w, :] = step_arr[mask, :].sum(axis=0)
            return out

        A = len(self.age_labels)
        weekly_age = np.zeros((A, n_weeks, self.NL), dtype=np.float64)

        # Primary hospitalization totals per age group use "incidH_*_age..." leaves
        for a_idx, age in enumerate(self.age_labels):
            # Sum across all leaf names tied to this age
            step_sum = None
            for out_name in self.age_to_names[age]:
                if out_name in series:
                    arr = series[out_name]
                    step_sum = arr if step_sum is None else (step_sum + arr)
            if step_sum is None:
                step_sum = np.zeros((n_steps, self.NL), dtype=np.float64)
            weekly_age[a_idx, :, :] = _sum_to_weeks(step_sum)

        return weekly_age, self.age_labels, n_weeks


# -------------------------- convenience entrypoint --------------------------

def build_pipeline_from_config(config_path: str | Path, *, dt_days: float = 0.1) -> WeeklyHospPipeline:
    return WeeklyHospPipeline(config_path, dt_days=dt_days)
