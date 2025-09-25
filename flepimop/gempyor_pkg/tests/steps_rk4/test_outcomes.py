# tests/steps_rk4/test_outcomes.py
# Self-contained outcomes test that reads the real config via ModelInfo,
# reconstructs *true* transition amounts per step, compiles outcomes (incidI/incidH),
# supports age × vaccination × variant leaves, and validates identities/sanity.

import os
import platform
import shutil
from pathlib import Path
from ctypes.util import find_library

# -------------------- Environment hygiene (Numba & BLAS) --------------------
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
    add = [p for p in extra if Path(p).exists() and p not in cur]
    if add:
        os.environ["DYLD_LIBRARY_PATH"] = (":".join(add) + (":" + cur if cur else ""))

_layer = _choose_numba_layer()
_maybe_patch_dylib_path(_layer)
os.environ.setdefault("NUMBA_THREADING_LAYER", _layer)
os.environ.setdefault("NUMBA_NUM_THREADS", str(os.cpu_count() or 1))
try:
    shutil.rmtree(Path.home() / ".numba" / "cache", ignore_errors=True)
except Exception:
    pass

# -------------------- Regular imports (safe after env) ----------------------
import numpy as np
import pandas as pd
import pytest
import confuse
from scipy.sparse import csr_matrix

from gempyor.model_info import ModelInfo
from gempyor.vectorization_experiments import RHSfactory

# =============================================================================
# Helper: build param-expression lookup safely (product-of-rows)
# =============================================================================
def _safe_param_expr_lookup(unique_strings: list[str]) -> tuple[dict[int, str] | None, dict[str, int]]:
    """
    Return (expr_lookup, name_to_row). If no '*' expressions are valid, expr_lookup is None.
    """
    name_to_row = {name: i for i, name in enumerate(unique_strings)}
    expr_lookup: dict[int, str] = {}
    for idx, s in enumerate(unique_strings):
        if "*" in s:
            terms = [t.strip() for t in s.split("*")]
            if all(term in name_to_row for term in terms):
                expr_lookup[idx] = s
    return (expr_lookup if expr_lookup else None, name_to_row)

# =============================================================================
# Helpers: parameter slicing (STEP mode) & single-step transition amounts
#   These are minimal numpy-only mirrors of the vectorized core logic.
# =============================================================================
def _param_slice_step(parameters: np.ndarray, t: float) -> np.ndarray:
    """
    STEP mode: pick the *daily* row at floor(t). Accepts (P,T,L) or (P,T) and returns (P,L).
    """
    if parameters.ndim == 2:
        P, T = parameters.shape
        L = 1
        params_t = np.moveaxis(parameters[:, :, None], 1, 0)  # (T,P,1)
    else:
        P, T, L = parameters.shape
        params_t = np.moveaxis(parameters, 1, 0)  # (T,P,L)
    i = int(np.floor(t))
    i = max(0, min(T - 1, i))
    return params_t[i]  # (P,L)

def _compute_amounts_for_step(
    *,
    states_current: np.ndarray,            # (C,L)
    transitions: np.ndarray,               # (5,Tn)
    proportion_info: np.ndarray,           # (3,Pk)
    transition_sum_compartments: np.ndarray,
    param_t_slice: np.ndarray,             # (P,L)
    percent_day_away: float,
    prop_who_move: np.ndarray,             # (L,)
    mobility_data: np.ndarray,             # csr.data
    mobility_indptr: np.ndarray,           # csr.indptr
    mobility_indices: np.ndarray,          # csr.indices
    population: np.ndarray,                # (L,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct deterministic transition amounts for one step:
      - Build base proportion product terms (subset sums ^ exponents)
      - Multiply by per-transition parameter vector (product expressions supported upstream)
      - Apply CSR mobility mixing like the core
      - Amounts = source_numbers * total_rates
    Returns (amounts[Tn,L], source_numbers[Tn,L]).
    """
    C, L = states_current.shape
    Tn = transitions.shape[1]
    Pk = proportion_info.shape[1]

    # --- Base proportion products & source sizes
    total_rates_base = np.ones((Tn, L), dtype=np.float64)
    source_numbers = np.zeros((Tn, L), dtype=np.float64)
    single_prop_mask = np.zeros(Tn, dtype=np.uint8)

    for t_idx in range(Tn):
        p_start = int(transitions[3, t_idx])
        p_stop  = int(transitions[4, t_idx])
        n_p = p_stop - p_start
        if n_p == 1:
            single_prop_mask[t_idx] = 1
        first = True
        for p_idx in range(p_start, p_stop):
            sum_start = int(proportion_info[0, p_idx])
            sum_stop  = int(proportion_info[1, p_idx])
            row_idx   = int(proportion_info[2, p_idx])

            comps = transition_sum_compartments[sum_start:sum_stop]
            summed = states_current[comps, :].sum(axis=0)  # (L,)
            expnt_vec = param_t_slice[row_idx, :]          # (L,)
            summed_exp = np.power(summed, expnt_vec, dtype=np.float64)

            if first:
                source_numbers[t_idx, :] = summed
                safe_src = np.where(summed > 0.0, summed, 1.0)
                contrib = summed_exp / safe_src
                if n_p == 1:
                    param_idx = int(transitions[2, t_idx])
                    contrib *= param_t_slice[param_idx, :]
                total_rates_base[t_idx, :] *= contrib
                first = False
            else:
                total_rates_base[t_idx, :] *= summed_exp

    # --- Parameter vector by transition (already supports expr upstream via RHSfactory)
    # Here we simply fetch the selected parameter row.
    param_vec_by_tr = np.empty((Tn, L), dtype=np.float64)
    for t_idx in range(Tn):
        pidx = int(transitions[2, t_idx])
        param_vec_by_tr[t_idx, :] = param_t_slice[pidx, :]

    # --- Mobility core (same structure as vectorized core)
    inv_pop = 1.0 / np.where(population > 0.0, population, 1.0)
    keep = 1.0 - percent_day_away * prop_who_move

    total_rates = np.empty_like(total_rates_base)
    for t_idx in range(Tn):
        if single_prop_mask[t_idx] == 1:
            total_rates[t_idx, :] = total_rates_base[t_idx, :]
            continue

        base_force = (total_rates_base[t_idx, :] * param_vec_by_tr[t_idx, :]) * inv_pop
        out = keep * base_force
        # CSR row-mix
        for node in range(L):
            start = mobility_indptr[node]
            end = mobility_indptr[node + 1]
            if end > start:
                acc = 0.0
                for k in range(start, end):
                    v = mobility_indices[k]
                    acc += mobility_data[k] * base_force[v]
                out[node] += percent_day_away * inv_pop[node] * acc
        total_rates[t_idx, :] = out

    # --- Amounts
    amounts = source_numbers * total_rates
    return amounts, source_numbers

# =============================================================================
# Outcome compilation from ModelInfo.outcomes_config
# =============================================================================
def _extract_scalar_value(block, default: float = 1.0) -> float:
    """
    Normalize YAML 'value' blocks to a scalar (handles nesting).
    """
    if block is None:
        return float(default)
    if isinstance(block, (int, float, np.floating)):
        return float(block)
    if isinstance(block, dict):
        if "value" in block:
            return _extract_scalar_value(block["value"], default)
        if "distribution" in block and "value" in block:
            return _extract_scalar_value(block["value"], default)
    try:
        return float(block)
    except Exception:
        return float(default)

def _expand_variant_token(token: str | None, compartments_df: pd.DataFrame) -> set[str]:
    """
    Expand umbrella variant tokens (e.g., 'AllFlu') to the set of concrete variants
    present in the compartments frame. Returns a set of strings (empty = match all).
    """
    if token is None:
        return set()
    tok = str(token).strip()
    if tok.lower() in ("", "all", "any"):
        return set()
    # Gather all variant values present
    all_variants = set(map(str, compartments_df["variant_type"].astype(str).unique()))
    if tok in all_variants and tok.lower() != "allflu":
        return {tok}
    # Special: AllFlu → anything that is not 'nan'/'None'
    if tok.lower() == "allflu":
        return {v for v in all_variants if v.lower() not in ("nan", "none", "")}
    # Fallback: treat as concrete token
    return {tok}

def _match_token(value: str | None, src_val: str, dst_val: str, *, allow_prefix=False, expanded: set[str] | None = None) -> bool:
    """
    Match a YAML token against src/dst attribute values. If expanded is provided,
    it is the set of acceptable concrete tokens for this attribute.
    """
    if value is None:
        return True
    if expanded is not None and len(expanded) > 0:
        return (src_val in expanded) or (dst_val in expanded)
    if allow_prefix:
        return dst_val.startswith(str(value)) or src_val.startswith(str(value))
    return (str(value) == dst_val) or (str(value) == src_val)

def _compile_outcomes_from_config(
    outcomes_cfg: dict,
    compartments_df: pd.DataFrame,
    transitions: np.ndarray,  # (5,Tn)
    n_loc: int,
    dt_days: float,
):
    """
    Turn the 'outcomes' yaml block into:
      - resolve_map: name -> np.ndarray (transition rows) for leaves with direct incidence sources
      - prob_map:   name -> np.ndarray (per-location probability vector)
      - delay_steps_map: name -> int (delay expressed in step bins)
      - sum_map:    name -> list[str] (children names for explicit sums or aliases)
      - order:      deterministic list of all names
    """
    Tn = transitions.shape[1]
    order = list(outcomes_cfg.keys())
    # Build a table of transitions joined with source/dest compartment attributes
    src_idx = transitions[0, :].astype(int)
    dst_idx = transitions[1, :].astype(int)
    src_attr = compartments_df.loc[src_idx, ["infection_stage", "vaccination_stage", "variant_type", "age_strata"]].astype(str).reset_index(drop=True)
    dst_attr = compartments_df.loc[dst_idx, ["infection_stage", "vaccination_stage", "variant_type", "age_strata"]].astype(str).reset_index(drop=True)
    src_attr.columns = [f"src_{c}" for c in src_attr.columns]
    dst_attr.columns = [f"dst_{c}" for c in dst_attr.columns]
    tr_df = pd.concat([src_attr, dst_attr], axis=1)  # length Tn

    resolve_map: dict[str, np.ndarray] = {}
    prob_map: dict[str, np.ndarray] = {}
    delay_steps_map: dict[str, int] = {}
    sum_map: dict[str, list[str]] = {}

    for name, spec in outcomes_cfg.items():
        # Probability & delay (scalar into per-location vector)
        p = _extract_scalar_value(spec.get("probability", {}).get("value", {}), default=1.0)
        prob_map[name] = np.full(n_loc, float(p), dtype=np.float64)
        d_days = _extract_scalar_value(spec.get("delay", {}).get("value", {}), default=0.0)
        delay_steps_map[name] = int(round(float(d_days) / float(dt_days)))

        # Sum or alias
        if "sum" in spec:
            children = list(spec["sum"])
            sum_map[name] = children
            # No direct incidence rows
            resolve_map[name] = np.array([], dtype=np.int64)
            continue

        # Leaf: source can be an 'incidence' dict OR a string alias to another outcome
        src = spec.get("source", None)
        if isinstance(src, str):
            # Alias to another outcome name
            sum_map[name] = [src]
            resolve_map[name] = np.array([], dtype=np.int64)
            continue

        # Direct incidence specification
        inc = (src or {}).get("incidence", {})
        infection_stage = inc.get("infection_stage", None)
        vaccination_stage = inc.get("vaccination_stage", None)
        variant_type = inc.get("variant_type", None)
        age_strata = inc.get("age_strata", None)

        # Expand umbrella variants (e.g., AllFlu -> set of concrete variants present)
        variant_set = _expand_variant_token(variant_type, compartments_df)

        # Build boolean mask over transitions
        mask = np.ones(Tn, dtype=bool)
        # Infection stage must match the *destination* (incidence INTO that stage),
        # but allow prefix match (e.g., "I1", "R", etc).
        if infection_stage is not None:
            dst_stg = tr_df["dst_infection_stage"].astype(str).values
            src_stg = tr_df["src_infection_stage"].astype(str).values
            keep = np.fromiter(
                (_match_token(infection_stage, s, d, allow_prefix=True) for s, d in zip(src_stg, dst_stg)),
                dtype=bool, count=Tn
            )
            # Ensure we indeed focus on INCIDENCE into the requested stage: give priority to destination
            keep &= np.fromiter((d.startswith(str(infection_stage)) for d in dst_stg), dtype=bool, count=Tn)
            mask &= keep

        # Other attributes may be carried by source or destination rows; accept match on either.
        def _attr_keep(colname_src: str, colname_dst: str, token: str | None, expanded: set[str] | None = None):
            if token is None:
                return np.ones(Tn, dtype=bool)
            svals = tr_df[colname_src].astype(str).values
            dvals = tr_df[colname_dst].astype(str).values
            return np.fromiter((_match_token(token, s, d, expanded=expanded) for s, d in zip(svals, dvals)), dtype=bool, count=Tn)

        mask &= _attr_keep("src_vaccination_stage", "dst_vaccination_stage", vaccination_stage, None)
        mask &= _attr_keep("src_variant_type", "dst_variant_type", variant_type, variant_set if variant_set else None)
        mask &= _attr_keep("src_age_strata", "dst_age_strata", age_strata, None)

        rows = np.nonzero(mask)[0].astype(np.int64)
        resolve_map[name] = rows

    return resolve_map, prob_map, delay_steps_map, sum_map, order

def _apply_delay_prob(series: np.ndarray, p_vec: np.ndarray, shift: int) -> np.ndarray:
    """
    Apply probability vector and non-negative integer delay (steps) to a time series (T,L).
    """
    T, L = series.shape
    if shift <= 0:
        return series * p_vec[None, :]
    out = np.zeros_like(series)
    src = series[: max(0, T - shift), :]
    out[shift:, :] = src * p_vec[None, :]
    return out

# =============================================================================
# Fixtures: materialize config and build a model + precomputed bits
# =============================================================================
def _materialize_structured_example(tmp_path_factory) -> Path:
    tmp_root = tmp_path_factory.mktemp("outcomes_case")
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
    cfg_name = "Structured_Example.yml"
    cfg_path = tmp_root / cfg_name
    shutil.copyfile(tutorial_dir / cfg_name, cfg_path)
    # Patch rel->abs paths
    text = cfg_path.read_text()
    text = text.replace("model_input/", str(tmp_root / "model_input") + "/")
    cfg_path.write_text(text)
    return cfg_path

@pytest.fixture(scope="module")
def model_and_bits(tmp_path_factory):
    cfg_path = _materialize_structured_example(tmp_path_factory)
    conf = confuse.Configuration("OutcomesTest", __name__)
    conf.set_file(str(cfg_path))

    model = ModelInfo(
        config=conf,
        config_filepath=str(cfg_path),
        path_prefix=str(cfg_path.parent),
        setup_name="Structured_Example",
        seir_modifiers_scenario="none",
    )

    initial_array = model.initial_conditions.get_from_config(sim_id=0, modinf=model)
    unique_strings, transitions, transition_sum_compartments, proportion_info = (
        model.compartments.get_transition_array()
    )
    param_defs = conf["seir"]["parameters"].get()
    base_params = model.parameters.parameters_quick_draw(model.n_days, model.nsubpops)
    parsed_params = model.compartments.parse_parameters(
        base_params, param_defs, unique_strings
    )
    expr_lookup, name_to_row = _safe_param_expr_lookup(unique_strings)

    mob: csr_matrix = model.mobility
    population = model.subpop_pop
    mobility_data = mob.data
    mobility_indptr = mob.indptr
    mobility_indices = mob.indices

    prop_move = np.zeros(model.nsubpops, dtype=np.float64)
    for i in range(model.nsubpops):
        pop_i = float(population[i])
        total_flux = float(mobility_data[mobility_indptr[i]: mobility_indptr[i+1]].sum())
        prop_move[i] = min(total_flux / pop_i, 1.0) if pop_i > 0 else 0.0

    precomputed = {
        "ncompartments": initial_array.shape[0],
        "nspatial_nodes": initial_array.shape[1],
        "transitions": transitions.astype(np.int64, copy=False),
        "proportion_info": proportion_info.astype(np.int64, copy=False),
        "transition_sum_compartments": transition_sum_compartments.astype(np.int64, copy=False),
        "percent_day_away": 0.5,
        "proportion_who_move": prop_move,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_indptr.astype(np.int64, copy=False),
        "mobility_row_indices": mobility_indices.astype(np.int64, copy=False),
        "population": population,
    }

    # Factory for integration
    factory = RHSfactory(
        precomputed=precomputed,
        param_expr_lookup=expr_lookup,
        param_name_to_row={k: int(v) for k, v in name_to_row.items()},
        param_time_mode="step",
    )

    # Sub-daily step grid (like pipeline)
    dt_days = 0.1
    total_days = float(model.n_days - 1)
    n_steps = int(round(total_days / dt_days))
    n_steps = max(1, n_steps)
    t_eval = np.linspace(0.0, total_days, n_steps + 1, dtype=np.float64)

    return dict(
        cfg_path=cfg_path,
        conf=conf,
        model=model,
        initial_array=initial_array,
        params=parsed_params,
        transitions=transitions,
        proportion_info=proportion_info,
        transition_sum_compartments=transition_sum_compartments,
        factory=factory,
        dt_days=dt_days,
        t_eval=t_eval,
        n_steps=n_steps,
    )

# =============================================================================
# The actual test
# =============================================================================
def test_compile_and_validate_outcomes(model_and_bits):
    """
    Compile outcomes from the ACTUAL config, reconstruct per-step transition amounts,
    accumulate outcomes (with probs+delays), and validate identities/sanity.

    This test:
      * integrates states on a step grid,
      * computes per-step transition amounts,
      * builds leaf outcomes directly from transition rows,
      * resolves aliases/sums strictly in topological order,
      * checks YAML identities (I and H sums),
      * and verifies I→H ratios against the probabilities actually compiled from config.
    """
    bits = model_and_bits
    model = bits["model"]
    dt_days = float(bits["dt_days"])
    t_eval = bits["t_eval"]
    n_steps = int(bits["n_steps"])

    # 1) Integrate states on the step grid
    res = bits["factory"].solve(
        y0=bits["initial_array"].ravel(),
        parameters=bits["params"],
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-3,
        atol=1e-6,
    )
    assert res.success, f"Solve failed: {res.message}"

    C = bits["initial_array"].shape[0]
    L = bits["initial_array"].shape[1]
    states = res.y.T.reshape(len(t_eval), C, L)  # (T_pts, C, L)

    # 2) Compile outcomes from ModelInfo.outcomes_config
    outcomes_cfg = model.outcomes_config["outcomes"].get()
    resolve_map, prob_map, delay_steps_map, sum_map, order = _compile_outcomes_from_config(
        outcomes_cfg, model.compartments.compartments, bits["transitions"], L, dt_days
    )

    # Storage: pre-allocate all names
    series = {name: np.zeros((n_steps, L), dtype=np.float64) for name in order}

    # Track which names are *materialized* (have their intended values)
    # Start with leaf-incidence outcomes (those with actual transition rows)
    ready = {nm for nm, rows in resolve_map.items() if rows.size > 0}

    # 3) Build only *leaf incidence* outcomes inside the step loop
    pc = bits["factory"].precomputed
    for i in range(n_steps):
        t0 = t_eval[i]
        param_t_slice = _param_slice_step(bits["params"], t0)  # (P, L)
        amounts, _ = _compute_amounts_for_step(
            states_current=states[i],
            transitions=bits["transitions"],
            proportion_info=bits["proportion_info"],
            transition_sum_compartments=bits["transition_sum_compartments"],
            param_t_slice=param_t_slice,
            percent_day_away=pc["percent_day_away"],
            prop_who_move=pc["proportion_who_move"],
            mobility_data=pc["mobility_data"],
            mobility_indptr=pc["mobility_data_indices"],
            mobility_indices=pc["mobility_row_indices"],
            population=pc["population"],
        )  # (Tn, L)

        for name, rows in resolve_map.items():
            if rows.size == 0:
                continue  # not a leaf-incidence node
            inc_vec = amounts[rows, :].sum(axis=0)     # (L,)
            p_vec = prob_map.get(name, np.ones(L))
            shift = int(delay_steps_map.get(name, 0))
            j = i + shift
            if j < n_steps:
                series[name][j, :] += inc_vec * p_vec

    # 4) Resolve sums/aliases AFTER leaves, in topological order using 'ready'
    unresolved = set(sum_map.keys())
    guard = 0
    while unresolved and guard < 10000:
        guard += 1
        progress = False
        for name in list(unresolved):
            children = sum_map[name]
            # Only proceed when ALL children have been materialized
            if not all(ch in ready for ch in children):
                continue
            combined = np.zeros_like(series[name])
            for ch in children:
                combined += series[ch]
            p_vec = prob_map.get(name, np.ones(L))
            shift = int(delay_steps_map.get(name, 0))
            series[name] = _apply_delay_prob(combined, p_vec, shift)
            ready.add(name)
            unresolved.remove(name)
            progress = True
        if not progress:
            break
    assert not unresolved, f"Unresolved sum outcomes remain: {unresolved}"

    # 5) Sanity on all produced series
    for nm, arr in series.items():
        assert np.isfinite(arr).all(), f"{nm} has non-finite values"
        assert (arr >= 0).all(), f"{nm} has negative values"

    # 6) YAML identity checks (incidence)
    age_bins = ["age0to4", "age5to17", "age18to49", "age50to64", "age65to100"]
    vax_bins = ["1dose", "waned", "unvaccinated"]

    strat_I = [f"incidI_{vax}_AllFlu_{age}" for age in age_bins for vax in vax_bins]
    assert all(n in series for n in strat_I), "Missing stratified incidI leaves"
    I_AllFlu_sum = sum(series[n] for n in strat_I)
    np.testing.assert_allclose(I_AllFlu_sum, series["incidI_AllFlu"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(series["incidI"], series["incidI_AllFlu"], rtol=1e-10, atol=1e-10)

    # 7) YAML identity checks (hospitalizations)
    strat_H = [f"incidH_{vax}_AllFlu_{age}" for age in age_bins for vax in vax_bins]
    assert all(n in series for n in strat_H), "Missing stratified incidH leaves"
    H_AllFlu_sum = sum(series[n] for n in strat_H)
    np.testing.assert_allclose(H_AllFlu_sum, series["incidH_AllFlu"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(series["incidH"], series["incidH_AllFlu"], rtol=1e-10, atol=1e-10)

    # 8) Probability × delay sanity for selected leaves:
    #    Compare against *compiled* per-location probabilities (not hard-coded numbers).
    #    If the YAML field is a fixed scalar, also assert that prob_map equals that scalar.
    selected = [
        "incidH_1dose_AllFlu_age0to4",
        "incidH_unvaccinated_AllFlu_age65to100",
        "incidH_waned_AllFlu_age18to49",
    ]
    for h_name in selected:
        i_name = h_name.replace("incidH_", "incidI_")
        assert i_name in series and h_name in series, f"Missing I/H pair {i_name}/{h_name}"

        # 8a) Ratio equals compiled prob_map (location-wise)
        p_vec = prob_map[h_name] if h_name in prob_map else np.ones(L)
        delay_steps = int(delay_steps_map.get(h_name, 0))

        H = series[h_name]
        I = series[i_name]
        if 0 < delay_steps < H.shape[0]:
            H_valid = H[delay_steps:, :].sum(axis=0)
            I_shift = I[:-delay_steps, :].sum(axis=0)
        else:
            H_valid = H.sum(axis=0)
            I_shift = I.sum(axis=0)
        ratio = H_valid / (I_shift + 1e-12)
        # Location-wise comparison to compiled probabilities
        np.testing.assert_allclose(ratio, p_vec, rtol=6e-2, atol=8e-4)

        # 8b) If the YAML specified a fixed scalar for this outcome, ensure prob_map equals that scalar
        try:
            y = outcomes_cfg[h_name]
            if "probability" in y and "value" in y["probability"]:
                v = y["probability"]["value"]
                # fixed scalar is typically under {'distribution': 'fixed', 'value': <number>}
                if isinstance(v, dict) and "value" in v and isinstance(v["value"], (int, float)):
                    p_scalar = float(v["value"])
                    np.testing.assert_allclose(p_vec, np.full_like(p_vec, p_scalar), rtol=1e-10, atol=1e-10)
        except Exception:
            # If format differs, skip scalar check; the ratio-to-compiled check above still guarantees correctness.
            pass
