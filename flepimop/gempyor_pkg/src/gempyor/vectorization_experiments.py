import os
import math
import time
from typing import Callable, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

# =============================================================================
# Global config (autotune will modify these)
# =============================================================================

# Workload (Tn*N) threshold to switch to parallel kernel in elementwise hotspot
_PARALLEL_THRESHOLD = 10_000_000

# Global switches the autotuner sets:
_FORCE_SERIAL = False           # force serial kernel regardless of workload
_FASTMATH_ENABLED = True        # choose fastmath or not for elementwise hotspot

# BLAS/Numba environment defaults to avoid oversubscription (can be overridden)
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "tbb")  # or "omp"

def set_parallel_threshold(n: int) -> None:
    global _PARALLEL_THRESHOLD
    _PARALLEL_THRESHOLD = int(n)

def set_force_serial(flag: bool) -> None:
    global _FORCE_SERIAL
    _FORCE_SERIAL = bool(flag)

def set_fastmath_enabled(flag: bool) -> None:
    global _FASTMATH_ENABLED
    _FASTMATH_ENABLED = bool(flag)

def get_autotune_config() -> dict:
    return {
        "threshold": _PARALLEL_THRESHOLD,
        "force_serial": _FORCE_SERIAL,
        "fastmath": _FASTMATH_ENABLED,
    }

# =============================================================================
# Utilities (timing, helpers)
# =============================================================================

def _timeit(fn, *args, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats

def _make_rect_from_size(total_elems: int) -> Tuple[int, int]:
    m = max(64, int(math.sqrt(total_elems)))
    n = max(64, total_elems // m)
    return m, n

def _max_rel_err(a: np.ndarray, b: np.ndarray, eps: float = 1e-300) -> float:
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < eps, 1.0, denom)
    return float(np.max(np.abs(a - b) / denom))

# =============================================================================
# Setup helpers (one-time)
# =============================================================================

def _prep_param_interpolator(parameters: npt.NDArray[np.float64], use_deltas: bool = False):
    if parameters.ndim == 2:
        parameters = parameters[:, :, None]  # (P,T) -> (P,T,1)
    params_t = np.moveaxis(parameters, 1, 0).copy(order="C")  # (T,P,N)
    deltas = None
    if use_deltas and params_t.shape[0] >= 2:
        deltas = np.ascontiguousarray(params_t[1:] - params_t[:-1])  # (T-1,P,N)
    return params_t, deltas

@njit(cache=True, fastmath=True)
def _blend_linear(A: np.ndarray, B: np.ndarray, alpha: float, out: np.ndarray) -> None:
    P, N = out.shape
    for p in range(P):
        for n in range(N):
            out[p, n] = A[p, n] + alpha * (B[p, n] - A[p, n])

@njit(cache=True, fastmath=True)
def _axpy_linear(A: np.ndarray, D: np.ndarray, alpha: float, out: np.ndarray) -> None:
    P, N = out.shape
    for p in range(P):
        for n in range(N):
            out[p, n] = A[p, n] + alpha * D[p, n]

def _param_slice(
    params_t: npt.NDArray[np.float64], t: float, mode: str = "linear",
    out: npt.NDArray[np.float64] | None = None, deltas: npt.NDArray[np.float64] | None = None
) -> npt.NDArray[np.float64]:
    T, P, N = params_t.shape
    if out is None:
        out = np.empty((P, N), dtype=np.float64)

    if mode == "step":
        i = int(t)
        i = 0 if i < 0 else (T - 1 if i >= T else i)
        np.copyto(out, params_t[i])
        return out

    if mode == "linear":
        if T == 1:
            np.copyto(out, params_t[0])
            return out
        i0 = int(t)
        if i0 < 0:
            i0 = 0; alpha = 0.0
        elif i0 >= T - 1:
            i0 = T - 2; alpha = 1.0
        else:
            alpha = t - i0
        if deltas is None:
            _blend_linear(params_t[i0], params_t[i0 + 1], float(alpha), out)
        else:
            _axpy_linear(params_t[i0], deltas[i0], float(alpha), out)
        return out

    raise ValueError(f"Unknown param_time_mode: {mode}")

# =============================================================================
# Helper for symbolic parameter expressions (from a time slice)
# =============================================================================

def _compile_param_expr(expr: str, param_name_to_row: dict[str, int]) -> np.ndarray:
    terms = [s.strip() for s in expr.split("*") if s.strip()]
    if not terms:
        raise ValueError("Empty parameter expression.")
    try:
        rows = np.fromiter((param_name_to_row[t] for t in terms), dtype=np.int64)
    except KeyError as e:
        raise KeyError(f"Unknown parameter in expression: {e}") from None
    return rows

@njit(cache=True, fastmath=True)
def _product_rows_into(param_t: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
    K = rows.shape[0]
    N = param_t.shape[1]
    r0 = rows[0]
    for n in range(N):
        out[n] = param_t[r0, n]
    for k in range(1, K):
        rk = rows[k]
        for n in range(N):
            out[n] *= param_t[rk, n]

def _resolve_param_expr_from_slice(
    expr: str | np.ndarray,
    param_t: npt.NDArray[np.float64],
    param_name_to_row: dict[str, int] | None = None,
    out: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    if isinstance(expr, str):
        if param_name_to_row is None:
            raise ValueError("param_name_to_row must be provided when expr is a string.")
        rows = _compile_param_expr(expr, param_name_to_row)
    else:
        rows = expr
    if out is None:
        out = np.empty(param_t.shape[1], dtype=np.float64)
    if not param_t.flags.c_contiguous:
        param_t = np.ascontiguousarray(param_t)
    _product_rows_into(param_t, rows, out)
    return out

# =============================================================================
# 1) Core Proportion Logic
#    Two implementations:
#    A) Manual Numba subset sums (original)
#    B) Using precomputed sums (produced by dense GEMM or sparse SpMM)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _compute_proportion_sums_exponents_manual(
    states_current: npt.NDArray[np.float64],       # (C, N)
    transitions: npt.NDArray[np.int64],            # (5, Tn)
    proportion_info: npt.NDArray[np.int64],        # (3, Pk)
    transition_sum_compartments: npt.NDArray[np.int64],  # (S,)
    param_t: npt.NDArray[np.float64],              # (P, N)
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]
    total_rates = np.ones((n_transitions, n_nodes), dtype=np.float64)
    source_numbers = np.zeros((n_transitions, n_nodes), dtype=np.float64)
    single_prop_mask = np.zeros(n_transitions, dtype=np.uint8)

    for t_idx in prange(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop  = transitions[4, t_idx]
        n_p = p_stop - p_start
        if n_p == 1:
            single_prop_mask[t_idx] = 1
        first = True
        for p_idx in range(p_start, p_stop):
            sum_start = proportion_info[0, p_idx]
            sum_stop  = proportion_info[1, p_idx]
            row_idx   = proportion_info[2, p_idx]

            summed = states_current[transition_sum_compartments[sum_start:sum_stop], :].sum(axis=0)
            expnt_vec = param_t[row_idx, :]
            summed_exp = summed ** expnt_vec

            if first:
                source_numbers[t_idx, :] = summed
                safe_src = np.where(summed > 0.0, summed, 1.0)
                contrib = summed_exp / safe_src
                if n_p == 1:
                    param_idx = transitions[2, t_idx]
                    contrib *= param_t[param_idx, :]
                for n in range(n_nodes):
                    total_rates[t_idx, n] *= contrib[n]
                first = False
            else:
                for n in range(n_nodes):
                    total_rates[t_idx, n] *= summed_exp[n]

    return total_rates, source_numbers, single_prop_mask

@njit(parallel=True, cache=True, fastmath=True)
def _compute_proportion_sums_exponents_from_sums(
    sums_all: npt.NDArray[np.float64],            # (Pk, N) precomputed sums
    transitions: npt.NDArray[np.int64],           # (5, Tn)
    proportion_info: npt.NDArray[np.int64],       # (3, Pk)
    param_t: npt.NDArray[np.float64],             # (P, N)
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    n_transitions = transitions.shape[1]
    n_nodes = sums_all.shape[1]
    total_rates = np.ones((n_transitions, n_nodes), dtype=np.float64)
    source_numbers = np.zeros((n_transitions, n_nodes), dtype=np.float64)
    single_prop_mask = np.zeros(n_transitions, dtype=np.uint8)

    for t_idx in prange(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop  = transitions[4, t_idx]
        n_p = p_stop - p_start
        if n_p == 1:
            single_prop_mask[t_idx] = 1
        first = True
        for p_idx in range(p_start, p_stop):
            summed = sums_all[p_idx, :]
            row_idx = proportion_info[2, p_idx]
            expnt_vec = param_t[row_idx, :]
            summed_exp = summed ** expnt_vec

            if first:
                source_numbers[t_idx, :] = summed
                safe_src = np.where(summed > 0.0, summed, 1.0)
                contrib = summed_exp / safe_src
                if n_p == 1:
                    param_idx = transitions[2, t_idx]
                    contrib *= param_t[param_idx, :]
                for n in range(n_nodes):
                    total_rates[t_idx, n] *= contrib[n]
                first = False
            else:
                for n in range(n_nodes):
                    total_rates[t_idx, n] *= summed_exp[n]

    return total_rates, source_numbers, single_prop_mask

# =============================================================================
# 2) Transition Rates (CSR mobility core + wrapper)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _compute_transition_rates_core(
    total_rates_base: npt.NDArray[np.float64],  # (Tn, N)
    transitions: npt.NDArray[np.int64],         # (5, Tn)
    param_vec_by_tr: npt.NDArray[np.float64],   # (Tn, N)
    percent_day_away: float,
    prop_who_move: npt.NDArray[np.float64],     # (N,)
    csr_data: npt.NDArray[np.float64],          # (nnz,)
    csr_indptr: npt.NDArray[np.int64],          # (N+1,)
    csr_indices: npt.NDArray[np.int64],         # (nnz,)
    population: npt.NDArray[np.float64],        # (N,)
    single_prop_mask: npt.NDArray[np.uint8],    # (Tn,)
) -> npt.NDArray[np.float64]:
    Tn, N = total_rates_base.shape
    out = np.empty_like(total_rates_base)

    inv_pop = np.empty(N, dtype=np.float64)
    keep = np.empty(N, dtype=np.float64)
    for n in range(N):
        pop = population[n]
        pop_safe = pop if pop > 0.0 else 1.0
        inv_pop[n] = 1.0 / pop_safe
        keep[n] = 1.0 - percent_day_away * prop_who_move[n]

    for t_idx in prange(Tn):
        if single_prop_mask[t_idx] == 1:
            for n in range(N):
                out[t_idx, n] = total_rates_base[t_idx, n]
            continue

        base_force = np.empty(N, dtype=np.float64)
        for n in range(N):
            base_force[n] = (total_rates_base[t_idx, n] * param_vec_by_tr[t_idx, n]) * inv_pop[n]

        for node in range(N):
            val = keep[node] * base_force[node]
            start = csr_indptr[node]
            end = csr_indptr[node + 1]
            acc = 0.0
            row_scale = percent_day_away * inv_pop[node]
            for k in range(start, end):
                v = csr_indices[k]
                acc += csr_data[k] * base_force[v]
            val += row_scale * acc
            out[t_idx, node] = val

    return out

def _compute_transition_rates(
    total_rates_base: npt.NDArray[np.float64],  # (Tn, N)
    source_numbers: npt.NDArray[np.float64],    # (Tn, N)
    transitions: npt.NDArray[np.int64],         # (5, Tn)
    param_t: npt.NDArray[np.float64],           # (P, N)
    percent_day_away: float,
    proportion_who_move: npt.NDArray[np.float64],  # (N,)
    mobility_data: npt.NDArray[np.float64],        # (nnz,)
    mobility_data_indices: npt.NDArray[np.int64],  # (N+1,)
    mobility_row_indices: npt.NDArray[np.int64],   # (nnz,)
    population: npt.NDArray[np.float64],           # (N,)
    single_prop_mask: npt.NDArray[np.uint8],       # (Tn,)
    param_expr_lookup: dict[int, str | np.ndarray] | None = None,
    param_name_to_row: dict[str, int] | None = None,
) -> npt.NDArray[np.float64]:
    Tn, N = total_rates_base.shape
    param_vec_by_tr = np.empty((Tn, N), dtype=np.float64)
    if param_expr_lookup is None:
        for t in range(Tn):
            pidx = int(transitions[2, t])
            param_vec_by_tr[t, :] = param_t[pidx, :]
    else:
        buf = np.empty(N, dtype=np.float64)
        for t in range(Tn):
            pidx = int(transitions[2, t])
            val = param_expr_lookup[pidx]
            if isinstance(val, np.ndarray):
                _resolve_param_expr_from_slice(val, param_t, out=buf)
            else:
                if param_name_to_row is None:
                    raise ValueError("param_name_to_row required when param_expr_lookup contains strings.")
                _resolve_param_expr_from_slice(val, param_t, param_name_to_row, out=buf)
            param_vec_by_tr[t, :] = buf

    indptr  = np.ascontiguousarray(mobility_data_indices, dtype=np.int64)
    indices = np.ascontiguousarray(mobility_row_indices, dtype=np.int64)
    data    = np.ascontiguousarray(mobility_data, dtype=np.float64)

    return _compute_transition_rates_core(
        np.ascontiguousarray(total_rates_base, dtype=np.float64),
        np.ascontiguousarray(transitions, dtype=np.int64),
        np.ascontiguousarray(param_vec_by_tr, dtype=np.float64),
        float(percent_day_away),
        np.ascontiguousarray(proportion_who_move, dtype=np.float64),
        data, indptr, indices,
        np.ascontiguousarray(population, dtype=np.float64),
        np.ascontiguousarray(single_prop_mask, dtype=np.uint8),
    )

# =============================================================================
# 3) Binomial Stochastic (NumPy) — optional, outside deterministic RHS
# =============================================================================

def _compute_transition_amounts_numpy_binomial(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    probs = 1.0 - np.exp(-dt * total_rates)
    probs = np.clip(probs, 0.0, 1.0)
    n = np.clip(source_numbers, 0, np.inf).astype(np.int64, copy=False)
    draws = np.random.binomial(n, probs)
    return draws.astype(np.float64, copy=False)

# =============================================================================
# 4) Transition Amounts (Deterministic dy/dt only) — hotspot with variants
# =============================================================================

# Non-fastmath variants
@njit(cache=True, fastmath=False)
def _compute_transition_amounts_serial_nm(source_numbers, total_rates):
    m, n = total_rates.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            out[i, j] = source_numbers[i, j] * total_rates[i, j]
    return out

@njit(parallel=True, cache=True, fastmath=False)
def _compute_transition_amounts_parallel_nm(source_numbers, total_rates):
    m, n = total_rates.shape
    size = m * n
    out = np.empty((m, n), dtype=np.float64)
    src = source_numbers.ravel()
    rate = total_rates.ravel()
    outf = out.ravel()
    for k in prange(size):
        outf[k] = src[k] * rate[k]
    return out

# Fastmath variants (your originals)
@njit(cache=True, fastmath=True)
def _compute_transition_amounts_serial_fm(source_numbers, total_rates):
    m, n = total_rates.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            out[i, j] = source_numbers[i, j] * total_rates[i, j]
    return out

@njit(parallel=True, cache=True, fastmath=True)
def _compute_transition_amounts_parallel_fm(source_numbers, total_rates):
    m, n = total_rates.shape
    size = m * n
    out = np.empty((m, n), dtype=np.float64)
    src = source_numbers.ravel()
    rate = total_rates.ravel()
    outf = out.ravel()
    for k in prange(size):
        outf[k] = src[k] * rate[k]
    return out

def compute_transition_amounts_meta(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Python dispatcher selecting kernel based on autotuned flags."""
    workload = source_numbers.shape[0] * source_numbers.shape[1]
    if _FASTMATH_ENABLED:
        serial_fn = _compute_transition_amounts_serial_fm
        par_fn    = _compute_transition_amounts_parallel_fm
    else:
        serial_fn = _compute_transition_amounts_serial_nm
        par_fn    = _compute_transition_amounts_parallel_nm

    if (not _FORCE_SERIAL) and (workload >= _PARALLEL_THRESHOLD):
        return par_fn(source_numbers, total_rates)
    else:
        return serial_fn(source_numbers, total_rates)

# =============================================================================
# 5) Assemble Flux Vector
# =============================================================================

@njit(cache=True, fastmath=True)
def _assemble_flux(
    amounts: npt.NDArray[np.float64],  # (Tn, N)
    transitions: npt.NDArray[np.int64],  # (5, Tn)
    ncompartments: int,
    nspatial_nodes: int,
) -> npt.NDArray[np.float64]:
    Tn = amounts.shape[0]
    N = nspatial_nodes
    dy_dt = np.zeros((ncompartments, N), dtype=np.float64)
    for t_idx in range(Tn):
        src = int(transitions[0, t_idx])
        dst = int(transitions[1, t_idx])
        src_row = dy_dt[src, :]
        dst_row = dy_dt[dst, :]
        a_row = amounts[t_idx, :]
        for n in range(N):
            a = a_row[n]
            src_row[n] -= a
            dst_row[n] += a
    return dy_dt.ravel()

# =============================================================================
# 6) Seeding
# =============================================================================

@njit(cache=True, fastmath=True)
def _apply_legacy_seeding_core(
    states_current: np.ndarray,  # (C, N), modified in place
    today: int,
    day_start_idx: np.ndarray,  # (D+1,)
    seeding_subpops: np.ndarray,
    seeding_sources: np.ndarray,
    seeding_dests: np.ndarray,
    seeding_amounts: np.ndarray,
    daily_incidence: np.ndarray,  # (D, C, N) or (0,0,0)
    update_incidence: int,
) -> None:
    if today < 0 or today + 1 >= day_start_idx.size:
        return
    start_idx = int(day_start_idx[today])
    stop_idx  = int(day_start_idx[today + 1])
    if stop_idx <= start_idx:
        return
    for e in range(start_idx, stop_idx):
        g = int(seeding_subpops[e])
        s = int(seeding_sources[e])
        d = int(seeding_dests[e])
        amt = seeding_amounts[e]
        states_current[s, g] -= amt
        if states_current[s, g] < 0.0:
            states_current[s, g] = 0.0
        states_current[d, g] += amt
        if update_incidence == 1:
            daily_incidence[today, d, g] += amt

def _apply_legacy_seeding(
    states_current: npt.NDArray[np.float64],
    today: int,
    seeding_data: dict[str, npt.NDArray[np.float64]],
    seeding_amounts: npt.NDArray[np.float64],
    daily_incidence: npt.NDArray[np.float64] | None = None,
) -> None:
    day_start_idx   = np.ascontiguousarray(seeding_data["day_start_idx"], dtype=np.int64)
    seeding_subpops = np.ascontiguousarray(seeding_data["seeding_subpops"], dtype=np.int64)
    seeding_sources = np.ascontiguousarray(seeding_data["seeding_sources"], dtype=np.int64)
    seeding_dests   = np.ascontiguousarray(seeding_data["seeding_destinations"], dtype=np.int64)
    amts = np.ascontiguousarray(seeding_amounts, dtype=states_current.dtype)
    if daily_incidence is None:
        di = np.empty((0, 0, 0), dtype=states_current.dtype); upd = 0
    else:
        di = daily_incidence; upd = 1
    _apply_legacy_seeding_core(states_current, int(today), day_start_idx,
                               seeding_subpops, seeding_sources, seeding_dests, amts, di, upd)

# =============================================================================
# 7) Factory class (solve_ivp-compatible) with auto-dispatch
#     Per-model tuning of selector-matrix path happens here.
# =============================================================================

def _has_seeding(precomputed: dict) -> bool:
    sd = precomputed.get("seeding_data", None)
    sa = precomputed.get("seeding_amounts", None)
    if sd is None or sa is None:
        return False
    try:
        dsi = np.asarray(sd["day_start_idx"])
    except Exception:
        return False
    if dsi.size < 1:
        return False
    has_events = bool(dsi[-1] > 0)
    return has_events and (np.asarray(sa).size > 0)

def _build_selector_dense(proportion_info: np.ndarray, transition_sum_compartments: np.ndarray, C: int) -> np.ndarray:
    Pk = proportion_info.shape[1]
    S = np.zeros((Pk, C), dtype=np.float64, order="F")  # Fortran order for GEMM
    for p_idx in range(Pk):
        sum_start = int(proportion_info[0, p_idx])
        sum_stop  = int(proportion_info[1, p_idx])
        if sum_stop > sum_start:
            comps = transition_sum_compartments[sum_start:sum_stop]
            for c in comps:
                S[p_idx, int(c)] = 1.0
    return S

def _build_selector_csr(proportion_info: np.ndarray, transition_sum_compartments: np.ndarray, C: int) -> csr_matrix:
    Pk = proportion_info.shape[1]
    indptr = np.zeros(Pk + 1, dtype=np.int64)
    indices_list = []
    data_list = []
    nnz = 0
    for p_idx in range(Pk):
        sum_start = int(proportion_info[0, p_idx])
        sum_stop  = int(proportion_info[1, p_idx])
        comps = transition_sum_compartments[sum_start:sum_stop]
        indices_list.extend([int(c) for c in comps])
        data_list.extend([1.0] * len(comps))
        nnz += len(comps)
        indptr[p_idx + 1] = nnz
    indices = np.array(indices_list, dtype=np.int64)
    data    = np.array(data_list, dtype=np.float64)
    return csr_matrix((data, indices, indptr), shape=(Pk, C))

class RHSfactory:
    REQUIRED_KEYS = [
        "ncompartments", "nspatial_nodes", "transitions",
        "proportion_info", "transition_sum_compartments",
        "percent_day_away", "proportion_who_move",
        "mobility_data", "mobility_data_indices", "mobility_row_indices",
        "population",
    ]
    OPTIONAL_KEYS = ["seeding_data", "seeding_amounts", "daily_incidence"]

    def __init__(
        self,
        precomputed: dict,
        param_expr_lookup: dict[int, str] | None = None,
        param_name_to_row: dict[str, int] | None = None,
        param_time_mode: str = "linear",
    ):
        for k in self.REQUIRED_KEYS:
            if k not in precomputed:
                raise KeyError(f"precomputed is missing required key: '{k}'")
        self.precomputed = precomputed
        self.param_expr_lookup = param_expr_lookup
        self.param_name_to_row = param_name_to_row
        self.param_time_mode = param_time_mode

        # Selector tuning storage
        self._selector_mode = "manual"       # 'manual' | 'dense' | 'sparse'
        self._selector_dense = None          # np.ndarray (Pk, C) if used
        self._selector_csr = None            # csr_matrix if used

        self._last_day_applied = {"day": None}
        self._rhs = None
        self._rhs_core = None

        self.build_rhs()

    def _reset_seeding_tracker(self) -> None:
        self._last_day_applied["day"] = None

    def _tune_selector_path(
        self, C: int, N: int, Pk: int, transitions: np.ndarray,
        proportion_info: np.ndarray, transition_sum_compartments: np.ndarray
    ) -> None:
        """Per-model autotune: choose manual vs dense GEMM vs sparse SpMM for sums."""
        if Pk == 0 or C == 0:
            self._selector_mode = "manual"
            return

        # Build selector matrices once
        S_dense = _build_selector_dense(proportion_info, transition_sum_compartments, C)
        S_csr   = _build_selector_csr(proportion_info, transition_sum_compartments, C)

        # Synthetic but shape-faithful inputs
        # P must cover both param rows from proportion_info and transitions[2,:]
        max_param_row = int(np.max(proportion_info[2, :])) if Pk > 0 else 0
        max_param_idx = int(np.max(transitions[2, :])) if transitions.size > 0 else 0
        P = max(max_param_row, max_param_idx) + 1
        rng = np.random.default_rng(0)
        states_current = rng.random((C, N), dtype=np.float64)
        param_t_slice  = rng.random((P, N), dtype=np.float64)

        # Warm JIT
        _compute_proportion_sums_exponents_manual(states_current, transitions, proportion_info,
                                                  transition_sum_compartments, param_t_slice)
        sums_probe = S_dense @ states_current
        _compute_proportion_sums_exponents_from_sums(sums_probe, transitions, proportion_info, param_t_slice)

        # Time MANUAL
        t_manual = _timeit(
            _compute_proportion_sums_exponents_manual,
            states_current, transitions, proportion_info, transition_sum_compartments, param_t_slice,
            repeats=3, warmup=1
        )

        # Time DENSE (GEMM + Numba combine)
        def _dense_call():
            sums = S_dense @ states_current
            return _compute_proportion_sums_exponents_from_sums(sums, transitions, proportion_info, param_t_slice)
        t_dense = _timeit(_dense_call, repeats=3, warmup=1)

        # Time SPARSE (SpMM + Numba combine)
        def _sparse_call():
            sums = S_csr @ states_current
            return _compute_proportion_sums_exponents_from_sums(sums, transitions, proportion_info, param_t_slice)
        t_sparse = _timeit(_sparse_call, repeats=3, warmup=1)

        # Pick best
        best = min([("manual", t_manual), ("dense", t_dense), ("sparse", t_sparse)], key=lambda x: x[1])[0]
        self._selector_mode = best
        if best == "dense":
            self._selector_dense = S_dense
            self._selector_csr = None
        elif best == "sparse":
            self._selector_csr = S_csr
            self._selector_dense = None
        else:
            self._selector_dense = None
            self._selector_csr = None

    def build_rhs(
        self,
        *,
        param_time_mode: str | None = None,
        seeding_data: dict[str, npt.NDArray[np.float64]] | None = None,
        seeding_amounts: npt.NDArray[np.float64] | None = None,
        daily_incidence: npt.NDArray[np.float64] | None = None,
    ) -> Callable[[float, npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        pc = self.precomputed
        C = int(pc["ncompartments"])
        N = int(pc["nspatial_nodes"])

        if param_time_mode is None:
            param_time_mode = self.param_time_mode
        if seeding_data is None:
            seeding_data = pc.get("seeding_data", None)
        if seeding_amounts is None:
            seeding_amounts = pc.get("seeding_amounts", None)
        if daily_incidence is None:
            daily_incidence = pc.get("daily_incidence", None)

        transitions  = np.ascontiguousarray(pc["transitions"], dtype=np.int64)
        proportion_info = np.ascontiguousarray(pc["proportion_info"], dtype=np.int64)
        transition_sum_compartments = np.ascontiguousarray(pc["transition_sum_compartments"], dtype=np.int64)
        percent_day_away = float(pc["percent_day_away"])
        proportion_who_move = np.ascontiguousarray(pc["proportion_who_move"], dtype=np.float64)
        mobility_data = np.ascontiguousarray(pc["mobility_data"], dtype=np.float64)
        mobility_data_indices = np.ascontiguousarray(pc["mobility_data_indices"], dtype=np.int64)
        mobility_row_indices  = np.ascontiguousarray(pc["mobility_row_indices"], dtype=np.int64)
        population = np.ascontiguousarray(pc["population"], dtype=np.float64)

        # Precompile parameter expressions (if provided) to row-index arrays once
        param_rows_lookup: dict[int, np.ndarray] | None = None
        if self.param_expr_lookup is not None:
            if self.param_name_to_row is None:
                raise ValueError("param_name_to_row required when param_expr_lookup is provided.")
            param_rows_lookup = {
                k: _compile_param_expr(v, self.param_name_to_row) for k, v in self.param_expr_lookup.items()
            }

        # Per-model autotune: choose selector path
        Pk = int(proportion_info.shape[1])
        self._tune_selector_path(C, N, Pk, transitions, proportion_info, transition_sum_compartments)

        # ---- core RHS used by both solve paths ----
        def _rhs_core(t: float, y: npt.NDArray[np.float64], param_t_slice: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            states_current = y.reshape((C, N))

            # 3) base proportion terms + source sizes
            if self._selector_mode == "dense":
                sums_all = self._selector_dense @ states_current  # (Pk, N)
                total_base, source_numbers, single_prop_mask = _compute_proportion_sums_exponents_from_sums(
                    sums_all, transitions, proportion_info, param_t_slice
                )
            elif self._selector_mode == "sparse":
                sums_all = self._selector_csr @ states_current    # (Pk, N)
                total_base, source_numbers, single_prop_mask = _compute_proportion_sums_exponents_from_sums(
                    sums_all, transitions, proportion_info, param_t_slice
                )
            else:
                total_base, source_numbers, single_prop_mask = _compute_proportion_sums_exponents_manual(
                    states_current, transitions, proportion_info, transition_sum_compartments, param_t_slice
                )

            # 4) parameter scaling + mobility mixing
            total_rates = _compute_transition_rates(
                total_rates_base=total_base,
                source_numbers=source_numbers,
                transitions=transitions,
                param_t=param_t_slice,
                percent_day_away=percent_day_away,
                proportion_who_move=proportion_who_move,
                mobility_data=mobility_data,
                mobility_data_indices=mobility_data_indices,
                mobility_row_indices=mobility_row_indices,
                population=population,
                single_prop_mask=single_prop_mask,
                param_expr_lookup=param_rows_lookup,
                param_name_to_row=None,
            )

            # 5) instantaneous flux and assembly: dy/dt
            amounts = compute_transition_amounts_meta(source_numbers, total_rates)
            dy = _assemble_flux(amounts, transitions, C, N)
            return dy

        def rhs(t: float, y: npt.NDArray[np.float64], parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            params_t, deltas = _prep_param_interpolator(parameters, use_deltas=(param_time_mode == "linear"))
            buf = np.empty((params_t.shape[1], params_t.shape[2]), dtype=np.float64)
            param_t_slice = _param_slice(params_t, t, mode=param_time_mode, out=buf, deltas=deltas)
            return _rhs_core(t, y, param_t_slice)

        self._rhs_core = _rhs_core
        self._rhs = rhs
        return rhs

    def create_rhs(self) -> Callable[[float, npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        if self._rhs is None:
            return self.build_rhs()
        return self._rhs

    def solve(
        self,
        y0: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],  # (P,T,N) or (P,T)
        t_span: tuple[float, float],
        t_eval: npt.NDArray[np.float64] | None = None,
        **solve_ivp_kwargs,
    ):
        self._reset_seeding_tracker()
        if self._rhs_core is None:
            self.build_rhs()
        rhs_core = self._rhs_core

        use_deltas = self.param_time_mode == "linear"
        params_t, deltas = _prep_param_interpolator(parameters, use_deltas=use_deltas)
        C = int(self.precomputed["ncompartments"])
        N = int(self.precomputed["nspatial_nodes"])
        buf = np.empty((params_t.shape[1], params_t.shape[2]), dtype=np.float64)

        def rhs_from_prepped(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            param_t_slice = _param_slice(params_t, t, mode=self.param_time_mode, out=buf, deltas=deltas)
            return rhs_core(t, y, param_t_slice)

        if not _has_seeding(self.precomputed):
            return solve_ivp(fun=rhs_from_prepped, t_span=t_span, y0=y0, t_eval=t_eval, **solve_ivp_kwargs)

        import types
        pc = self.precomputed
        seeding_data = pc.get("seeding_data", None)
        seeding_amounts = pc.get("seeding_amounts", None)
        daily_incidence = pc.get("daily_incidence", None)

        t0, tf = float(t_span[0]), float(t_span[1])
        eps = 1e-12
        acc_t: list[np.ndarray] = []
        acc_y: list[np.ndarray] = []
        total_success = True
        last_message = ""
        y_cur = np.array(y0, dtype=np.float64, copy=True)
        t_cur = t0

        def _slice_t_eval(t_start: float, t_end: float, first_segment: bool):
            if t_eval is None:
                return None
            if first_segment:
                mask = (t_eval >= t_start - eps) & (t_eval <= t_end + eps)
            else:
                mask = (t_eval > t_start + eps) & (t_eval <= t_end + eps)
            seg = t_eval[mask]
            return seg if seg.size > 0 else None

        first = True
        while t_cur < tf - eps:
            next_boundary = math.floor(t_cur + eps) + 1.0
            seg_end = min(tf, next_boundary)
            if seg_end <= t_cur + eps:
                t_cur = seg_end + eps
                first = False
                continue

            day_start = int(round(t_cur))
            if abs(t_cur - float(day_start)) < 1e-12:
                states_current = y_cur.reshape((C, N))
                _apply_legacy_seeding(states_current, day_start, seeding_data, seeding_amounts, daily_incidence=daily_incidence)
                y_cur = states_current.ravel()

            t_eval_seg = _slice_t_eval(t_cur, seg_end, first_segment=first)
            res = solve_ivp(fun=rhs_from_prepped, t_span=(t_cur, seg_end), y0=y_cur, t_eval=t_eval_seg, **solve_ivp_kwargs)
            total_success = total_success and bool(res.success)
            last_message = res.message
            t_arr = np.asarray(res.t)
            y_arr = np.asarray(res.y)
            if t_eval_seg is not None and t_arr.size > 0:
                acc_t.append(t_arr)
                acc_y.append(y_arr)
            if y_arr.size > 0:
                y_cur = y_arr[:, -1]
            t_cur = seg_end
            first = False

        if t_eval is not None:
            if acc_t:
                T = np.concatenate(acc_t)
                Y = np.concatenate(acc_y, axis=1)
            else:
                T = np.array([], dtype=float)
                Y = np.empty((y0.size, 0), dtype=float)
            return types.SimpleNamespace(t=T, y=Y, success=bool(total_success), message=last_message)

        return types.SimpleNamespace(t=np.array([tf]), y=y_cur.reshape(-1, 1), success=bool(total_success), message=last_message)

# =============================================================================
# 8) Global AUTOTUNE: threads, fastmath on/off, threshold crossover
# =============================================================================

def autotune_all(
    candidate_threads: Sequence[int] | None = None,
    size_grid: Sequence[int] | None = None,
    dtype=np.float64,
    rng_seed: int = 42,
    fm_tol: float = 1e-11,
    quiet: bool = False,
) -> dict:
    """
    Tunes:
      - NUMBA_NUM_THREADS
      - fastmath on/off for the elementwise hotspot (w/ accuracy guard)
      - parallel threshold (serial↔parallel crossover)
    """
    import numba
    from numba import set_num_threads, get_num_threads, threading_layer

    # Candidates
    hw_threads = os.cpu_count() or 8
    if candidate_threads is None:
        candidate_threads = sorted(set([1, 2, 4, 6, 8, 12, 16, 24, 32, hw_threads]))
        candidate_threads = [t for t in candidate_threads if t <= hw_threads]
    if size_grid is None:
        size_grid = [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000]

    # Synthetic inputs for hotspot
    rng = np.random.default_rng(rng_seed)
    m_ref, n_ref = _make_rect_from_size(1_000_000)
    ref_src  = rng.random((m_ref, n_ref), dtype=dtype)
    ref_rate = rng.random((m_ref, n_ref), dtype=dtype)

    # Warm: JIT all four kernels
    _ = _compute_transition_amounts_serial_nm(ref_src, ref_rate)
    _ = _compute_transition_amounts_parallel_nm(ref_src, ref_rate)
    _ = _compute_transition_amounts_serial_fm(ref_src, ref_rate)
    _ = _compute_transition_amounts_parallel_fm(ref_src, ref_rate)

    # -------- fastmath selection (accuracy first, then speed) --------
    y_ref = _compute_transition_amounts_serial_nm(ref_src, ref_rate)
    y_fm  = _compute_transition_amounts_serial_fm(ref_src, ref_rate)
    err_fm = _max_rel_err(y_ref, y_fm)
    t_ser_nm = _timeit(_compute_transition_amounts_serial_nm, ref_src, ref_rate, repeats=6, warmup=1)
    t_ser_fm = _timeit(_compute_transition_amounts_serial_fm, ref_src, ref_rate, repeats=6, warmup=1)
    use_fastmath = (err_fm <= fm_tol) and (t_ser_fm <= t_ser_nm)
    set_fastmath_enabled(use_fastmath)
    if not quiet:
        print(f"[autotune] fastmath err={err_fm:.3e} tol={fm_tol:.1e} -> {'use' if use_fastmath else 'skip'} "
              f"(t_ser_nm={t_ser_nm:.6f}s, t_ser_fm={t_ser_fm:.6f}s)")

    # -------- choose best thread count on a big workload (parallel kernel) ----
    big_size = max(size_grid)
    m_big, n_big = _make_rect_from_size(big_size)
    src_big  = rng.random((m_big, n_big), dtype=dtype)
    rate_big = rng.random((m_big, n_big), dtype=dtype)
    par_fn = _compute_transition_amounts_parallel_fm if _FASTMATH_ENABLED else _compute_transition_amounts_parallel_nm

    best_threads, best_t = candidate_threads[0], float("inf")
    for t in candidate_threads:
        set_num_threads(t)
        dt = _timeit(par_fn, src_big, rate_big, repeats=4, warmup=1)
        if not quiet:
            print(f"[autotune] threads={t} (actual={get_num_threads()}) -> {dt:.6f}s")
        if dt < best_t:
            best_t = dt
            best_threads = t
    set_num_threads(best_threads)
    if not quiet:
        print(f"[autotune] Selected NUMBA_NUM_THREADS={best_threads} (layer={threading_layer()})")

    # -------- serial↔parallel crossover threshold -----------------------------
    serial_fn = _compute_transition_amounts_serial_fm if _FASTMATH_ENABLED else _compute_transition_amounts_serial_nm
    cross_at = None
    for total in size_grid:
        m, n = _make_rect_from_size(total)
        src = rng.random((m, n), dtype=dtype)
        rate = rng.random((m, n), dtype=dtype)
        t_ser = _timeit(serial_fn, src, rate, repeats=6, warmup=1)
        t_par = _timeit(par_fn,   src, rate, repeats=6, warmup=1)
        if not quiet:
            print(f"[autotune] size={m}x{n} ({m*n:,}): serial={t_ser:.6f}s  parallel={t_par:.6f}s")
        if t_par < t_ser:
            cross_at = m * n
            break

    forced_serial = False
    if cross_at is None:
        threshold = int(max(size_grid) * 10)
        forced_serial = True
        set_parallel_threshold(threshold)
        set_force_serial(True)
        if not quiet:
            print(f"[autotune] Parallel never faster; forcing serial. Threshold={threshold:,}")
    else:
        threshold = int(cross_at * 1.5)  # safety margin
        set_parallel_threshold(threshold)
        set_force_serial(False)
        if not quiet:
            print(f"[autotune] _PARALLEL_THRESHOLD set to {threshold:,} elements")

    return {
        "threads": best_threads,
        "threshold": threshold,
        "forced_serial": forced_serial,
        "fastmath": _FASTMATH_ENABLED,
    }

