# src/gempyor/vectorization_experiments.py
from __future__ import annotations

import os
import math
import time
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from numba import njit, prange
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

# Type aliases
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
U8Array = NDArray[np.uint8]

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
    """Set the elementwise workload threshold for switching to the parallel kernel.

    Args:
        n: Total element count (M*N) where parallelization becomes beneficial.
    """
    global _PARALLEL_THRESHOLD
    _PARALLEL_THRESHOLD = int(n)


def set_force_serial(flag: bool) -> None:
    """Force the serial kernel regardless of workload."""
    global _FORCE_SERIAL
    _FORCE_SERIAL = bool(flag)


def set_fastmath_enabled(flag: bool) -> None:
    """Enable/disable Numba fastmath for the elementwise hotspot."""
    global _FASTMATH_ENABLED
    _FASTMATH_ENABLED = bool(flag)


def get_autotune_config() -> dict:
    """Return the current autotune configuration flags."""
    return {
        "threshold": _PARALLEL_THRESHOLD,
        "force_serial": _FORCE_SERIAL,
        "fastmath": _FASTMATH_ENABLED,
    }

# =============================================================================
# Utilities (timing, helpers)
# =============================================================================

def _timeit(fn: Callable, *args, repeats: int = 5, warmup: int = 1) -> float:
    """Time a function call with warmup and averaging.

    Args:
        fn: Callable to time.
        *args: Positional args forwarded to `fn`.
        repeats: Number of measurements to average.
        warmup: Number of warmup runs not included in timing.

    Returns:
        Average elapsed time in seconds.
    """
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def _make_rect_from_size(total_elems: int) -> tuple[int, int]:
    """Heuristic M×N rectangle for a given total element count."""
    m = max(64, int(math.sqrt(total_elems)))
    n = max(64, total_elems // m)
    return m, n


def _max_rel_err(a: FloatArray, b: FloatArray, eps: float = 1e-300) -> float:
    """Maximum relative error between arrays, safe near zero."""
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom < eps, 1.0, denom)
    return float(np.max(np.abs(a - b) / denom))

# =============================================================================
# Setup helpers (one-time)
# =============================================================================

def _prep_param_interpolator(parameters: FloatArray, use_deltas: bool = False) -> tuple[FloatArray, FloatArray | None]:
    """Prepare parameter tensor for fast slicing at time t.

    Args:
        parameters: Parameter tensor (P,T,N) or (P,T). If (P,T), a trailing singleton node axis is added.
        use_deltas: If True and linear mode is used, precompute (T-1,P,N) deltas for axpy blending.

    Returns:
        params_t: Array of shape (T,P,N) in C-order.
        deltas: Optional (T-1,P,N) deltas (or None).
    """
    if parameters.ndim == 2:
        parameters = parameters[:, :, None]  # (P,T) -> (P,T,1)
    params_t = np.moveaxis(parameters, 1, 0).copy(order="C")  # (T,P,N)
    deltas: FloatArray | None = None
    if use_deltas and params_t.shape[0] >= 2:
        deltas = np.ascontiguousarray(params_t[1:] - params_t[:-1])  # (T-1,P,N)
    return params_t, deltas


@njit(cache=True, fastmath=True)
def _blend_linear(A: FloatArray, B: FloatArray, alpha: float, out: FloatArray) -> None:
    """Compute out = A + alpha*(B-A) elementwise."""
    P, N = out.shape
    for p in range(P):
        for n in range(N):
            out[p, n] = A[p, n] + alpha * (B[p, n] - A[p, n])


@njit(cache=True, fastmath=True)
def _axpy_linear(A: FloatArray, D: FloatArray, alpha: float, out: FloatArray) -> None:
    """Compute out = A + alpha*D elementwise (D is precomputed B-A)."""
    P, N = out.shape
    for p in range(P):
        for n in range(N):
            out[p, n] = A[p, n] + alpha * D[p, n]


def _param_slice(
    params_t: FloatArray,
    t: float,
    mode: str = "step",
    out: FloatArray | None = None,
    deltas: FloatArray | None = None,
) -> FloatArray:
    """Slice parameters at (possibly fractional) time t.

    Args:
        params_t: Parameters laid out as (T,P,N) in C-order.
        t: Time in days (may be fractional).
        mode: "step" (nearest-left day) or "linear" interpolation.
        out: Optional output buffer (P,N).
        deltas: Optional (T-1,P,N) deltas for axpy when mode="linear".

    Returns:
        param_t at time t, shape (P,N).
    """
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

def _compile_param_expr(expr: str, param_name_to_row: dict[str, int]) -> IntArray:
    """Compile a '*' product expression into a vector of row indices.

    Example:
        expr="a*b*c" -> rows=[row(a), row(b), row(c)]
    """
    terms = [s.strip() for s in expr.split("*") if s.strip()]
    if not terms:
        raise ValueError("Empty parameter expression.")
    try:
        rows = np.fromiter((param_name_to_row[t] for t in terms), dtype=np.int64)
    except KeyError as e:
        raise KeyError(f"Unknown parameter in expression: {e}") from None
    return rows


@njit(cache=True, fastmath=True)
def _product_rows_into(param_t: FloatArray, rows: IntArray, out: FloatArray) -> None:
    """Multiply parameter rows (per node): out[n] = Π_k param_t[rows[k], n]."""
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
    expr: str | IntArray,
    param_t: FloatArray,
    param_name_to_row: dict[str, int] | None = None,
    out: FloatArray | None = None,
) -> FloatArray:
    """Resolve an expression or row-list into a per-node vector from a (P,N) time slice."""
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


def _safe_param_expr_lookup(unique_strings: Sequence[str]) -> tuple[dict[int, str] | None, dict[str, int]]:
    """Build a safe param-expression lookup for any '*' rows in a parameter name list."""
    name_to_row = {name: i for i, name in enumerate(unique_strings)}
    expr_lookup: dict[int, str] = {}
    for idx, s in enumerate(unique_strings):
        if "*" in s:
            terms = [t.strip() for t in s.split("*")]
            if all(term in name_to_row for term in terms):
                expr_lookup[idx] = s
    return (expr_lookup if expr_lookup else None, name_to_row)

# =============================================================================
# 1) Core Proportion Logic
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _compute_proportion_sums_exponents_manual(
    states_current: FloatArray,                    # (C, N)
    transitions: IntArray,                         # (5, Tn)
    proportion_info: IntArray,                     # (3, Pk)
    transition_sum_compartments: IntArray,         # (S,)
    param_t: FloatArray,                           # (P, N)
) -> tuple[FloatArray, FloatArray, U8Array]:
    """Compute total base rates and sources with subset sums performed on the fly."""
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
    sums_all: FloatArray,                          # (Pk, N) precomputed sums
    transitions: IntArray,                         # (5, Tn)
    proportion_info: IntArray,                     # (3, Pk)
    param_t: FloatArray,                           # (P, N)
) -> tuple[FloatArray, FloatArray, U8Array]:
    """Same as above, but proportion subset sums are precomputed via S @ states."""
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
    total_rates_base: FloatArray,  # (Tn, N)
    transitions: IntArray,         # (5, Tn)
    param_vec_by_tr: FloatArray,   # (Tn, N)
    percent_day_away: float,
    prop_who_move: FloatArray,     # (N,)
    csr_data: FloatArray,          # (nnz,)
    csr_indptr: IntArray,          # (N+1,)
    csr_indices: IntArray,         # (nnz,)
    population: FloatArray,        # (N,)
    single_prop_mask: U8Array,     # (Tn,)
) -> FloatArray:
    """Mix per-node forces across mobility graph; handle single-proportion fast path."""
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
    total_rates_base: FloatArray,  # (Tn, N)
    source_numbers: FloatArray,    # (Tn, N)  (not used here but kept for API symmetry/future)
    transitions: IntArray,         # (5, Tn)
    param_t: FloatArray,           # (P, N)
    percent_day_away: float,
    proportion_who_move: FloatArray,      # (N,)
    mobility_data: FloatArray,            # (nnz,)
    mobility_data_indices: IntArray,      # (N+1,)
    mobility_row_indices: IntArray,       # (nnz,)
    population: FloatArray,               # (N,)
    single_prop_mask: U8Array,            # (Tn,)
    param_expr_lookup: dict[int, str | IntArray] | None = None,
    param_name_to_row: dict[str, int] | None = None,
) -> FloatArray:
    """Resolve per-transition parameter vectors and mix across mobility."""
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
    source_numbers: FloatArray,
    total_rates: FloatArray,
    dt: float,
) -> FloatArray:
    """Sample binomial draws for transition amounts (used outside the deterministic RHS)."""
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
def _compute_transition_amounts_serial_nm(source_numbers: FloatArray, total_rates: FloatArray) -> FloatArray:
    """Elementwise multiply (serial, no fastmath)."""
    m, n = total_rates.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            out[i, j] = source_numbers[i, j] * total_rates[i, j]
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _compute_transition_amounts_parallel_nm(source_numbers: FloatArray, total_rates: FloatArray) -> FloatArray:
    """Elementwise multiply (parallel, no fastmath)."""
    m, n = total_rates.shape
    size = m * n
    out = np.empty((m, n), dtype=np.float64)
    src = source_numbers.ravel()
    rate = total_rates.ravel()
    outf = out.ravel()
    for k in prange(size):
        outf[k] = src[k] * rate[k]
    return out

# Fastmath variants
@njit(cache=True, fastmath=True)
def _compute_transition_amounts_serial_fm(source_numbers: FloatArray, total_rates: FloatArray) -> FloatArray:
    """Elementwise multiply (serial, fastmath)."""
    m, n = total_rates.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            out[i, j] = source_numbers[i, j] * total_rates[i, j]
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _compute_transition_amounts_parallel_fm(source_numbers: FloatArray, total_rates: FloatArray) -> FloatArray:
    """Elementwise multiply (parallel, fastmath)."""
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
    source_numbers: FloatArray,
    total_rates: FloatArray,
) -> FloatArray:
    """Dispatch to the best elementwise multiply kernel given autotune flags.

    Args:
        source_numbers: (Tn,N) sources.
        total_rates: (Tn,N) rates.

    Returns:
        (Tn,N) amounts.
    """
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
    amounts: FloatArray,         # (Tn, N)
    transitions: IntArray,       # (5, Tn)
    ncompartments: int,
    nspatial_nodes: int,
) -> FloatArray:
    """Assemble dy/dt from transition amounts and transition edges."""
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
    states_current: FloatArray,   # (C, N), modified in place
    today: int,
    day_start_idx: IntArray,      # (D+1,)
    seeding_subpops: IntArray,
    seeding_sources: IntArray,
    seeding_dests: IntArray,
    seeding_amounts: FloatArray,
    daily_incidence: FloatArray,  # (D, C, N) or (0,0,0)
    update_incidence: int,
) -> None:
    """Apply discrete seeding events for a given day index."""
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
    states_current: FloatArray,
    today: int,
    seeding_data: dict[str, FloatArray],
    seeding_amounts: FloatArray,
    daily_incidence: FloatArray | None = None,
) -> None:
    """Python wrapper to call the njit seeding kernel with properly typed arrays."""
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
# 7) Accumulator kernels (JIT, streaming, no big temporaries)
# =============================================================================

@njit(parallel=True, cache=True, fastmath=True)
def _accumulate_fluxes_for_groups_core(
    amounts: FloatArray,                # (Tn, N)
    param_t_slice: FloatArray,          # (P, N)
    acc_tr_indices: IntArray,           # (K,)
    acc_tr_starts: IntArray,            # (A+1,)
    prob_rows_flat: IntArray,           # (K,)  row idx or -1
    prob_const_flat: FloatArray,        # (K,)
    out: FloatArray                     # (A, N), preallocated, overwritten
) -> None:
    """Compute per-accumulator incident flux vectors out[a, n] = Σ_t amounts[t,n]*const*prob(row,n)."""
    A = acc_tr_starts.shape[0] - 1
    N = amounts.shape[1]
    for a in prange(A):
        s0 = acc_tr_starts[a]
        s1 = acc_tr_starts[a + 1]
        if s1 <= s0:
            for n in range(N):
                out[a, n] = 0.0
            continue
        for n in range(N):
            acc_val = 0.0
            for k in range(s0, s1):
                t_idx  = acc_tr_indices[k]
                prow   = prob_rows_flat[k]
                pconst = prob_const_flat[k]
                pval   = 1.0 if prow < 0 else param_t_slice[prow, n]
                acc_val += amounts[t_idx, n] * (pconst * pval)
            out[a, n] = acc_val


@njit(parallel=True, cache=True, fastmath=True)
def _apply_accumulators_core(
    dy: FloatArray,                 # (C, N) in-place
    states_current: FloatArray,     # (C, N)
    acc_flux: FloatArray,           # (A, N)
    offsets: IntArray,              # (A,)
    nstages: IntArray,              # (A,)
    cum_offsets: IntArray,          # (A,)
    delay_row: IntArray,            # (A,)    row idx or -1
    delay_const: FloatArray,        # (A,)
    param_t_slice: FloatArray       # (P, N)
) -> None:
    """Apply accumulator influx to chains (instant sink or Erlang k-stage with per-node rate)."""
    A = offsets.shape[0]
    N = acc_flux.shape[1]
    for a in prange(A):
        off = int(offsets[a])
        k   = int(nstages[a])
        row = int(delay_row[a])

        if row >= 0:
            for n in range(N):
                mean = param_t_slice[row, n]
                if k <= 1 and mean <= 1e-12:
                    # pure cumulative sink into first accumulator cell
                    dy[off, n] += acc_flux[a, n]
                else:
                    mu = (k / mean) if (k > 0 and mean > 1e-12) else 0.0
                    rate = mu * max(1, k)
                    # stage 0
                    dy[off + 0, n] += acc_flux[a, n] - rate * states_current[off + 0, n]
                    # intermediate stages
                    for s in range(1, k):
                        dy[off + s, n] += rate * states_current[off + s - 1, n] - rate * states_current[off + s, n]
                    # optional cumulative outflow sink
                    c_off = int(cum_offsets[a])
                    if c_off >= 0 and k > 0:
                        dy[c_off, n] += rate * states_current[off + k - 1, n]
        else:
            mean_c = delay_const[a]
            if k <= 1 and mean_c <= 0.0:
                for n in range(N):
                    dy[off, n] += acc_flux[a, n]
            else:
                rate = (k / mean_c) if (k > 0 and mean_c > 0.0) else 0.0
                for n in range(N):
                    dy[off + 0, n] += acc_flux[a, n] - rate * states_current[off + 0, n]
                    for s in range(1, k):
                        dy[off + s, n] += rate * states_current[off + s - 1, n] - rate * states_current[off + s, n]
                    c_off = int(cum_offsets[a])
                    if c_off >= 0 and k > 0:
                        dy[c_off, n] += rate * states_current[off + k - 1, n]

# =============================================================================
# 8) Factory class (solve_ivp-compatible) with auto-dispatch + accumulators
# =============================================================================

def _has_seeding(precomputed: dict) -> bool:
    """Return True if precomputed includes one or more seeding events."""
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


def _build_selector_dense(proportion_info: IntArray, transition_sum_compartments: IntArray, C: int) -> FloatArray:
    """Build a dense selector matrix (Pk,C) where row p selects compartments to sum."""
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


def _build_selector_csr(proportion_info: IntArray, transition_sum_compartments: IntArray, C: int) -> csr_matrix:
    """Build a CSR selector matrix (Pk,C) like above."""
    Pk = proportion_info.shape[1]
    indptr = np.zeros(Pk + 1, dtype=np.int64)
    indices_list: list[int] = []
    data_list: list[float] = []
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
    """Create a solve_ivp-compatible RHS callable with model-specific tuning.

    Required `precomputed` keys:
        - "ncompartments": int
        - "nspatial_nodes": int
        - "transitions": int64 array (5,Tn)
        - "proportion_info": int64 array (3,Pk)
        - "transition_sum_compartments": int64 array (S,)
        - "percent_day_away": float
        - "proportion_who_move": float64 array (N,)
        - "mobility_data": float64 array (nnz,)
        - "mobility_data_indices": int64 array (N+1,)
        - "mobility_row_indices": int64 array (nnz,)
        - "population": float64 array (N,)

    Optional `precomputed` keys:
        - "seeding_data": dict with int64 arrays like {"day_start_idx", "seeding_*"}
        - "seeding_amounts": float64 array
        - "daily_incidence": float64 array (D,C,N) used by legacy seeding
        - "accumulators": dict describing optional accumulator compartments:

            accumulators = {
                "n_acc": int A,
                "offsets": int64 array (A,),          # first compartment row for each accumulator
                "nstages": int64 array (A,),          # 1 => instant cumulative sink; k>=1 => k-stage Erlang chain
                "cum_offsets": int64 array (A,),      # cumulative sink row per acc, or -1 to skip
                # Flattened transition groups:
                "acc_tr_indices": int64 array (K,),   # concatenated transition indices
                "acc_tr_starts": int64 array (A+1,),  # boundaries per accumulator into acc_tr_indices
                # Per (acc,transition) probability spec, flattened to length K:
                "acc_prob_param_row_flat": int64 array (K,),   # param row for prob, or -1 for constant-only
                "acc_prob_const_flat": float64 array (K,),     # constant multiplier per (acc, tr)
                # Delay mean (days) per accumulator:
                "delay_mean_param_row": int64 array (A,),      # param row index for mean days, or -1
                "delay_mean_const": float64 array (A,),         # constant mean days if no param row
            }
    """

    REQUIRED_KEYS = [
        "ncompartments", "nspatial_nodes", "transitions",
        "proportion_info", "transition_sum_compartments",
        "percent_day_away", "proportion_who_move",
        "mobility_data", "mobility_data_indices", "mobility_row_indices",
        "population",
    ]
    OPTIONAL_KEYS = ["seeding_data", "seeding_amounts", "daily_incidence", "accumulators"]

    def __init__(
        self,
        precomputed: dict,
        param_expr_lookup: dict[int, str] | None = None,
        param_name_to_row: dict[str, int] | None = None,
        param_time_mode: str = "step",
    ):
        """Initialize the factory and pre-tune selector path."""
        for k in self.REQUIRED_KEYS:
            if k not in precomputed:
                raise KeyError(f"precomputed is missing required key: '{k}'")
        self.precomputed = precomputed
        self._param_expr_lookup = param_expr_lookup
        self.param_name_to_row = param_name_to_row
        self.param_time_mode = param_time_mode

        # Selector tuning storage
        self._selector_mode = "manual"       # 'manual' | 'dense' | 'sparse'
        self._selector_dense: FloatArray | None = None
        self._selector_csr: csr_matrix | None = None

        # Accumulator spec and buffers
        self._acc: dict | None = None
        self._acc_flux_buf: FloatArray | None = None  # (A, N) reused buffer

        self._last_day_applied = {"day": None}
        self._rhs: Callable | None = None
        self._rhs_core: Callable | None = None

        self.build_rhs()

    def _reset_seeding_tracker(self) -> None:
        """Reset internal seeding state between solves."""
        self._last_day_applied["day"] = None

    def _tune_selector_path(
        self, C: int, N: int, Pk: int, transitions: IntArray,
        proportion_info: IntArray, transition_sum_compartments: IntArray
    ) -> None:
        """Per-model autotune: choose manual vs dense GEMM vs sparse SpMM for sums."""
        if Pk == 0 or C == 0:
            self._selector_mode = "manual"
            return

        # Build selector matrices once
        S_dense = _build_selector_dense(proportion_info, transition_sum_compartments, C)
        S_csr   = _build_selector_csr(proportion_info, transition_sum_compartments, C)

        # Synthetic but shape-faithful inputs
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

    # ---------- Accumulator utilities ----------

    @staticmethod
    def _parse_accumulators(pc: dict) -> dict | None:
        """Parse and validate the optional accumulators spec from precomputed."""
        acc = pc.get("accumulators", None)
        if not acc:
            return None
        required = [
            "n_acc", "offsets", "nstages", "acc_tr_indices", "acc_tr_starts",
            "acc_prob_param_row_flat", "acc_prob_const_flat",
            "delay_mean_param_row", "delay_mean_const", "cum_offsets"
        ]
        for k in required:
            if k not in acc:
                raise KeyError(f"accumulators missing key '{k}'")
        A = int(acc["n_acc"])
        offsets = np.ascontiguousarray(np.asarray(acc["offsets"], dtype=np.int64))
        nstages = np.ascontiguousarray(np.asarray(acc["nstages"], dtype=np.int64))
        cum_offsets = np.ascontiguousarray(np.asarray(acc["cum_offsets"], dtype=np.int64))

        acc_tr_indices = np.ascontiguousarray(np.asarray(acc["acc_tr_indices"], dtype=np.int64))
        acc_tr_starts  = np.ascontiguousarray(np.asarray(acc["acc_tr_starts"], dtype=np.int64))
        if acc_tr_starts.shape[0] != A + 1:
            raise ValueError("acc_tr_starts must have length A+1")

        prob_rows_flat  = np.ascontiguousarray(np.asarray(acc["acc_prob_param_row_flat"], dtype=np.int64))
        prob_const_flat = np.ascontiguousarray(np.asarray(acc["acc_prob_const_flat"], dtype=np.float64))
        if prob_rows_flat.shape[0] != acc_tr_indices.shape[0] or prob_const_flat.shape[0] != acc_tr_indices.shape[0]:
            raise ValueError("acc_prob_*_flat must align with acc_tr_indices length")

        delay_row = np.ascontiguousarray(np.asarray(acc["delay_mean_param_row"], dtype=np.int64))
        delay_const = np.ascontiguousarray(np.asarray(acc["delay_mean_const"], dtype=np.float64))
        if delay_row.shape[0] != A or delay_const.shape[0] != A:
            raise ValueError("delay arrays must be length A")

        return {
            "A": A,
            "offsets": offsets,
            "nstages": nstages,
            "cum_offsets": cum_offsets,
            "acc_tr_indices": acc_tr_indices,
            "acc_tr_starts": acc_tr_starts,
            "prob_rows_flat": prob_rows_flat,
            "prob_const_flat": prob_const_flat,
            "delay_row": delay_row,
            "delay_const": delay_const,
        }

    def build_rhs(
        self,
        *,
        param_time_mode: str | None = None,
        seeding_data: dict[str, FloatArray] | None = None,
        seeding_amounts: FloatArray | None = None,
        daily_incidence: FloatArray | None = None,
    ) -> Callable[[float, FloatArray, FloatArray], FloatArray]:
        """Construct the RHS function and cache tuned paths."""
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
        param_rows_lookup: dict[int, IntArray] | None = None
        if self._param_expr_lookup is not None:
            if self.param_name_to_row is None:
                raise ValueError("param_name_to_row required when param_expr_lookup is provided.")
            param_rows_lookup = {
                k: _compile_param_expr(v, self.param_name_to_row) for k, v in self._param_expr_lookup.items()
            }

        # Per-model autotune: choose selector path
        Pk = int(proportion_info.shape[1])
        self._tune_selector_path(C, N, Pk, transitions, proportion_info, transition_sum_compartments)

        # Accumulators (optional)
        self._acc = self._parse_accumulators(pc)
        self._acc_flux_buf = None  # reset; will lazily (re)allocate sized to (A,N)

        # ---- core RHS used by both solve paths ----
        def _rhs_core(t: float, y: FloatArray, param_t_slice: FloatArray) -> FloatArray:
            """Compute dy/dt at time t for flattened state y."""
            states_current = y.reshape((C, N))

            # 1) base proportion terms + source sizes
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

            # 2) parameter scaling + mobility mixing
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

            # 3) instantaneous flux and assembly: dy/dt for base compartments
            amounts = compute_transition_amounts_meta(source_numbers, total_rates)  # (Tn,N)
            dy = _assemble_flux(amounts, transitions, C, N).reshape((C, N))

            # 4) optional accumulators (instantaneous or Erlang delay chains)
            if self._acc is not None:
                acc = self._acc
                A = int(acc["A"])

                # Reuse a single (A, N) buffer across RHS calls
                if (self._acc_flux_buf is None) or (self._acc_flux_buf.shape[0] != A) or (self._acc_flux_buf.shape[1] != N):
                    self._acc_flux_buf = np.zeros((A, N), dtype=np.float64, order="C")
                acc_flux = self._acc_flux_buf  # (A, N)

                # 4a) incident fluxes per accumulator (A,N) — JIT streaming
                _accumulate_fluxes_for_groups_core(
                    amounts, param_t_slice,
                    acc["acc_tr_indices"], acc["acc_tr_starts"],
                    acc["prob_rows_flat"], acc["prob_const_flat"],
                    acc_flux
                )

                # 4b) inject into accumulator compartments — JIT chain/sink
                _apply_accumulators_core(
                    dy, states_current, acc_flux,
                    acc["offsets"], acc["nstages"], acc["cum_offsets"],
                    acc["delay_row"], acc["delay_const"],
                    param_t_slice
                )

            return dy.ravel()

        def rhs(t: float, y: FloatArray, parameters: FloatArray) -> FloatArray:
            """solve_ivp-compatible RHS wrapper resolving param slice at time t."""
            params_t, deltas = _prep_param_interpolator(parameters, use_deltas=(param_time_mode == "step"))
            buf = np.empty((params_t.shape[1], params_t.shape[2]), dtype=np.float64)
            param_t_slice = _param_slice(params_t, t, mode=param_time_mode, out=buf, deltas=deltas)
            return _rhs_core(t, y, param_t_slice)

        self._rhs_core = _rhs_core
        self._rhs = rhs
        return rhs

    def create_rhs(self) -> Callable[[float, FloatArray, FloatArray], FloatArray]:
        """Return the cached RHS (build if needed)."""
        if self._rhs is None:
            return self.build_rhs()
        return self._rhs

    def solve(
        self,
        y0: FloatArray,
        parameters: FloatArray,               # (P,T,N) or (P,T)
        t_span: tuple[float, float],
        t_eval: FloatArray | None = None,
        **solve_ivp_kwargs,
    ):
        """Integrate the system, applying daily seeding boundaries if configured."""
        self._reset_seeding_tracker()
        if self._rhs_core is None:
            self.build_rhs()
        rhs_core = self._rhs_core  # type: ignore[assignment]

        use_deltas = self.param_time_mode == "step"
        params_t, deltas = _prep_param_interpolator(parameters, use_deltas=use_deltas)
        C = int(self.precomputed["ncompartments"])
        N = int(self.precomputed["nspatial_nodes"])
        buf = np.empty((params_t.shape[1], params_t.shape[2]), dtype=np.float64)

        # Ensure y0 is flat (C*N,)
        y0_flat = y0.ravel() if y0.ndim != 1 else y0

        def rhs_from_prepped(t: float, y: FloatArray) -> FloatArray:
            param_t_slice = _param_slice(params_t, t, mode=self.param_time_mode, out=buf, deltas=deltas)
            return rhs_core(t, y, param_t_slice)  # type: ignore[misc]

        if not _has_seeding(self.precomputed):
            return solve_ivp(fun=rhs_from_prepped, t_span=t_span, y0=y0_flat, t_eval=t_eval, **solve_ivp_kwargs)

        import types
        pc = self.precomputed
        seeding_data = pc.get("seeding_data", None)
        seeding_amounts = pc.get("seeding_amounts", None)
        daily_incidence = pc.get("daily_incidence", None)

        t0, tf = float(t_span[0]), float(t_span[1])
        eps = 1e-12
        acc_t: list[FloatArray] = []
        acc_y: list[FloatArray] = []
        total_success = True
        last_message = ""
        y_cur = np.array(y0_flat, dtype=np.float64, copy=True)
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
                Y = np.empty((y0_flat.size, 0), dtype=float)
            return types.SimpleNamespace(t=T, y=Y, success=bool(total_success), message=last_message)

        return types.SimpleNamespace(t=np.array([tf]), y=y_cur.reshape(-1, 1), success=bool(total_success), message=last_message)

# =============================================================================
# 9) Global AUTOTUNE: threads, fastmath on/off, threshold crossover
# =============================================================================

def autotune_all(
    candidate_threads: Sequence[int] | None = None,
    size_grid: Sequence[int] | None = None,
    dtype=np.float64,
    rng_seed: int = 42,
    fm_tol: float = 1e-11,
    quiet: bool = False,
) -> dict:
    """Autotune Numba threads, fastmath, and serial↔parallel crossover.

    Tunes:
      - NUMBA_NUM_THREADS
      - fastmath on/off for the elementwise hotspot (w/ accuracy guard)
      - parallel threshold (serial↔parallel crossover)

    Returns:
      A dict with selected values.
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
    cross_at: int | None = None
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
