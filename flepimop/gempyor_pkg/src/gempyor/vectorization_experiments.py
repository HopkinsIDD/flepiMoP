import numpy as np
import numpy.typing as npt
from numba import njit, prange
from typing import Callable
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# Globals / thresholds
# -----------------------------------------------------------------------------

# Element-count threshold (Tn * N) for switching to parallel kernels
_PARALLEL_THRESHOLD = 10_000_000  # tune as needed

# -----------------------------------------------------------------------------
# Setup helpers (one-time)
# -----------------------------------------------------------------------------


def _prep_param_interpolator(
    parameters: npt.NDArray[np.float64], use_deltas: bool = False
):
    """
    Prepare time-major parameters and (optionally) their time deltas.
    Returns (params_t, deltas_or_None).

    Input:
        parameters: (P, T, N) or (P, T). If (P, T) -> broadcast to (P, T, 1).
    Output:
        params_t: (T, P, N) C-contiguous
        deltas:   (T-1, P, N) C-contiguous or None
    """
    if parameters.ndim == 2:
        parameters = parameters[:, :, None]  # (P,T) -> (P,T,1)
    # (T, P, N) contiguous
    params_t = np.moveaxis(parameters, 1, 0).copy(order="C")
    deltas = None
    if use_deltas and params_t.shape[0] >= 2:
        deltas = np.ascontiguousarray(params_t[1:] - params_t[:-1])  # (T-1, P, N)
    return params_t, deltas


@njit(cache=True, fastmath=True)
def _blend_linear(A: np.ndarray, B: np.ndarray, alpha: float, out: np.ndarray) -> None:
    P, N = out.shape
    for p in range(P):
        for n in range(N):
            out[p, n] = A[p, n] + alpha * (B[p, n] - A[p, n])


@njit(cache=True, fastmath=True)
def _axpy_linear(A: np.ndarray, D: np.ndarray, alpha: float, out: np.ndarray) -> None:
    # out = A + alpha * D
    P, N = out.shape
    for p in range(P):
        for n in range(N):
            out[p, n] = A[p, n] + alpha * D[p, n]


# -----------------------------------------------------------------------------
# Fast slicer
# -----------------------------------------------------------------------------


def _param_slice(
    params_t: npt.NDArray[np.float64],  # (T, P, N), contiguous
    t: float,
    mode: str = "linear",
    out: npt.NDArray[np.float64] | None = None,  # (P, N) buffer to reuse
    deltas: npt.NDArray[np.float64] | None = None,  # (T-1, P, N), optional
) -> npt.NDArray[np.float64]:
    """
    Slice/interpolate parameters at time t.
    params_t must be (T, P, N) C-contiguous (use _prep_param_interpolator).
    """
    T, P, N = params_t.shape
    if out is None:
        out = np.empty((P, N), dtype=np.float64)

    if mode == "step":
        i = int(t)
        if i < 0:
            i = 0
        elif i >= T:
            i = T - 1
        np.copyto(out, params_t[i])  # contiguous block copy
        return out

    if mode == "linear":
        # Degenerate case: only one time point -> no interpolation
        if T == 1:
            np.copyto(out, params_t[0])
            return out

        i0 = int(t)
        if i0 < 0:
            i0 = 0
            alpha = 0.0
        elif i0 >= T - 1:
            i0 = T - 2
            alpha = 1.0
        else:
            alpha = t - i0  # in [0,1)

        if deltas is None:
            _blend_linear(params_t[i0], params_t[i0 + 1], float(alpha), out)
        else:
            _axpy_linear(params_t[i0], deltas[i0], float(alpha), out)
        return out

    raise ValueError(f"Unknown param_time_mode: {mode}")


# -----------------------------------------------------------------------------
# Helper for symbolic parameter expressions (from a time slice)
# -----------------------------------------------------------------------------


def _compile_param_expr(expr: str, param_name_to_row: dict[str, int]) -> np.ndarray:
    """
    Parse 'beta * gamma * kappa' -> int64 row indices, e.g. array([2, 5, 7]).
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
def _product_rows_into(param_t: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
    """
    Compute product across selected rows of param_t into out.
    param_t: (P, N), rows: (K,), out: (N,)
    """
    K = rows.shape[0]
    N = param_t.shape[1]

    # init from first row
    r0 = rows[0]
    for n in range(N):
        out[n] = param_t[r0, n]

    for k in range(1, K):
        rk = rows[k]
        for n in range(N):
            out[n] *= param_t[rk, n]


def _resolve_param_expr_from_slice(
    expr: str | np.ndarray,
    param_t: npt.NDArray[np.float64],  # (P, N)
    param_name_to_row: dict[str, int] | None = None,  # needed if expr is str
    out: npt.NDArray[np.float64] | None = None,  # (N,) buffer to reuse
) -> npt.NDArray[np.float64]:
    """
    Evaluate a product expression like 'beta * gamma' on a (P, N) slice without per-call allocations.

    Usage:
        rows = _compile_param_expr("beta * gamma", name_to_row)  # once
        buf  = np.empty(param_t.shape[1], dtype=np.float64)     # reuse
        v    = _resolve_param_expr_from_slice(rows, param_t, out=buf)
    """
    if isinstance(expr, str):
        if param_name_to_row is None:
            raise ValueError(
                "param_name_to_row must be provided when expr is a string."
            )
        rows = _compile_param_expr(expr, param_name_to_row)
    else:
        rows = expr  # precompiled indices

    if out is None:
        out = np.empty(param_t.shape[1], dtype=np.float64)

    # Ensure contiguity only if necessary to avoid copies
    if not param_t.flags.c_contiguous:
        param_t = np.ascontiguousarray(param_t)

    _product_rows_into(param_t, rows, out)
    return out


# -----------------------------------------------------------------------------
# 1) Core Proportion Logic
# -----------------------------------------------------------------------------


@njit(parallel=True, cache=True, fastmath=True)
def _compute_proportion_sums_exponents(
    states_current: npt.NDArray[np.float64],  # (C, N)
    transitions: npt.NDArray[
        np.int64
    ],  # (5, Tn) [src, dst, param_idx, p_start, p_stop]
    proportion_info: npt.NDArray[np.int64],  # (3, Pk) [sum_start, sum_stop, row_idx]
    transition_sum_compartments: npt.NDArray[np.int64],  # (S,)
    param_t: npt.NDArray[np.float64],  # (P, N)  <-- time-sliced parameters
) -> tuple[
    npt.NDArray[np.float64],  # total_rates: (Tn, N)
    npt.NDArray[np.float64],  # source_numbers: (Tn, N)
    npt.NDArray[np.uint8],  # single_prop_mask: (Tn,)
]:
    """
    Compute multiplicative proportion terms, source sizes, and a flag for n_p == 1.
    """
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]

    total_rates = np.ones((n_transitions, n_nodes), dtype=np.float64)
    source_numbers = np.zeros((n_transitions, n_nodes), dtype=np.float64)
    single_prop_mask = np.zeros(n_transitions, dtype=np.uint8)

    # Parallel over transitions; each iteration writes a disjoint row.
    for t_idx in prange(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        n_p = p_stop - p_start

        if n_p == 1:
            single_prop_mask[t_idx] = 1  # mark for skipping mobility later

        # Initialize running product directly in the output row (in-place).
        first = True

        for p_idx in range(p_start, p_stop):
            sum_start = proportion_info[0, p_idx]
            sum_stop = proportion_info[1, p_idx]
            row_idx = proportion_info[2, p_idx]

            # Sum of selected compartments → (N,)
            summed = states_current[
                transition_sum_compartments[sum_start:sum_stop], :
            ].sum(axis=0)

            expnt_vec = param_t[row_idx, :]  # (N,)
            summed_exp = summed**expnt_vec  # (N,)

            if first:
                # Save source sizes for the first proportion term
                source_numbers[t_idx, :] = summed

                # Normalize first term by its source size (legacy behavior).
                # Avoid divide-by-zero by substituting 1 where summed == 0.
                safe_src = np.where(summed > 0.0, summed, 1.0)
                contrib = summed_exp / safe_src  # (N,)

                if n_p == 1:
                    # Legacy behavior: multiply by parameter immediately.
                    param_idx = transitions[2, t_idx]
                    contrib *= param_t[param_idx, :]

                # Multiply into the row in-place.
                for n in range(n_nodes):
                    total_rates[t_idx, n] *= contrib[n]

                first = False
            else:
                # Additional proportion contributions multiply directly.
                for n in range(n_nodes):
                    total_rates[t_idx, n] *= summed_exp[n]

    return total_rates, source_numbers, single_prop_mask


# -----------------------------------------------------------------------------
# 2) Transition Rates (CSR mobility core + wrapper)
# -----------------------------------------------------------------------------


@njit(parallel=True, cache=True, fastmath=True)
def _compute_transition_rates_core(
    total_rates_base: npt.NDArray[np.float64],  # (Tn, N)
    transitions: npt.NDArray[
        np.int64
    ],  # (5, Tn) [src, dst, param_idx, p_start, p_stop]
    param_vec_by_tr: npt.NDArray[np.float64],  # (Tn, N)
    percent_day_away: float,
    prop_who_move: npt.NDArray[np.float64],  # (N,)
    csr_data: npt.NDArray[np.float64],  # (nnz,)
    csr_indptr: npt.NDArray[np.int64],  # (N+1,)
    csr_indices: npt.NDArray[np.int64],  # (nnz,)
    population: npt.NDArray[np.float64],  # (N,)
    single_prop_mask: npt.NDArray[np.uint8],  # (Tn,)
) -> npt.NDArray[np.float64]:
    """
    For single_prop_mask==1, copies base through.
    Else: parameter scaling + mobility mix (CSR-like SpMV per row).
    """
    Tn, N = total_rates_base.shape
    out = np.empty_like(total_rates_base)

    # Per-node constants
    inv_pop = np.empty(N, dtype=np.float64)
    keep = np.empty(N, dtype=np.float64)
    for n in range(N):
        pop = population[n]
        pop_safe = pop if pop > 0.0 else 1.0
        inv_pop[n] = 1.0 / pop_safe
        keep[n] = 1.0 - percent_day_away * prop_who_move[n]

    # Parallel over transitions; each iteration writes disjoint row
    for t_idx in prange(Tn):
        if single_prop_mask[t_idx] == 1:
            for n in range(N):
                out[t_idx, n] = total_rates_base[t_idx, n]
            continue

        # base_force = (total_rates_base * param_vec) / pop
        base_force = np.empty(N, dtype=np.float64)
        for n in range(N):
            base_force[n] = (
                total_rates_base[t_idx, n] * param_vec_by_tr[t_idx, n]
            ) * inv_pop[n]

        # CSR mixing
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
    source_numbers: npt.NDArray[np.float64],  # (Tn, N)  [unused; kept for API]
    transitions: npt.NDArray[np.int64],  # (5, Tn)
    param_t: npt.NDArray[np.float64],  # (P, N)
    percent_day_away: float,
    proportion_who_move: npt.NDArray[np.float64],  # (N,)
    mobility_data: npt.NDArray[np.float64],  # CSR data (nnz,)
    mobility_data_indices: npt.NDArray[np.int64],  # CSR indptr (N+1,)
    mobility_row_indices: npt.NDArray[np.int64],  # CSR indices (nnz,)
    population: npt.NDArray[np.float64],  # (N,)
    single_prop_mask: npt.NDArray[np.uint8],  # (Tn,)
    param_expr_lookup: dict[int, str | np.ndarray] | None = None,
    param_name_to_row: dict[str, int] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Wrapper building per-transition parameter vectors once, then calling the njit core.
    If param_expr_lookup maps to np.ndarray (precompiled rows), no parsing occurs.
    """
    Tn, N = total_rates_base.shape

    # Prepare param vectors per transition
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
                # precompiled row indices
                _resolve_param_expr_from_slice(val, param_t, out=buf)
            else:
                # string expression: parse using provided mapping
                if param_name_to_row is None:
                    raise ValueError(
                        "param_name_to_row required when param_expr_lookup contains strings."
                    )
                _resolve_param_expr_from_slice(val, param_t, param_name_to_row, out=buf)
            param_vec_by_tr[t, :] = buf  # copy

    indptr = np.ascontiguousarray(mobility_data_indices, dtype=np.int64)
    indices = np.ascontiguousarray(mobility_row_indices, dtype=np.int64)
    data = np.ascontiguousarray(mobility_data, dtype=np.float64)

    return _compute_transition_rates_core(
        np.ascontiguousarray(total_rates_base, dtype=np.float64),
        np.ascontiguousarray(transitions, dtype=np.int64),
        np.ascontiguousarray(param_vec_by_tr, dtype=np.float64),
        float(percent_day_away),
        np.ascontiguousarray(proportion_who_move, dtype=np.float64),
        data,
        indptr,
        indices,
        np.ascontiguousarray(population, dtype=np.float64),
        np.ascontiguousarray(single_prop_mask, dtype=np.uint8),
    )


# -----------------------------------------------------------------------------
# 3) Binomial Stochastic (NumPy) — optional, outside deterministic RHS
# -----------------------------------------------------------------------------


def _compute_transition_amounts_numpy_binomial(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """
    Perform binomial draws for stochastic updates over a time step (returns counts, not dy/dt).
    """
    probs = 1.0 - np.exp(-dt * total_rates)
    probs = np.clip(probs, 0.0, 1.0)
    n = np.clip(source_numbers, 0, np.inf).astype(np.int64, copy=False)
    draws = np.random.binomial(n, probs)
    return draws.astype(np.float64, copy=False)


# -----------------------------------------------------------------------------
# 4) Transition Amounts (Deterministic dy/dt only)
# -----------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _compute_transition_amounts_serial(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    m, n = total_rates.shape
    out = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            out[i, j] = source_numbers[i, j] * total_rates[i, j]
    return out


@njit(parallel=True, cache=True, fastmath=True)
def _compute_transition_amounts_parallel(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    m, n = total_rates.shape
    size = m * n
    out = np.empty((m, n), dtype=np.float64)

    src = source_numbers.ravel()
    rate = total_rates.ravel()
    outf = out.ravel()

    for k in prange(size):
        outf[k] = src[k] * rate[k]

    return out


@njit(cache=True)
def compute_transition_amounts_meta(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    workload = source_numbers.shape[0] * source_numbers.shape[1]
    if workload >= _PARALLEL_THRESHOLD:
        return _compute_transition_amounts_parallel(source_numbers, total_rates)
    else:
        return _compute_transition_amounts_serial(source_numbers, total_rates)


# -----------------------------------------------------------------------------
# 5) Assemble Flux Vector
# -----------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _assemble_flux(
    amounts: npt.NDArray[np.float64],  # (Tn, N)
    transitions: npt.NDArray[np.int64],  # (5, Tn) [src, dst, ...]
    ncompartments: int,
    nspatial_nodes: int,
) -> npt.NDArray[np.float64]:
    """
    Assemble dy/dt by subtracting from source compartment rows and adding to destination rows, per node.
    """
    Tn = amounts.shape[0]
    N = nspatial_nodes

    dy_dt = np.zeros((ncompartments, N), dtype=np.float64)

    for t_idx in range(Tn):
        src = int(transitions[0, t_idx])
        dst = int(transitions[1, t_idx])

        # 1-D views for fast inner loop
        src_row = dy_dt[src, :]
        dst_row = dy_dt[dst, :]
        a_row = amounts[t_idx, :]

        for n in range(N):
            a = a_row[n]
            src_row[n] -= a
            dst_row[n] += a

    return dy_dt.ravel()  # C-order view, solver-friendly


# -----------------------------------------------------------------------------
# 6) Seeding
# -----------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _apply_legacy_seeding_core(
    states_current: np.ndarray,  # (C, N), modified in place
    today: int,
    day_start_idx: np.ndarray,  # (D+1,) int64 CSR-like pointers
    seeding_subpops: np.ndarray,  # (E,)   int64
    seeding_sources: np.ndarray,  # (E,)   int64
    seeding_dests: np.ndarray,  # (E,)   int64
    seeding_amounts: np.ndarray,  # (E,)   float64
    daily_incidence: np.ndarray,  # (D, C, N) or (0,0,0) sentinel
    update_incidence: int,  # 1 if daily_incidence valid, else 0
) -> None:
    # Bounds check: valid today in [0, day_start_idx.size-2]
    if today < 0 or today + 1 >= day_start_idx.size:
        return

    start_idx = int(day_start_idx[today])
    stop_idx = int(day_start_idx[today + 1])
    if stop_idx <= start_idx:
        return

    for e in range(start_idx, stop_idx):
        g = int(seeding_subpops[e])
        s = int(seeding_sources[e])
        d = int(seeding_dests[e])
        amt = seeding_amounts[e]

        # legacy: subtract, clamp at zero, then add full amt
        states_current[s, g] -= amt
        if states_current[s, g] < 0.0:
            states_current[s, g] = 0.0
        states_current[d, g] += amt

        if update_incidence == 1:
            daily_incidence[today, d, g] += amt


def _apply_legacy_seeding(
    states_current: npt.NDArray[np.float64],  # (C, N), modified in place
    today: int,
    seeding_data: dict[str, npt.NDArray[np.float64]],
    seeding_amounts: npt.NDArray[np.float64],
    daily_incidence: npt.NDArray[np.float64] | None = None,  # (D, C, N) if provided
) -> None:
    """
    Apply legacy-style seeding to the model state for a given day (in place).
    Matches rk4_integration behavior: subtract then clamp source to zero, add full amt.
    """
    # Pull arrays once and enforce expected dtypes/contiguity (no-op if already correct)
    day_start_idx = np.ascontiguousarray(seeding_data["day_start_idx"], dtype=np.int64)
    seeding_subpops = np.ascontiguousarray(
        seeding_data["seeding_subpops"], dtype=np.int64
    )
    seeding_sources = np.ascontiguousarray(
        seeding_data["seeding_sources"], dtype=np.int64
    )
    seeding_dests = np.ascontiguousarray(
        seeding_data["seeding_destinations"], dtype=np.int64
    )
    amts = np.ascontiguousarray(seeding_amounts, dtype=states_current.dtype)

    if daily_incidence is None:
        # Pass a zero-sized sentinel so njit signature stays the same
        di = np.empty((0, 0, 0), dtype=states_current.dtype)
        upd = 0
    else:
        di = daily_incidence
        upd = 1

    _apply_legacy_seeding_core(
        states_current,
        int(today),
        day_start_idx,
        seeding_subpops,
        seeding_sources,
        seeding_dests,
        amts,
        di,
        upd,
    )


# -----------------------------------------------------------------------------
# 7) Factory class (solve_ivp-compatible) with auto-dispatch
# -----------------------------------------------------------------------------


def _has_seeding(precomputed: dict) -> bool:
    """
    Decide whether we should enable the seeding-aware solve path.

    Returns True only if:
      - 'seeding_data' and 'seeding_amounts' exist,
      - day_start_idx exists,
      - and there is at least one event scheduled (day_start_idx[-1] > 0) AND
        seeding_amounts.size > 0.
    """
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
    # Number of events is encoded by the last pointer in CSR-like layout
    has_events = bool(dsi[-1] > 0)
    return has_events and (np.asarray(sa).size > 0)


class RHSfactory:
    """
    Wraps precomputed model structures and builds an RHS for SciPy's solve_ivp.
    Auto-selects a fast path when seeding is absent, and a piecewise-daily
    integration path when seeding is present.
    """

    REQUIRED_KEYS = [
        "ncompartments",
        "nspatial_nodes",
        "transitions",
        "proportion_info",
        "transition_sum_compartments",
        "percent_day_away",
        "proportion_who_move",
        "mobility_data",
        "mobility_data_indices",
        "mobility_row_indices",
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

        self._last_day_applied = {"day": None}
        self._rhs = None
        self._rhs_core = None  # set by build_rhs()

        self.build_rhs()

    def _reset_seeding_tracker(self) -> None:
        self._last_day_applied["day"] = None

    def build_rhs(
        self,
        *,
        param_time_mode: str | None = None,
        seeding_data: dict[str, npt.NDArray[np.float64]] | None = None,
        seeding_amounts: npt.NDArray[np.float64] | None = None,
        daily_incidence: npt.NDArray[np.float64] | None = None,
    ) -> Callable[
        [float, npt.NDArray[np.float64], npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]:
        """
        Build and cache:
          - _rhs_core(t, y, param_t_slice)  -> dy/dt
          - rhs(t, y, parameters)           -> slices then calls _rhs_core
        Note: seeding is handled in solve(); the public rhs() here does NOT perform seeding.
        """
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

        # One-time dtype/contiguity normalization
        transitions = np.ascontiguousarray(pc["transitions"], dtype=np.int64)  # (5, Tn)
        proportion_info = np.ascontiguousarray(
            pc["proportion_info"], dtype=np.int64
        )  # (3, Pk)
        transition_sum_compartments = np.ascontiguousarray(
            pc["transition_sum_compartments"], dtype=np.int64
        )
        percent_day_away = float(pc["percent_day_away"])
        proportion_who_move = np.ascontiguousarray(
            pc["proportion_who_move"], dtype=np.float64
        )  # (N,)
        mobility_data = np.ascontiguousarray(pc["mobility_data"], dtype=np.float64)
        mobility_data_indices = np.ascontiguousarray(
            pc["mobility_data_indices"], dtype=np.int64
        )  # indptr
        mobility_row_indices = np.ascontiguousarray(
            pc["mobility_row_indices"], dtype=np.int64
        )  # indices
        population = np.ascontiguousarray(pc["population"], dtype=np.float64)

        # Precompile parameter expressions (if provided) to row-index arrays once
        param_rows_lookup: dict[int, np.ndarray] | None = None
        if self.param_expr_lookup is not None:
            if self.param_name_to_row is None:
                raise ValueError(
                    "param_name_to_row required when param_expr_lookup is provided."
                )
            param_rows_lookup = {
                k: _compile_param_expr(v, self.param_name_to_row)
                for k, v in self.param_expr_lookup.items()
            }

        # ---- core: expects param_t_slice (P, N) already prepared for time t ----
        def _rhs_core(
            t: float,
            y: npt.NDArray[np.float64],
            param_t_slice: npt.NDArray[np.float64],  # (P, N)
        ) -> npt.NDArray[np.float64]:

            states_current = y.reshape((C, N))

            # 3) base proportion terms + source sizes
            total_base, source_numbers, single_prop_mask = (
                _compute_proportion_sums_exponents(
                    states_current,
                    transitions,
                    proportion_info,
                    transition_sum_compartments,
                    param_t_slice,
                )
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
                # pass precompiled rows if available; else None
                param_expr_lookup=param_rows_lookup,
                param_name_to_row=None,
                single_prop_mask=single_prop_mask,
            )

            # 5) instantaneous flux and assembly: dy/dt
            amounts = compute_transition_amounts_meta(source_numbers, total_rates)
            dy = _assemble_flux(amounts, transitions, C, N)
            return dy

        # ---- outer rhs: parameter time-slicing only (no seeding here) ----
        def rhs(
            t: float, y: npt.NDArray[np.float64], parameters: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            params_t, deltas = _prep_param_interpolator(
                parameters, use_deltas=(param_time_mode == "linear")
            )
            buf = np.empty((params_t.shape[1], params_t.shape[2]), dtype=np.float64)
            param_t_slice = _param_slice(
                params_t, t, mode=param_time_mode, out=buf, deltas=deltas
            )
            return _rhs_core(t, y, param_t_slice)

        self._rhs_core = _rhs_core
        self._rhs = rhs
        return rhs

    def create_rhs(
        self,
    ) -> Callable[
        [float, npt.NDArray[np.float64], npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]:
        if self._rhs is None:
            return self.build_rhs()
        return self._rhs

    def solve(
        self,
        y0: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],  # (P, T, N) or (P, T)
        t_span: tuple[float, float],
        t_eval: npt.NDArray[np.float64] | None = None,
        **solve_ivp_kwargs,
    ):
        """
        Auto-dispatch:
          - If seeding is absent -> fast single-call integration with precomputed param slicing.
          - If seeding is present -> piecewise-daily integration applying seeding at day starts.
        """
        self._reset_seeding_tracker()
        if self._rhs_core is None:
            self.build_rhs()
        rhs_core = self._rhs_core

        # Precompute time-major params and deltas once (used by both paths)
        use_deltas = self.param_time_mode == "linear"
        params_t, deltas = _prep_param_interpolator(parameters, use_deltas=use_deltas)
        buf = np.empty((params_t.shape[1], params_t.shape[2]), dtype=np.float64)

        # Common closure: slice from prepped (T,P,N) into reusable (P,N) buffer, then core
        def rhs_from_prepped(
            t: float, y: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            param_t_slice = _param_slice(
                params_t, t, mode=self.param_time_mode, out=buf, deltas=deltas
            )
            return rhs_core(t, y, param_t_slice)

        # Dispatch based on presence of seeding events
        if not _has_seeding(self.precomputed):
            # ---- Fast path: no seeding at all, single solve_ivp call ----
            return solve_ivp(
                fun=rhs_from_prepped,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                **solve_ivp_kwargs,
            )

        # ---- Seeding path: piecewise integrate across integer-day boundaries ----
        import types

        pc = self.precomputed
        C = int(pc["ncompartments"])
        N = int(pc["nspatial_nodes"])
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

        # Helper to slice t_eval for a segment, excluding the left endpoint for non-first segments
        def _slice_t_eval(
            t_start: float, t_end: float, first_segment: bool
        ) -> npt.NDArray[np.float64] | None:
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
            # Next integer boundary AFTER (or at) t_cur
            next_boundary = np.floor(t_cur + eps) + 1.0
            seg_end = min(tf, next_boundary)

            # Guard against zero-length segments due to rounding
            if seg_end <= t_cur + eps:
                t_cur = seg_end + eps
                first = False
                continue

            # Apply seeding at the start of an integer day only
            day_start = int(np.round(t_cur))
            if abs(t_cur - float(day_start)) < 1e-12:
                states_current = y_cur.reshape((C, N))
                _apply_legacy_seeding(
                    states_current,
                    day_start,
                    seeding_data,
                    seeding_amounts,
                    daily_incidence=daily_incidence,
                )
                y_cur = states_current.ravel()

            # Determine t_eval for this segment
            t_eval_seg = _slice_t_eval(t_cur, seg_end, first_segment=first)

            # Integrate this segment
            res = solve_ivp(
                fun=rhs_from_prepped,
                t_span=(t_cur, seg_end),
                y0=y_cur,
                t_eval=t_eval_seg,
                **solve_ivp_kwargs,
            )
            total_success = total_success and bool(res.success)
            last_message = res.message

            # Coerce to arrays (res.t / res.y may be Python lists in some builds)
            t_arr = np.asarray(res.t)
            y_arr = np.asarray(res.y)

            # Accumulate only if this segment had evaluation points
            if t_eval_seg is not None and t_arr.size > 0:
                acc_t.append(t_arr)
                acc_y.append(y_arr)

            # Advance current state to the segment end
            if y_arr.size > 0:
                y_cur = y_arr[:, -1]

            # Move to next segment
            t_cur = seg_end
            first = False

        # If t_eval provided, stitch results
        if t_eval is not None:
            if acc_t:
                T = np.concatenate(acc_t)
                Y = np.concatenate(acc_y, axis=1)
            else:
                # No points evaluated; return empty arrays with correct shapes
                T = np.array([], dtype=float)
                Y = np.empty((y0.size, 0), dtype=float)

            return types.SimpleNamespace(
                t=T,
                y=Y,
                success=bool(total_success),
                message=last_message,
            )

        # Fallback: no t_eval. Return final state at tf.
        return types.SimpleNamespace(
            t=np.array([tf]),
            y=y_cur.reshape(-1, 1),
            success=bool(total_success),
            message=last_message,
        )
