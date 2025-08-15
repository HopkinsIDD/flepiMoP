import numpy as np
from numba import njit, prange
from scipy.interpolate import interp1d
from collections.abc import Callable

# === Debug switch ===
DEBUG = False  # set to False to silence debug prints


def _dbg_stats(name: str, arr: np.ndarray, prefix: str = "[DBG]"):
    if not DEBUG:
        return
    a = np.asarray(arr)
    if a.size == 0:
        print(f"{prefix} {name}: EMPTY")
        return
    # Use nan-safe reductions
    amin = np.nanmin(a)
    amax = np.nanmax(a)
    any_nan = np.isnan(a).any()
    any_inf = np.isinf(a).any()
    print(
        f"{prefix} {name}: shape={a.shape}, dtype={a.dtype}, min={amin}, max={amax}, nan={any_nan}, inf={any_inf}"
    )


def _dbg_nonfinite(name: str, arr: np.ndarray, prefix: str = "[DBG]"):
    if not DEBUG:
        return
    mask = ~np.isfinite(arr)
    if mask.any():
        idx = np.argwhere(mask)
        sample = idx[:10]  # avoid huge dumps
        print(
            f"{prefix} NON-FINITE in {name}: count={mask.sum()}, sample_indices={sample.tolist()}"
        )


# === Constants ===
_PARALLEL_THRESHOLD = 1e7


# === Parameter time slicing (for solve_ivp's continuous t) ===


def param_slice(parameters: np.ndarray, t: float, mode: str = "linear") -> np.ndarray:
    """
    Slice parameters at time t.

    Parameters
    ----------
    parameters : np.ndarray
        Shape (P, T, N) or (P, T). T is unit-spaced discrete time (0,1,2,...).
        If (P, T), it's broadcast to (P, T, 1).
    t : float
        Continuous time used by the solver.
    mode : {"step","linear"}
        "step"  => piecewise-constant in time (floor(t))
        "linear"=> linear interpolation between floor(t) and ceil(t)

    Returns
    -------
    param_t : np.ndarray
        Shape (P, N) view at time t.
    """
    if parameters.ndim == 2:  # (P, T) -> (P, T, 1)
        parameters = parameters[:, :, None]

    P, T, N = parameters.shape

    if mode == "step":
        i = int(np.clip(np.floor(t), 0, T - 1))
        out = parameters[:, i, :]
        if DEBUG:
            _dbg_stats("param_slice(step)", out)
        return out

    elif mode == "linear":
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, T - 1)
        alpha = float(np.clip(t - i0, 0.0, 1.0))
        out = (1.0 - alpha) * parameters[:, i0, :] + alpha * parameters[:, i1, :]
        if DEBUG:
            _dbg_stats("param_slice(linear)", out)
        return out

    else:
        raise ValueError(f"Unknown param_time_mode: {mode}")


# === Helper for symbolic expression resolution (from a time slice) ===


def resolve_param_expr_from_slice(
    expr: str,
    param_t: np.ndarray,  # (P, N)
    param_name_to_row: dict[str, int],  # "beta" -> row index
) -> np.ndarray:
    """
    Resolve a product expression like 'beta * gamma' against a (P,N) parameter slice.

    Returns
    -------
    vec : np.ndarray of shape (N,)
    """
    terms = [s.strip() for s in expr.split("*")]
    out = None
    for name in terms:
        row = param_name_to_row[name]
        vec = param_t[row, :]  # (N,)
        out = vec if out is None else out * vec
    if DEBUG:
        _dbg_stats(f"resolve_param_expr_from_slice('{expr}')", out)
        _dbg_nonfinite(f"resolve_param_expr_from_slice('{expr}')", out)
    return out


# === 1. Core Proportion Logic ===


@njit(fastmath=True)
def prod_along_axis0(arr_2d: np.ndarray) -> np.ndarray:
    """
    Computes the product along the first axis (axis=0) of a 2D array.

    We don't use np.prod here to keep this nopython-friendly.
    """
    n_cols = arr_2d.shape[1]
    result = np.ones(n_cols, dtype=arr_2d.dtype)
    for i in range(arr_2d.shape[0]):
        for j in range(n_cols):
            result[j] *= arr_2d[i, j]
    return result


@njit(fastmath=True)
def compute_proportion_sums_exponents(
    states_current: np.ndarray,  # (C, N)
    transitions: np.ndarray,  # (5, Tn)
    proportion_info: np.ndarray,  # (3, Pk)
    transition_sum_compartments: np.ndarray,  # (S,)
    param_t: np.ndarray,  # (P, N)  <-- time-sliced parameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multiplicative proportion terms, source sizes, and a flag for n_p == 1.

    Matches legacy `rk4_integration` math:
    - If only one proportion term (n_p == 1), we normalize the first proportion term
      and multiply by the rate parameter immediately.
    - Otherwise, we compute proportion contributions for all terms and leave parameter
      multiplication / mobility mixing to later.

    Returns
    -------
    total_rates : (Tn, N)  # proportion products (includes param if n_p==1)
    source_numbers : (Tn, N)  # summed counts for first proportion term
    single_prop_mask : (Tn,)  # boolean flag for n_p == 1 transitions
    """
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]
    n_props = proportion_info.shape[1]

    total_rates = np.ones((n_transitions, n_nodes))
    source_numbers = np.zeros((n_transitions, n_nodes))
    proportion_contribs = np.zeros((n_props, n_nodes))
    single_prop_mask = np.zeros(n_transitions, dtype=np.uint8)

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        n_p = p_stop - p_start
        first = True

        if n_p == 1:
            single_prop_mask[t_idx] = 1  # mark for skipping mobility later

        for p_idx in range(p_start, p_stop):
            sum_start = proportion_info[0, p_idx]
            sum_stop = proportion_info[1, p_idx]
            row_idx = proportion_info[2, p_idx]

            # sum of selected compartments
            summed = states_current[
                transition_sum_compartments[sum_start:sum_stop], :
            ].sum(
                axis=0
            )  # (N,)

            expnt_vec = param_t[row_idx, :]  # (N,)
            summed_exp = summed**expnt_vec  # (N,)

            if first:
                source_numbers[t_idx, :] = summed
                safe_src = np.where(summed > 0.0, summed, 1.0)
                contrib = summed_exp / safe_src
                if n_p == 1:
                    # legacy behavior: multiply by param immediately
                    param_idx = transitions[2, t_idx]
                    contrib *= param_t[param_idx, :]
                proportion_contribs[p_idx, :] = contrib
                first = False
            else:
                proportion_contribs[p_idx, :] = summed_exp

        if n_p > 0:
            total_rates[t_idx, :] *= prod_along_axis0(
                proportion_contribs[p_start:p_stop, :]
            )

    return total_rates, source_numbers, single_prop_mask


# === 2. Transition Rate Computation ===
def compute_transition_rates(
    total_rates_base: np.ndarray,  # (Tn, N) e.g., I for S->E
    source_numbers: np.ndarray,  # (Tn, N) e.g., S for S->E  (NOT used here)
    transitions: np.ndarray,  # (5, Tn)
    param_t: np.ndarray,  # (P, N)
    percent_day_away: float,
    proportion_who_move: np.ndarray,  # (N,)
    mobility_data: np.ndarray,
    mobility_data_indices: np.ndarray,
    mobility_row_indices: np.ndarray,
    population: np.ndarray,  # (N,)
    single_prop_mask: np.ndarray,  # (Tn,)
    param_expr_lookup: dict[int, str] | None = None,
    param_name_to_row: dict[str, int] | None = None,
) -> np.ndarray:
    if DEBUG:
        _dbg_stats("compute_transition_rates.total_rates_base", total_rates_base)
        _dbg_stats("compute_transition_rates.param_t", param_t)

    n_transitions, n_nodes = total_rates_base.shape
    # IMPORTANT: start from zeros; we will assign the mixed “force of transition”
    total_rates = np.zeros_like(total_rates_base)

    for t_idx in range(n_transitions):

        # If there was exactly one proportion term (e.g., E->I with sigma*E),
        # the parameter has already been applied in compute_proportion_sums_exponents,
        # and there is no mobility mixing. Just carry the base through.
        if single_prop_mask[t_idx]:
            total_rates[t_idx, :] = total_rates_base[t_idx, :]
            continue

        param_idx = transitions[2, t_idx]

        # Resolve parameter vector (name or expression)
        if param_expr_lookup is None:
            param_vec = param_t[param_idx, :]
        else:
            if param_name_to_row is None:
                raise ValueError(
                    "param_name_to_row required when param_expr_lookup is provided."
                )
            expr = param_expr_lookup[param_idx]
            param_vec = resolve_param_expr_from_slice(expr, param_t, param_name_to_row)

        if DEBUG:
            _dbg_stats(f"param_vec[t_idx={t_idx}]", param_vec)

        # Mix infectious pressure across mobility; never multiply by S here.
        for node in range(n_nodes):
            pop_n = population[node]
            pop_n_safe = pop_n if pop_n > 0.0 else 1.0

            prop_keep = 1.0 - percent_day_away * proportion_who_move[node]

            # slice CSR
            v_slice = slice(
                mobility_data_indices[node], mobility_data_indices[node + 1]
            )
            visitors = mobility_row_indices[v_slice]
            mobility_vals = mobility_data[v_slice]
            prop_change = percent_day_away * mobility_vals / pop_n_safe

            # residents’ infectious pressure (e.g., I/N) scaled by beta at this node
            force_keep = param_vec[node] * (total_rates_base[t_idx, node] / pop_n_safe)

            # visitors’ infectious pressure
            force_change = 0.0
            for i in range(visitors.size):
                v = visitors[i]
                pop_v = population[v]
                pop_v_safe = pop_v if pop_v > 0.0 else 1.0
                force_change += prop_change[i] * (
                    param_vec[v] * (total_rates_base[t_idx, v] / pop_v_safe)
                )

            total_rates[t_idx, node] = prop_keep * force_keep + force_change

    if DEBUG:
        _dbg_stats("compute_transition_rates.total_rates(OUT)", total_rates)

    return total_rates


# === 3. Binomial Stochastic (NumPy) ===
# (Kept for optional tau-leaping outside solve_ivp; not used in deterministic RHS.)


def compute_transition_amounts_numpy_binomial(
    source_numbers: np.ndarray, total_rates: np.ndarray, dt: float
) -> np.ndarray:
    """
    Binomial draws for stochastic updates over dt. Returns per-step counts (not dy/dt).
    """
    probs = 1.0 - np.exp(-dt * total_rates)
    draws = np.random.binomial(source_numbers.astype(np.int32), probs)
    return draws.astype(np.float32)


# === 4. Transition Amounts (Deterministic dy/dt only) ===


@njit(fastmath=True)
def compute_transition_amounts_serial(
    source_numbers: np.ndarray, total_rates: np.ndarray
) -> np.ndarray:
    """
    Deterministic instantaneous flux (dy/dt contribution), serial.
    amounts = source * rate  (units: per time)
    """
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))
    for t_idx in range(n_transitions):
        for node in range(n_nodes):
            amounts[t_idx, node] = (
                source_numbers[t_idx, node] * total_rates[t_idx, node]
            )
    return amounts


@njit(parallel=True, fastmath=True)
def compute_transition_amounts_parallel(
    source_numbers: np.ndarray, total_rates: np.ndarray
) -> np.ndarray:
    """
    Deterministic instantaneous flux (dy/dt contribution), parallel.
    """
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))
    for t_idx in prange(n_transitions):
        for node in range(n_nodes):
            amounts[t_idx, node] = (
                source_numbers[t_idx, node] * total_rates[t_idx, node]
            )
    return amounts


@njit
def compute_transition_amounts_meta(
    source_numbers: np.ndarray, total_rates: np.ndarray
) -> np.ndarray:
    """
    Dispatch based on workload; returns dy/dt contributions (no dt, no method).
    """
    workload = source_numbers.shape[0] * source_numbers.shape[1]
    if workload >= _PARALLEL_THRESHOLD:
        return compute_transition_amounts_parallel(source_numbers, total_rates)
    else:
        return compute_transition_amounts_serial(source_numbers, total_rates)


# === 5. Assemble Flux Vector ===


@njit(fastmath=True)
def assemble_flux(
    amounts: np.ndarray,  # (Tn, N), instantaneous flux
    transitions: np.ndarray,  # (5, Tn)
    ncompartments: int,
    nspatial_nodes: int,
) -> np.ndarray:
    """
    Assemble dy/dt from transition fluxes.
    """
    dy_dt = np.zeros((ncompartments, nspatial_nodes))
    for t_idx in range(amounts.shape[0]):
        src = transitions[0, t_idx]
        dst = transitions[1, t_idx]
        for node in range(nspatial_nodes):
            a = amounts[t_idx, node]
            dy_dt[src, node] -= a
            dy_dt[dst, node] += a
    return dy_dt.ravel()


# === Seeding ===


def apply_legacy_seeding(
    states_current: np.ndarray,  # (C, N), modified in place
    today: int,
    seeding_data: dict[str, np.ndarray],
    seeding_amounts: np.ndarray,
    daily_incidence: np.ndarray | None = None,  # (D, C, N) if provided
) -> None:
    starts = seeding_data["day_start_idx"]
    if today < 0 or today + 1 >= starts.size:
        return
    start_idx = int(starts[today])
    stop_idx = int(starts[min(today + 1, starts.size - 1)])
    if stop_idx <= start_idx:
        return

    nodes = seeding_data["seeding_subpops"][start_idx:stop_idx].astype(
        np.int64, copy=False
    )
    srcs = seeding_data["seeding_sources"][start_idx:stop_idx].astype(
        np.int64, copy=False
    )
    dsts = seeding_data["seeding_destinations"][start_idx:stop_idx].astype(
        np.int64, copy=False
    )
    amts = seeding_amounts[start_idx:stop_idx].astype(states_current.dtype, copy=False)

    for g, s, d, amt in zip(nodes, srcs, dsts, amts):
        states_current[s, g] -= amt
        if states_current[s, g] < 0.0:
            states_current[s, g] = 0.0
        states_current[d, g] += amt
        if daily_incidence is not None:
            daily_incidence[today, d, g] += amt


# === 9. factory class (solve_ivp-compatible) ===

# --- factory with build_rhs as a method ---
from scipy.integrate import solve_ivp


class RHSfactory:
    """
    Wraps precomputed structures and builds an rhs(t, y, parameters) method suitable for solve_ivp.
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
        # validate keys
        for k in self.REQUIRED_KEYS:
            if k not in precomputed:
                raise KeyError(f"precomputed is missing required key: '{k}'")

        self.precomputed = precomputed
        self.param_expr_lookup = param_expr_lookup
        self.param_name_to_row = param_name_to_row
        self.param_time_mode = param_time_mode

        # tracker survives across RHS calls
        self._last_day_applied = {"day": None}
        self._rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray] | None = None

        # build once using defaults present in precomputed (if any)
        self.build_rhs()

    def reset_seeding_tracker(self) -> None:
        """Reset the day-change tracker (useful before a new solve)."""
        self._last_day_applied["day"] = None

    def build_rhs(
        self,
        *,
        param_time_mode: str | None = None,
        seeding_data: dict[str, np.ndarray] | None = None,
        seeding_amounts: np.ndarray | None = None,
        daily_incidence: np.ndarray | None = None,
    ) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
        """
        Construct and store an RHS that optionally injects legacy seeding once per day boundary.
        Call again to rebuild with different options.
        """
        pc = self.precomputed
        C = int(pc["ncompartments"])
        N = int(pc["nspatial_nodes"])

        # default to ctor's values / precomputed payloads
        if param_time_mode is None:
            param_time_mode = self.param_time_mode
        if seeding_data is None:
            seeding_data = pc.get("seeding_data", None)
        if seeding_amounts is None:
            seeding_amounts = pc.get("seeding_amounts", None)
        if daily_incidence is None:
            daily_incidence = pc.get("daily_incidence", None)

        # local aliases for speed/readability
        transitions = pc["transitions"]
        proportion_info = pc["proportion_info"]
        transition_sum_compartments = pc["transition_sum_compartments"]
        percent_day_away = pc["percent_day_away"]
        proportion_who_move = pc["proportion_who_move"]
        mobility_data = pc["mobility_data"]
        mobility_data_indices = pc["mobility_data_indices"]
        mobility_row_indices = pc["mobility_row_indices"]
        population = pc["population"]
        param_expr_lookup = self.param_expr_lookup
        param_name_to_row = self.param_name_to_row
        last_day_applied = self._last_day_applied  # closure to persist across calls

        # ---- define RHS ----
        def rhs(t: float, y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
            # 1) inject seeding (once per integer day) before computing rates
            if seeding_data is not None and seeding_amounts is not None:
                # small epsilon to stabilize floor at boundaries
                today = int(np.floor(t + 1e-12))
                if last_day_applied["day"] != today:
                    states_current = y.reshape((C, N))
                    apply_legacy_seeding(
                        states_current,
                        today,
                        seeding_data,
                        seeding_amounts,
                        daily_incidence=daily_incidence,
                    )
                    y[:] = states_current.ravel()
                    last_day_applied["day"] = today

            # 2) parameter time-slice
            param_t = param_slice(parameters, t, mode=param_time_mode)

            # 3) base terms
            states_current = y.reshape((C, N))
            total_base, source_numbers, single_prop_mask = (
                compute_proportion_sums_exponents(
                    states_current,
                    transitions,
                    proportion_info,
                    transition_sum_compartments,
                    param_t,
                )
            )

            # 4) scaling + mobility (mask controls single-proportion fast path)
            total_rates = compute_transition_rates(
                total_rates_base=total_base,
                source_numbers=source_numbers,
                transitions=transitions,
                param_t=param_t,
                percent_day_away=percent_day_away,
                proportion_who_move=proportion_who_move,
                mobility_data=mobility_data,
                mobility_data_indices=mobility_data_indices,
                mobility_row_indices=mobility_row_indices,
                population=population,
                param_expr_lookup=param_expr_lookup,
                param_name_to_row=param_name_to_row,
                single_prop_mask=single_prop_mask,
            )

            # 5) instantaneous flux and assembly: dy/dt
            amounts = compute_transition_amounts_meta(source_numbers, total_rates)
            dy = assemble_flux(amounts, transitions, C, N)
            return dy

        self._rhs = rhs
        return rhs

    def create_rhs(self) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
        """Return the currently built RHS (builds on demand if missing)."""
        if self._rhs is None:
            return self.build_rhs()
        return self._rhs

    def solve(
        self,
        y0: np.ndarray,
        parameters: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        **solve_ivp_kwargs,
    ):
        """Convenience wrapper around scipy.integrate.solve_ivp using the stored RHS."""
        self.reset_seeding_tracker()  # fresh run, re-apply day 0 seeding as needed
        rhs = self.create_rhs()
        return solve_ivp(
            fun=lambda t, y: rhs(t, y, parameters),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            **solve_ivp_kwargs,
        )
