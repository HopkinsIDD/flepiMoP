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
    print(f"{prefix} {name}: shape={a.shape}, dtype={a.dtype}, min={amin}, max={amax}, nan={any_nan}, inf={any_inf}")

def _dbg_nonfinite(name: str, arr: np.ndarray, prefix: str = "[DBG]"):
    if not DEBUG:
        return
    mask = ~np.isfinite(arr)
    if mask.any():
        idx = np.argwhere(mask)
        sample = idx[:10]  # avoid huge dumps
        print(f"{prefix} NON-FINITE in {name}: count={mask.sum()}, sample_indices={sample.tolist()}")


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute multiplicative proportion terms and source sizes.

    Notes
    -----
    - Fixes bug: we write proportion_contribs at absolute row index `p_idx`,
      not a local counter, so the [p_start:p_stop] slice aligns correctly.
    - Exponents may be node-wise (vector) via `param_t[row_idx, :]`.
    """
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]
    n_props = proportion_info.shape[1]

    total_rates = np.ones((n_transitions, n_nodes))
    source_numbers = np.zeros((n_transitions, n_nodes))
    proportion_contribs = np.zeros((n_props, n_nodes))

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        n_p = p_stop - p_start
        first = True

        for p_idx in range(p_start, p_stop):
            sum_start = proportion_info[0, p_idx]
            sum_stop = proportion_info[1, p_idx]
            row_idx = proportion_info[2, p_idx]

            # sum of selected compartments across nodes
            summed = states_current[
                transition_sum_compartments[sum_start:sum_stop], :
            ].sum(axis=0)  # (N,)

            # node-wise exponent (vector) or scalar if constant across nodes
            expnt_vec = param_t[row_idx, :]  # (N,)

            # elementwise power
            summed_exp = summed ** expnt_vec  # (N,)

            if first:
                source_numbers[t_idx, :] = summed
                safe_src = np.where(summed > 0.0, summed, 1.0)
                proportion_contribs[p_idx, :] = summed_exp / safe_src
                first = False
            else:
                proportion_contribs[p_idx, :] = summed_exp

        if n_p > 0:
            total_rates[t_idx, :] *= prod_along_axis0(
                proportion_contribs[p_start:p_stop, :]
            )

    return total_rates, source_numbers


# === 2. Transition Rate Computation ===

def compute_transition_rates(
    total_rates_base: np.ndarray,  # (Tn, N)
    source_numbers: np.ndarray,  # (Tn, N)
    transitions: np.ndarray,  # (5, Tn)
    param_t: np.ndarray,  # (P, N)
    percent_day_away: float,
    proportion_who_move: np.ndarray,  # (N,)
    mobility_data: np.ndarray,  # CSR data
    mobility_data_indices: np.ndarray,  # CSR indptr (N+1,)
    mobility_row_indices: np.ndarray,  # CSR indices
    population: np.ndarray,  # (N,)
    param_expr_lookup: dict[int, str] | None = None,  # transition-param-index -> expr
    param_name_to_row: dict[str, int] | None = None,  # name -> row index
) -> np.ndarray:
    """
    Compute adjusted transition rates per node, accounting for parameter scaling and mobility mixing.

    - If a transition has exactly one proportion term (n_p == 1), we simply scale by the parameter vector.
    - Otherwise we compute keep/change mixing with mobility.
    """
    if DEBUG:
        _dbg_stats("compute_transition_rates.total_rates_base", total_rates_base)
        _dbg_stats("compute_transition_rates.source_numbers", source_numbers)
        _dbg_stats("compute_transition_rates.param_t", param_t)

    n_transitions, n_nodes = total_rates_base.shape
    total_rates = total_rates_base.copy()

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        param_idx = transitions[2, t_idx]
        n_p = p_stop - p_start

        # parameter vector for this transition (N,)
        if param_expr_lookup is None:
            param_vec = param_t[param_idx, :]  # (N,)
        else:
            if param_name_to_row is None:
                raise ValueError(
                    "param_name_to_row is required when param_expr_lookup is provided."
                )
            expr = param_expr_lookup[param_idx]
            param_vec = resolve_param_expr_from_slice(
                expr, param_t, param_name_to_row
            )  # (N,)

        if DEBUG:
            _dbg_stats(f"param_vec[t_idx={t_idx}]", param_vec)
            _dbg_nonfinite(f"param_vec[t_idx={t_idx}]", param_vec)

        if n_p == 1:
            # simple scaling by parameter
            if DEBUG:
                # Check before multiply
                if not np.all(np.isfinite(param_vec)):
                    bad = np.argwhere(~np.isfinite(param_vec))[:10]
                    print(f"[DBG] n_p==1: non-finite param_vec at indices {bad.tolist()} for transition {t_idx}")
            total_rates[t_idx, :] *= param_vec
            if DEBUG:
                _dbg_nonfinite(f"total_rates_after_scale[t_idx={t_idx}]", total_rates[t_idx, :])
        else:
            # mobility-aware mixing (per node)
            for node in range(n_nodes):
                pop_n = population[node]
                pop_n_safe = pop_n if pop_n > 0.0 else 1.0

                param_node = param_vec[node]
                source_node = source_numbers[t_idx, node]

                prop_keep = 1.0 - percent_day_away * proportion_who_move[node]
                v_slice = slice(
                    mobility_data_indices[node], mobility_data_indices[node + 1]
                )
                visitors = mobility_row_indices[v_slice]
                mobility_vals = mobility_data[v_slice]
                prop_change = (
                    percent_day_away * mobility_vals / pop_n_safe
                )  # normalized contribution

                # residents who stay
                rate_keep = (
                    (prop_keep * source_node * param_node / pop_n)
                    if (pop_n > 0.0 and param_node >= 0.0)
                    else 0.0
                )

                # visitors contribution
                rate_change = 0.0
                for i in range(visitors.size):
                    v = visitors[i]
                    pop_v = population[v]
                    pop_v_safe = pop_v if pop_v > 0.0 else 1.0
                    src_v = source_numbers[t_idx, v]
                    param_v = param_vec[v]
                    if pop_v > 0.0 and param_v >= 0.0:
                        rate_change += prop_change[i] * src_v * param_v / pop_v_safe

                total = rate_keep + rate_change
                if DEBUG and not np.isfinite(total):
                    print(
                        f"[DBG] non-finite total (t_idx={t_idx}, node={node}): "
                        f"rate_keep={rate_keep}, rate_change={rate_change}, "
                        f"source_node={source_node}, param_node={param_node}, pop_n={pop_n}, "
                        f"prop_keep={prop_keep}, sum(prop_change)={prop_change.sum() if prop_change.size else 0.0}"
                    )
                total_rates[t_idx, node] *= total if np.isfinite(total) else 0.0

        if DEBUG:
            _dbg_nonfinite(f"total_rates_row_end[t_idx={t_idx}]", total_rates[t_idx, :])

    if DEBUG:
        _dbg_stats("compute_transition_rates.total_rates(OUT)", total_rates)
        _dbg_nonfinite("compute_transition_rates.total_rates(OUT)", total_rates)
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


# === 6. RHS Builder (solve_ivp-compatible) ===

def build_rhs_for_solve_ivp(
    ncompartments: int,
    nspatial_nodes: int,
    transitions: np.ndarray,
    proportion_info: np.ndarray,
    transition_sum_compartments: np.ndarray,
    percent_day_away: float,
    proportion_who_move: np.ndarray,
    mobility_data: np.ndarray,
    mobility_data_indices: np.ndarray,
    mobility_row_indices: np.ndarray,
    population: np.ndarray,
    *,
    param_expr_lookup: dict[int, str] | None = None,
    param_name_to_row: dict[str, int] | None = None,
    param_time_mode: str = "linear",
) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
    """
    Construct an RHS suitable for SciPy's solve_ivp: f(t, y, parameters) -> dy/dt.

    'parameters' must be passed at call time (via solve_ivp's args or functools.partial),
    and have shape (P, T, N) or (P, T). Time is sliced internally using `param_time_mode`.
    """

    def rhs(t: float, y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        if DEBUG:
            print(f"[DBG] RHS call t={t}")
            _dbg_stats("RHS.y(in)", y)

        # Time-slice parameters to (P, N)
        param_t = param_slice(parameters, t, mode=param_time_mode)
        if DEBUG:
            _dbg_nonfinite("RHS.param_t", param_t)

        states_current = y.reshape((ncompartments, nspatial_nodes))
        if DEBUG:
            _dbg_stats("RHS.states_current", states_current)

        total_base, source_numbers = compute_proportion_sums_exponents(
            states_current,
            transitions,
            proportion_info,
            transition_sum_compartments,
            param_t,  # (P, N)
        )
        if DEBUG:
            _dbg_stats("RHS.total_base", total_base)
            _dbg_stats("RHS.source_numbers", source_numbers)
            _dbg_nonfinite("RHS.total_base", total_base)
            _dbg_nonfinite("RHS.source_numbers", source_numbers)

        total_rates = compute_transition_rates(
            total_base,
            source_numbers,
            transitions,
            param_t,
            percent_day_away,
            proportion_who_move,
            mobility_data,
            mobility_data_indices,
            mobility_row_indices,
            population,
            param_expr_lookup=param_expr_lookup,
            param_name_to_row=param_name_to_row,
        )

        if DEBUG:
            _dbg_stats("RHS.total_rates", total_rates)
            _dbg_nonfinite("RHS.total_rates", total_rates)

        # Instantaneous flux (dy/dt)
        amounts = compute_transition_amounts_meta(source_numbers, total_rates)
        if DEBUG:
            _dbg_stats("RHS.amounts", amounts)
            _dbg_nonfinite("RHS.amounts", amounts)

        dy = assemble_flux(amounts, transitions, ncompartments, nspatial_nodes)
        if DEBUG:
            _dbg_stats("RHS.dy(out)", dy)
            _dbg_nonfinite("RHS.dy(out)", dy)

        return dy

    return rhs


# === 7. Solver Step Functions (unchanged; for your custom runner) ===

@njit(inline="always", fastmath=True)
def update(y: np.ndarray, delta_t: float, dy: np.ndarray) -> np.ndarray:
    """
    y(t + dt) = y(t) + dt * dy
    """
    return y + delta_t * dy


def rk4_step(
    rhs_fn: Callable[[float, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    """
    Single RK4 step assuming rhs_fn returns dy/dt.
    """
    k1 = rhs_fn(t, y)
    k2 = rhs_fn(t + dt / 2, update(y, dt / 2, k1))
    k3 = rhs_fn(t + dt / 2, update(y, dt / 2, k2))
    k4 = rhs_fn(t + dt, update(y, dt, k3))
    return update(y, dt / 6, k1 + 2 * k2 + 2 * k3 + k4)


def euler_or_stochastic_step(
    rhs_fn: Callable[[float, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    """
    Forward Euler using dy/dt from rhs_fn.
    """
    return update(y, dt, rhs_fn(t, y))


# === 8. Optional: your existing custom fixed-grid solver (works by currying parameters) ===

def run_solver(
    rhs_fn: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_grid: np.ndarray,
    method: str = "rk4",
    record_daily: bool = False,
    ncompartments: int | None = None,
    nspatial_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run a simple fixed-grid solver. Provide rhs_fn(t, y) that returns dy/dt.
    To use with parameters, curry them: rhs_curried = functools.partial(rhs_with_params, parameters=theta)
    """
    state_shape = y0.shape
    y = y0.copy().ravel()
    n_steps = len(t_grid)
    states = np.zeros((n_steps, *state_shape))
    incid = np.zeros((n_steps, *state_shape)) if record_daily else None

    if DEBUG:
        print(f"[DBG] run_solver: start, method={method}, steps={n_steps}")
        _dbg_stats("run_solver.y0", y)
        _dbg_stats("run_solver.t_grid", t_grid)

    for i, t in enumerate(t_grid):
        states[i] = y.reshape(state_shape)

        if record_daily and method != "rk4":
            dy = rhs_fn(t, y).reshape(state_shape)
            incid[i] = np.maximum(dy, 0.0)

        if i < n_steps - 1:
            dt = t_grid[i + 1] - t
            if method == "rk4":
                y = rk4_step(rhs_fn, y, t, dt)
            elif method == "euler":
                y = euler_or_stochastic_step(rhs_fn, y, t, dt)
            else:
                raise ValueError(f"Unknown method: {method}")

            if DEBUG:
                if not np.all(np.isfinite(y)):
                    print(f"[DBG] run_solver: non-finite state after step i={i}, t={t}, dt={dt}")
                    _dbg_nonfinite("run_solver.y(after step)", y)
                    # Optional: early break to reduce log noise during debug
                    # break

    # Interpolate from coarse to daily resolution, if applicable
    if len(t_grid) >= 2 and (t_grid[1] - t_grid[0] == 2.0):
        interp = interp1d(
            t_grid[::2], states[::2], axis=0, kind="linear", fill_value="extrapolate"
        )
        states = interp(t_grid)
        if record_daily:
            incid /= 2
            incid[1::2] = incid[:-1:2]

    if DEBUG:
        _dbg_stats("run_solver.states(out)", states)
        if incid is not None:
            _dbg_stats("run_solver.incid(out)", incid)

    return states, incid


# === 9. Optional factory class (solve_ivp-compatible) ===

class RHSfactory:
    """
    Convenience wrapper. `precomputed` must provide the keys below; this class
    builds an rhs(t, y, parameters) suitable for solve_ivp.
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

    def __init__(
        self,
        precomputed: dict,
        param_expr_lookup: dict[int, str] | None = None,
        param_name_to_row: dict[str, int] | None = None,
        param_time_mode: str = "linear",
    ):
        # Validate required keys
        for k in self.REQUIRED_KEYS:
            if k not in precomputed:
                raise KeyError(f"precomputed is missing required key: '{k}'")
        self.precomputed = precomputed
        self.param_expr_lookup = param_expr_lookup
        self.param_name_to_row = param_name_to_row
        self.param_time_mode = param_time_mode

        # Build the RHS once (it closes over static structures)
        self._rhs = build_rhs_for_solve_ivp(
            precomputed["ncompartments"],
            precomputed["nspatial_nodes"],
            precomputed["transitions"],
            precomputed["proportion_info"],
            precomputed["transition_sum_compartments"],
            precomputed["percent_day_away"],
            precomputed["proportion_who_move"],
            precomputed["mobility_data"],
            precomputed["mobility_data_indices"],
            precomputed["mobility_row_indices"],
            precomputed["population"],
            param_expr_lookup=self.param_expr_lookup,
            param_name_to_row=self.param_name_to_row,
            param_time_mode=self.param_time_mode,
        )

    def create_rhs(self) -> Callable[[float, np.ndarray, np.ndarray], np.ndarray]:
        """
        Returns an rhs(t, y, parameters) function for solve_ivp.
        """
        return self._rhs
