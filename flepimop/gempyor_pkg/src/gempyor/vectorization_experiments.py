import numpy as np
from numba import njit, prange
from scipy.interpolate import interp1d
from typing import Callable, Tuple, Optional, Dict

# === Constants ===
FLOAT_TOLERANCE = 1e-9
_PARALLEL_THRESHOLD = 1e7

# === Helper for symbolic expression resolution ===


def resolve_parameter_expression(
    expr: str,
    param_lookup: dict[str, np.ndarray],
    today: int
) -> np.ndarray:
    """
    Resolve a symbolic parameter expression (e.g. 'beta * gamma') at a given time index.

    Args:
        expr: A string representing a parameter or a product of parameters (e.g., 'Ro', 'Ro * gamma').
        param_lookup: Dictionary mapping parameter names to arrays of shape
            (n_nodes,) or (n_nodes, n_times). Values must be NumPy arrays.
        today: Integer time index used to slice time-varying parameters.

    Returns:
        Array of shape (n_nodes,) with resolved parameter values evaluated at `today`.
    """
    if "*" in expr:
        terms = [s.strip() for s in expr.split("*")]
        result = param_lookup[terms[0]].copy()
        if result.ndim > 1:
            result = result[:, today]
        for term in terms[1:]:
            val = param_lookup[term]
            result *= val if val.ndim == 1 else val[:, today]
        return result
    else:
        val = param_lookup[expr]
        return val if val.ndim == 1 else val[:, today]



# === 1. Core Proportion Logic ===




@njit(fastmath=True)
def prod_along_axis0(arr_2d: np.ndarray) -> np.ndarray:
    """
    Computes the product along the first axis (axis=0) of a 2D array.

    Args:
        arr_2d: A 2D NumPy array of shape (n_rows, n_cols). Assumes numeric dtype.

    Returns:
        A 1D NumPy array of shape (n_cols,) where each element is the product over rows at that column.
    """
    n_cols = arr_2d.shape[1]
    result = np.ones(n_cols, dtype=arr_2d.dtype)
    for i in range(arr_2d.shape[0]):
        for j in range(n_cols):
            result[j] *= arr_2d[i, j]
    return result




@njit(fastmath=True)
def compute_proportion_sums_exponents(
    states_current: np.ndarray,
    transitions: np.ndarray,
    proportion_info: np.ndarray,
    transition_sum_compartments: np.ndarray,
    parameters: np.ndarray,
    today: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute total rates and source population sizes based on exponentiated proportion-based terms.

    Args:
        states_current: Array of shape (n_compartments, n_nodes) representing current state values.
        transitions: Array of shape (5, n_transitions). Used to index proportion info per transition.
        proportion_info: Array of shape (3, n_props). Each column specifies:
                         - row 0: start index for summing compartments
                         - row 1: stop index for summing compartments
                         - row 2: parameter index for exponentiation
        transition_sum_compartments: 1D array listing compartment indices used in summation windows.
        parameters: Array of shape (n_params, n_times, n_nodes) or (n_params, n_times).
                    Only parameters[*, today] is accessed.
        today: Integer index into the time axis.

    Returns:
        total_rates: Array of shape (n_transitions, n_nodes) with full proportion-modified rates.
        source_numbers: Array of shape (n_transitions, n_nodes) with raw source compartment sums.
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

        for i, p_idx in enumerate(range(p_start, p_stop)):
            sum_start = proportion_info[0, p_idx]
            sum_stop = proportion_info[1, p_idx]
            expnt = parameters[proportion_info[2, p_idx], today]
            summed = states_current[transition_sum_compartments[sum_start:sum_stop], :].sum(
                axis=0
            )
            summed_exp = summed**expnt

            if first:
                source_numbers[t_idx] = summed
                safe_src = np.where(summed > 0, summed, 1.0)
                proportion_contribs[i, :] = summed_exp / safe_src
                first = False
            else:
                proportion_contribs[i, :] = summed_exp

        if n_p > 0:
            total_rates[t_idx, :] *= prod_along_axis0(
                proportion_contribs[p_start:p_stop, :]
            )

    return total_rates, source_numbers


# === 2. Transition Rate Computation ===


def compute_transition_rates(
    total_rates_base: np.ndarray,
    source_numbers: np.ndarray,
    transitions: np.ndarray,
    parameters: np.ndarray,
    today: int,
    percent_day_away: float,
    proportion_who_move: np.ndarray,
    mobility_data: np.ndarray,
    mobility_data_indices: np.ndarray,
    mobility_row_indices: np.ndarray,
    population: np.ndarray,
    param_expr_lookup: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    """
    Compute adjusted transition rates per node, accounting for parameter values and mobility mixing.

    Args:
        total_rates_base: Array of shape (n_transitions, n_nodes) with base rates from proportion logic.
        source_numbers: Array of shape (n_transitions, n_nodes) with current source population sizes.
        transitions: Array of shape (5, n_transitions). Columns hold metadata including parameter index
                     and slice indices into the proportion list.
        parameters: Array of shape (n_params, n_times, n_nodes) or (n_params, n_times).
                    All parameter values are indexed at `today`.
        today: Integer time index into the parameter array.
        percent_day_away: Scalar in [0, 1] indicating proportion of the population that is mobile.
        proportion_who_move: Array of shape (n_nodes,) with per-node mobility fractions.
        mobility_data: Flattened array of mobility matrix values (CSR format).
        mobility_data_indices: CSR `indptr`, shape (n_nodes + 1,).
        mobility_row_indices: CSR `indices`, shape matching `mobility_data`.
        population: Array of shape (n_nodes,) with current population sizes per node.
        param_expr_lookup: Optional dictionary mapping parameter indices (from `transitions[2, :]`)
                           to symbolic parameter expressions (e.g., 'beta * gamma').

    Returns:
        total_rates: Array of shape (n_transitions, n_nodes) with adjusted rates after scaling
                     by parameters and mobility-weighted terms.
    """
    n_transitions, n_nodes = total_rates_base.shape
    total_rates = total_rates_base.copy()

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        param_idx = transitions[2, t_idx]

        if (p_stop - p_start) == 1:
            if param_expr_lookup is None:
                param_val = parameters[param_idx, today]
            else:
                param_val = resolve_parameter_expression(
                    param_expr_lookup[param_idx], parameters, today
                )
            total_rates[t_idx] *= param_val
        else:
            for node in range(n_nodes):
                pop_n = population[node]
                pop_n_safe = pop_n if pop_n > 0 else 1.0

                if param_expr_lookup is None:
                    param_node = parameters[param_idx, today][node]
                else:
                    param_node = resolve_parameter_expression(
                        param_expr_lookup[param_idx], parameters, today
                    )[node]

                source_node = source_numbers[t_idx, node]
                prop_keep = 1.0 - percent_day_away * proportion_who_move[node]
                visitors_idx = slice(
                    mobility_data_indices[node], mobility_data_indices[node + 1]
                )
                visitors = mobility_row_indices[visitors_idx]
                mobility_vals = mobility_data[visitors_idx]
                prop_change = percent_day_away * mobility_vals / pop_n_safe

                rate_keep = (
                    prop_keep * source_node * param_node / pop_n
                    if pop_n > 0 and param_node >= 0
                    else 0.0
                )
                rate_change = 0.0
                for i in range(visitors.size):
                    v = visitors[i]
                    pop_v = population[v]
                    pop_v_safe = pop_v if pop_v > 0 else 1.0
                    src_v = source_numbers[t_idx, v]
                    param_v = (
                        resolve_parameter_expression(
                            param_expr_lookup[param_idx], parameters, today
                        )[v]
                        if param_expr_lookup
                        else parameters[param_idx, today][v]
                    )
                    if pop_v > 0 and param_v >= 0:
                        rate_change += prop_change[i] * src_v * param_v / pop_v_safe

                total = rate_keep + rate_change
                total_rates[t_idx, node] *= total if np.isfinite(total) else 0.0

    return total_rates

# === 3. Binomial Stochastic (NumPy) ===


def compute_transition_amounts_numpy_binomial(
    source_numbers: np.ndarray,
    total_rates: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute stochastic transition amounts using binomial draws, assuming Poisson process approximation.

    Args:
        source_numbers: Array of shape (n_transitions, n_nodes), with integer-like source population counts.
        total_rates: Array of shape (n_transitions, n_nodes), with transition rates per unit time.
        dt: Time step size.

    Returns:
        Array of shape (n_transitions, n_nodes) with binomially sampled transition counts,
        cast to float32.
    """
    probs = 1.0 - np.exp(-dt * total_rates)
    draws = np.random.binomial(source_numbers.astype(np.int32), probs)
    return draws.astype(np.float32)



# === 4. Transition Amount Dispatch ===


@njit
def compute_transition_amounts_meta(
    source_numbers: np.ndarray,
    total_rates: np.ndarray,
    method: str,
    dt: float
) -> np.ndarray:
    """
    Dispatch to serial or parallel deterministic transition computation based on workload.

    This function does not support stochastic methods inside JIT and will raise an error if used.

    Args:
        source_numbers: Array of shape (n_transitions, n_nodes), source population counts.
        total_rates: Array of shape (n_transitions, n_nodes), transition rates per unit time.
        method: Integration method ('euler' or 'rk4'). Must not be 'stochastic'.
        dt: Time step size.

    Returns:
        Array of shape (n_transitions, n_nodes) with computed transition amounts.

    Raises:
        RuntimeError: If method is 'stochastic', which must be handled outside JIT.
    """
    workload = source_numbers.shape[0] * source_numbers.shape[1]
    if method == "stochastic":
        raise RuntimeError("Stochastic method must be handled outside JIT")
    elif workload >= _PARALLEL_THRESHOLD:
        return compute_transition_amounts_parallel(source_numbers, total_rates, dt)
    else:
        return compute_transition_amounts_serial(source_numbers, total_rates, method, dt)





@njit(fastmath=True)
def compute_transition_amounts_serial(
    source_numbers: np.ndarray,
    total_rates: np.ndarray,
    method: str,
    dt: float
) -> np.ndarray:
    """
    Compute deterministic transition amounts serially for each transition and node.

    Args:
        source_numbers: Array of shape (n_transitions, n_nodes), source population counts.
        total_rates: Array of shape (n_transitions, n_nodes), transition rates per unit time.
        method: Integration method; must be either 'rk4' or 'euler'.
        dt: Time step size.

    Returns:
        Array of shape (n_transitions, n_nodes) with computed transition amounts.
    """
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))
    for t_idx in range(n_transitions):
        for node in range(n_nodes):
            if method == "rk4":
                amounts[t_idx, node] = (
                    source_numbers[t_idx, node] * total_rates[t_idx, node]
                )
            elif method == "euler":
                rate = 1 - np.exp(-dt * total_rates[t_idx, node])
                amounts[t_idx, node] = source_numbers[t_idx, node] * rate
    return amounts



@njit(parallel=True, fastmath=True)
def compute_transition_amounts_parallel(
    source_numbers: np.ndarray,
    total_rates: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Compute deterministic transition amounts in parallel using Euler integration logic.

    This is used when the workload exceeds a defined threshold and parallelization
    is expected to provide a speedup.

    Args:
        source_numbers: Array of shape (n_transitions, n_nodes), source population counts.
        total_rates: Array of shape (n_transitions, n_nodes), transition rates per unit time.
        dt: Time step size.

    Returns:
        Array of shape (n_transitions, n_nodes) with computed transition amounts.
    """
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))
    for t_idx in prange(n_transitions):
        for node in range(n_nodes):
            rate = 1 - np.exp(-dt * total_rates[t_idx, node])
            amounts[t_idx, node] = source_numbers[t_idx, node] * rate
    return amounts



# === 5. Assemble Flux Vector ===


@njit(fastmath=True)
def assemble_flux(
    amounts: np.ndarray,
    transitions: np.ndarray,
    ncompartments: int,
    nspatial_nodes: int
) -> np.ndarray:
    """
    Assemble the full system flux vector from transition amounts.

    This function accumulates inflow and outflow for each compartment and spatial node,
    using the transition source and destination definitions.

    Args:
        amounts: Array of shape (n_transitions, nspatial_nodes), transition magnitudes per node.
        transitions: Array of shape (5, n_transitions); row 0 = source compartment index,
                     row 1 = destination compartment index.
        ncompartments: Total number of compartments in the model.
        nspatial_nodes: Total number of spatial nodes.

    Returns:
        A flattened array of shape (ncompartments * nspatial_nodes,) representing the time derivative dy/dt.
    """
    dy_dt = np.zeros((ncompartments, nspatial_nodes))
    for t_idx in range(amounts.shape[0]):
        src = transitions[0, t_idx]
        dst = transitions[1, t_idx]
        for node in range(nspatial_nodes):
            dy_dt[src, node] -= amounts[t_idx, node]
            dy_dt[dst, node] += amounts[t_idx, node]
    return dy_dt.ravel()



# === 6. RHS Builder ===


from typing import Callable, Optional, Dict

def build_rhs(
    ncompartments: int,
    nspatial_nodes: int,
    transitions: np.ndarray,
    proportion_info: np.ndarray,
    transition_sum_compartments: np.ndarray,
    parameters: np.ndarray,
    method: str,
    dt: float,
    percent_day_away: float,
    proportion_who_move: np.ndarray,
    mobility_data: np.ndarray,
    mobility_data_indices: np.ndarray,
    mobility_row_indices: np.ndarray,
    population: np.ndarray,
    param_expr_lookup: Optional[Dict[int, str]] = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """
    Construct the RHS function for the ODE/stepper solver using mobility-aware transition logic.

    Args:
        ncompartments: Total number of compartments in the model.
        nspatial_nodes: Total number of spatial nodes or subpopulations.
        transitions: Transition table of shape (5, n_transitions), used in all rate calculations.
        proportion_info: Array of shape (3, n_props), with proportion group structure.
        transition_sum_compartments: Flat array of compartment indices used for summation in proportions.
        parameters: Array of shape (n_params, n_times, n_nodes) or (n_params, n_times).
        method: String specifying the solver method ('rk4', 'euler', 'stochastic').
        dt: Time step size.
        percent_day_away: Fraction of population mixing away from home location.
        proportion_who_move: Array of shape (n_nodes,), giving mobility proportion by node.
        mobility_data: CSR data array with mobility weights.
        mobility_data_indices: CSR indptr array for per-node lookups.
        mobility_row_indices: CSR indices array giving visitor destinations.
        population: Array of shape (n_nodes,), population size at each node.
        param_expr_lookup: Optional mapping of parameter index (from `transitions`) to symbolic expressions.

    Returns:
        rhs: A function `rhs(t, y)` that returns a flat array of dy/dt at time `t` and state `y`.
    """

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        today = int(t)
        states_current = y.reshape((ncompartments, nspatial_nodes))

        total_base, source_numbers = compute_proportion_sums_exponents(
            states_current,
            transitions,
            proportion_info,
            transition_sum_compartments,
            parameters,
            today,
        )

        total_rates = compute_transition_rates(
            total_base,
            source_numbers,
            transitions,
            parameters,
            today,
            percent_day_away,
            proportion_who_move,
            mobility_data,
            mobility_data_indices,
            mobility_row_indices,
            population,
            param_expr_lookup=param_expr_lookup,
        )

        if method == "stochastic":
            amounts = compute_transition_amounts_numpy_binomial(
                source_numbers, total_rates, dt
            )
        else:
            amounts = compute_transition_amounts_meta(
                source_numbers, total_rates, method, dt
            )

        return assemble_flux(amounts, transitions, ncompartments, nspatial_nodes)

    return rhs



# === 7. Solver Step Functions ===


@njit(inline="always", fastmath=True)
def update(
    y: np.ndarray,
    delta_t: float,
    dy: np.ndarray
) -> np.ndarray:
    """
    Basic time update: y(t + dt) = y(t) + dt * dy.

    Args:
        y: Current state vector (flattened).
        delta_t: Time step size.
        dy: Time derivative of state, same shape as `y`.

    Returns:
        Updated state vector of same shape as `y`.
    """
    return y + delta_t * dy



from typing import Callable

def rk4_step(
    rhs_fn: Callable[[float, np.ndarray], np.ndarray],
    y: np.ndarray,
    t: float,
    dt: float
) -> np.ndarray:
    """
    Perform a single Runge-Kutta 4th order step.

    Args:
        rhs_fn: Right-hand side function of the form rhs(t, y).
        y: Current state vector (flattened).
        t: Current time.
        dt: Time step size.

    Returns:
        Updated state vector after one RK4 step.
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
    dt: float
) -> np.ndarray:
    """
    Perform a single step using forward Euler or stochastic update logic.

    Args:
        rhs_fn: Right-hand side function of the form rhs(t, y).
        y: Current state vector (flattened).
        t: Current time.
        dt: Time step size.

    Returns:
        Updated state vector after one forward Euler (or stochastic) step.
    """
    return update(y, dt, rhs_fn(t, y))



# === 8. Solver Interface ===




def run_solver(
    rhs_fn: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_grid: np.ndarray,
    method: str = "rk4",
    record_daily: bool = False,
    ncompartments: Optional[int] = None,
    nspatial_nodes: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run a numerical solver on a system with a flattened state vector using a fixed time grid.

    Args:
        rhs_fn: Right-hand side function returning dy/dt, of the form rhs(t, y).
        y0: Initial state array of shape (n_compartments, n_nodes).
        t_grid: Array of times at which to evaluate the solution.
        method: Solver stepping method: one of {'rk4', 'euler', 'stochastic'}.
        record_daily: If True, also record daily incidence (nonnegative flow).
        ncompartments: Optional override for number of compartments (usually inferred from `y0`).
        nspatial_nodes: Optional override for number of nodes (usually inferred from `y0`).

    Returns:
        states: Array of shape (n_steps, n_compartments, n_nodes), the simulated states.
        incid:  Array of same shape, recording daily incidence if `record_daily=True`; otherwise `None`.

    Notes:
        - State vector `y` is flattened internally and reshaped at each time step.
        - If `t_grid` is spaced every 2 days, linear interpolation fills in missing daily values.
    """
    state_shape = y0.shape
    y = y0.copy().ravel()
    n_steps = len(t_grid)
    states = np.zeros((n_steps, *state_shape))
    incid = np.zeros((n_steps, *state_shape)) if record_daily else None

    for i, t in enumerate(t_grid):
        today = int(np.floor(t))
        states[i] = y.reshape(state_shape)

        if record_daily and method != "rk4":
            dy = rhs_fn(t, y).reshape(state_shape)
            incid[i] = np.maximum(dy, 0.0)

        if i < n_steps - 1:
            dt = t_grid[i + 1] - t
            if method == "rk4":
                y = rk4_step(rhs_fn, y, t, dt)
            elif method in {"euler", "stochastic"}:
                y = euler_or_stochastic_step(rhs_fn, y, t, dt)
            else:
                raise ValueError(f"Unknown method: {method}")

    # Interpolate from coarse to daily resolution, if applicable
    if t_grid[1] - t_grid[0] == 2.0:
        interp = interp1d(
            t_grid[::2], states[::2], axis=0, kind="linear", fill_value="extrapolate"
        )
        states = interp(t_grid)
        if record_daily:
            incid /= 2
            incid[1::2] = incid[:-1:2]

    return states, incid

