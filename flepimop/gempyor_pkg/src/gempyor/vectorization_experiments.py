import numpy as np
from numba import njit, prange
from scipy.interpolate import interp1d

# === Constants ===
FLOAT_TOLERANCE = 1e-9
_PARALLEL_THRESHOLD = 1e7

# === Helper for symbolic expression resolution ===

def resolve_parameter_expression(expr, param_lookup, today):
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

@njit(fastmath=False)
def prod_along_axis0(arr_2d):
    n_cols = arr_2d.shape[1]
    result = np.ones(n_cols, dtype=arr_2d.dtype)
    for i in range(arr_2d.shape[0]):
        for j in range(n_cols):
            result[j] *= arr_2d[i, j]
    return result

@njit(fastmath=False)
def compute_proportion_sums_exponents(states_current, transitions, proportion_info,
                                      transition_sum_compartments, parameters, today):
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
            summed = states_current[transition_sum_compartments[sum_start:sum_stop], :].sum(axis=0)
            summed_exp = summed ** expnt

            if first:
                source_numbers[t_idx] = summed
                safe_src = np.where(summed > 0, summed, 1.0)
                proportion_contribs[i, :] = summed_exp / safe_src
                first = False
            else:
                proportion_contribs[i, :] = summed_exp

        if n_p > 0:
            total_rates[t_idx, :] *= prod_along_axis0(proportion_contribs[p_start:p_stop, :])

    return total_rates, source_numbers

# === 2. Transition Rate Computation ===

def compute_transition_rates(total_rates_base, source_numbers, transitions, parameters, today,
                             percent_day_away, proportion_who_move,
                             mobility_data, mobility_data_indices,
                             mobility_row_indices, population,
                             param_expr_lookup=None):
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
                param_val = resolve_parameter_expression(param_expr_lookup[param_idx], parameters, today)
            total_rates[t_idx] *= param_val
        else:
            for node in range(n_nodes):
                pop_n = population[node]
                pop_n_safe = pop_n if pop_n > 0 else 1.0

                if param_expr_lookup is None:
                    param_node = parameters[param_idx, today][node]
                else:
                    param_node = resolve_parameter_expression(param_expr_lookup[param_idx], parameters, today)[node]

                source_node = source_numbers[t_idx, node]
                prop_keep = 1.0 - percent_day_away * proportion_who_move[node]
                visitors_idx = slice(mobility_data_indices[node], mobility_data_indices[node + 1])
                visitors = mobility_row_indices[visitors_idx]
                mobility_vals = mobility_data[visitors_idx]
                prop_change = percent_day_away * mobility_vals / pop_n_safe

                rate_keep = prop_keep * source_node * param_node / pop_n if pop_n > 0 and param_node >= 0 else 0.0
                rate_change = 0.0
                for i in range(visitors.size):
                    v = visitors[i]
                    pop_v = population[v]
                    pop_v_safe = pop_v if pop_v > 0 else 1.0
                    src_v = source_numbers[t_idx, v]
                    param_v = resolve_parameter_expression(param_expr_lookup[param_idx], parameters, today)[v] if param_expr_lookup else parameters[param_idx, today][v]
                    if pop_v > 0 and param_v >= 0:
                        rate_change += prop_change[i] * src_v * param_v / pop_v_safe

                total = rate_keep + rate_change
                total_rates[t_idx, node] *= total if np.isfinite(total) else 0.0

    return total_rates

# === 3. Binomial Stochastic (NumPy) ===

def compute_transition_amounts_numpy_binomial(source_numbers, total_rates, dt):
    probs = 1.0 - np.exp(-dt * total_rates)
    draws = np.random.binomial(source_numbers.astype(np.int32), probs)
    return draws.astype(np.float32)

# === 4. Transition Amount Dispatch ===

@njit
def compute_transition_amounts_meta(source_numbers, total_rates, method, dt):
    workload = source_numbers.shape[0] * source_numbers.shape[1]
    if method == "stochastic":
        raise RuntimeError("Stochastic method must be handled outside JIT")
    elif workload >= _PARALLEL_THRESHOLD:
        return compute_transition_amounts_parallel(source_numbers, total_rates, dt)
    else:
        return compute_transition_amounts_serial(source_numbers, total_rates, method, dt)

@njit(fastmath=False)
def compute_transition_amounts_serial(source_numbers, total_rates, method, dt):
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))
    for t_idx in range(n_transitions):
        for node in range(n_nodes):
            if method == "rk4":
                amounts[t_idx, node] = source_numbers[t_idx, node] * total_rates[t_idx, node]
            elif method == "euler":
                rate = 1 - np.exp(-dt * total_rates[t_idx, node])
                amounts[t_idx, node] = source_numbers[t_idx, node] * rate
    return amounts

@njit(parallel=True, fastmath=False)
def compute_transition_amounts_parallel(source_numbers, total_rates, dt):
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))
    for t_idx in prange(n_transitions):
        for node in range(n_nodes):
            rate = 1 - np.exp(-dt * total_rates[t_idx, node])
            amounts[t_idx, node] = source_numbers[t_idx, node] * rate
    return amounts

# === 5. Assemble Flux Vector ===

@njit(fastmath=False)
def assemble_flux(amounts, transitions, ncompartments, nspatial_nodes):
    dy_dt = np.zeros((ncompartments, nspatial_nodes))
    for t_idx in range(amounts.shape[0]):
        src = transitions[0, t_idx]
        dst = transitions[1, t_idx]
        for node in range(nspatial_nodes):
            dy_dt[src, node] -= amounts[t_idx, node]
            dy_dt[dst, node] += amounts[t_idx, node]
    return dy_dt.ravel()

# === 6. RHS Builder ===

def build_rhs(ncompartments, nspatial_nodes, transitions, proportion_info,
              transition_sum_compartments, parameters, method, dt,
              percent_day_away, proportion_who_move,
              mobility_data, mobility_data_indices,
              mobility_row_indices, population,
              param_expr_lookup=None):
    def rhs(t, y):
        today = int(t)
        states_current = y.reshape((ncompartments, nspatial_nodes))

        total_base, source_numbers = compute_proportion_sums_exponents(
            states_current, transitions, proportion_info,
            transition_sum_compartments, parameters, today
        )

        total_rates = compute_transition_rates(
            total_base, source_numbers, transitions, parameters, today,
            percent_day_away, proportion_who_move,
            mobility_data, mobility_data_indices,
            mobility_row_indices, population,
            param_expr_lookup=param_expr_lookup
        )

        if method == "stochastic":
            amounts = compute_transition_amounts_numpy_binomial(source_numbers, total_rates, dt)
        else:
            amounts = compute_transition_amounts_meta(source_numbers, total_rates, method, dt)

        return assemble_flux(amounts, transitions, ncompartments, nspatial_nodes)

    return rhs

# === 7. Solver Step Functions ===

@njit(inline='always', fastmath=False)
def update(y, delta_t, dy):
    return y + delta_t * dy

def rk4_step(rhs_fn, y, t, dt):
    k1 = rhs_fn(t, y)
    k2 = rhs_fn(t + dt / 2, update(y, dt / 2, k1))
    k3 = rhs_fn(t + dt / 2, update(y, dt / 2, k2))
    k4 = rhs_fn(t + dt, update(y, dt, k3))
    return update(y, dt / 6, k1 + 2 * k2 + 2 * k3 + k4)

def euler_or_stochastic_step(rhs_fn, y, t, dt):
    return update(y, dt, rhs_fn(t, y))

# === 8. Solver Interface ===

def run_solver(rhs_fn, y0, t_grid, method="rk4",
               record_daily=False, ncompartments=None, nspatial_nodes=None):
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

    if t_grid[1] - t_grid[0] == 2.0:
        interp = interp1d(t_grid[::2], states[::2], axis=0, kind="linear", fill_value="extrapolate")
        states = interp(t_grid)
        if record_daily:
            incid /= 2
            incid[1::2] = incid[:-1:2]

    return states, incid
