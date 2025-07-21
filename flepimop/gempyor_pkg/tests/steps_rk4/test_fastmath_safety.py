import numpy as np
from numpy.testing import assert_allclose
from numba import njit, prange

# === Constants ===
FLOAT_RTol = 1e-6
FLOAT_ATol = 1e-6

# === Test Data Generation ===

def generate_test_data(C=5, G=10, T=10):
    np.random.seed(42)

    transitions = np.zeros((5, 6), dtype=np.int64)
    transitions[0] = np.random.randint(0, C, size=6)  # src
    transitions[1] = np.random.randint(0, C, size=6)  # dst
    transitions[2] = np.random.randint(0, 3, size=6)  # param index
    transitions[3] = np.arange(0, 12, 2)
    transitions[4] = np.arange(2, 14, 2)

    proportion_info = np.zeros((3, 12), dtype=np.int64)
    proportion_info[0] = 0
    proportion_info[1] = np.random.randint(1, 4, size=12)
    proportion_info[2] = np.random.randint(0, 3, size=12)

    transition_sum_compartments = np.arange(C, dtype=np.int64)
    parameters = np.abs(np.random.randn(3, T + 1))
    dt = 0.5
    today = 3

    states_current = np.abs(np.random.randn(C, G))
    percent_day_away = 0.2
    proportion_who_move = np.random.rand(G)
    mobility_data = np.random.rand(G * 2)
    mobility_data_indices = np.array([0, 2, 4, 6] + [G * 2] * (G - 3), dtype=np.int64)
    mobility_row_indices = np.random.randint(0, G, size=G * 2)
    population = np.random.randint(100, 1000, size=G).astype(np.float64)

    return dict(
        transitions=transitions,
        proportion_info=proportion_info,
        transition_sum_compartments=transition_sum_compartments,
        parameters=parameters,
        today=today,
        states_current=states_current,
        percent_day_away=percent_day_away,
        proportion_who_move=proportion_who_move,
        mobility_data=mobility_data,
        mobility_data_indices=mobility_data_indices,
        mobility_row_indices=mobility_row_indices,
        population=population,
        dt=dt,
        C=C,
        G=G
    )

# === 1. Proportion Summation ===

@njit(fastmath=True)
def prop_yes(*args):
    return prop_no.py_func(*args)

@njit(fastmath=False)
def prop_no(states_current, transitions, proportion_info, transition_sum_compartments, parameters, today):
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]
    n_props = proportion_info.shape[1]

    total_rates = np.ones((n_transitions, n_nodes))
    source_numbers = np.zeros((n_transitions, n_nodes))
    proportion_contribs = np.zeros((n_props, n_nodes))

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
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

        if (p_stop - p_start) > 0:
            total_rates[t_idx, :] *= np.prod(proportion_contribs[p_start:p_stop, :], axis=0)

    return total_rates, source_numbers


# === 2. Transition Rate ===

@njit(fastmath=True)
def transrate_yes(*args):
    return transrate_no.py_func(*args)

@njit(fastmath=False)
def transrate_no(total_rates_base, source_numbers, transitions, parameters, today,
                 percent_day_away, proportion_who_move,
                 mobility_data, mobility_data_indices,
                 mobility_row_indices, population):

    n_transitions, n_nodes = total_rates_base.shape
    total_rates = total_rates_base.copy()

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]

        if (p_stop - p_start) == 1:
            total_rates[t_idx] *= parameters[transitions[2, t_idx], today]
        else:
            for node in range(n_nodes):
                prop_keep = 1.0 - percent_day_away * proportion_who_move[node]
                visitors_idx = slice(mobility_data_indices[node], mobility_data_indices[node + 1])
                visitors = mobility_row_indices[visitors_idx]
                prop_change = percent_day_away * mobility_data[visitors_idx] / population[node]

                rate_keep = (
                    prop_keep * source_numbers[t_idx, node] *
                    parameters[transitions[2, t_idx], today][node] / population[node]
                )

                rate_change = 0.0
                for i in range(visitors.size):
                    v = visitors[i]
                    rate_change += (
                        prop_change[i] * source_numbers[t_idx, v] *
                        parameters[transitions[2, t_idx], today][v] / population[v]
                    )

                total_rates[t_idx, node] *= rate_keep + rate_change

    return total_rates


# === 3. Assemble Flux ===

@njit(fastmath=True)
def flux_yes(*args):
    return flux_no.py_func(*args)

@njit(fastmath=False)
def flux_no(amounts, transitions, ncompartments, nspatial_nodes):
    dy_dt = np.zeros((ncompartments, nspatial_nodes))
    for t_idx in range(amounts.shape[0]):
        src = transitions[0, t_idx]
        dst = transitions[1, t_idx]
        for node in range(nspatial_nodes):
            dy_dt[src, node] -= amounts[t_idx, node]
            dy_dt[dst, node] += amounts[t_idx, node]
    return dy_dt.ravel()


# === 4. Euler/Exponential Update ===

@njit(fastmath=True)
def euler_yes(rhs_fn, y, t, dt):
    return y + dt * rhs_fn(t, y)

@njit(fastmath=False)
def euler_no(rhs_fn, y, t, dt):
    return y + dt * rhs_fn(t, y)


# === Top-level test ===

def test_all_fastmath_blocks_consistent():
    data = generate_test_data()
    today = data["today"]

    # 1. Proportion exponentiation
    base_yes, src_yes = prop_yes(
        data["states_current"], data["transitions"], data["proportion_info"],
        data["transition_sum_compartments"], data["parameters"], today
    )
    base_no, src_no = prop_no(
        data["states_current"], data["transitions"], data["proportion_info"],
        data["transition_sum_compartments"], data["parameters"], today
    )
    assert_allclose(base_yes, base_no, rtol=FLOAT_RTol, atol=FLOAT_ATol)
    assert_allclose(src_yes, src_no, rtol=FLOAT_RTol, atol=FLOAT_ATol)

    # 2. Transition rate logic
    rates_yes = transrate_yes(
        base_yes, src_yes, data["transitions"], data["parameters"], today,
        data["percent_day_away"], data["proportion_who_move"],
        data["mobility_data"], data["mobility_data_indices"],
        data["mobility_row_indices"], data["population"]
    )
    rates_no = transrate_no(
        base_no, src_no, data["transitions"], data["parameters"], today,
        data["percent_day_away"], data["proportion_who_move"],
        data["mobility_data"], data["mobility_data_indices"],
        data["mobility_row_indices"], data["population"]
    )
    assert_allclose(rates_yes, rates_no, rtol=FLOAT_RTol, atol=FLOAT_ATol)

    # 3. Flux
    flux_out_yes = flux_yes(rates_yes, data["transitions"], data["C"], data["G"])
    flux_out_no = flux_no(rates_no, data["transitions"], data["C"], data["G"])
    assert_allclose(flux_out_yes, flux_out_no, rtol=FLOAT_RTol, atol=FLOAT_ATol)

    # 4. Euler step (wrapped)
    def dummy_rhs(t, y):
        return np.ones_like(y) * 0.01

    y0 = np.random.rand(data["C"] * data["G"])
    t, dt = 1.0, 0.1
    y1_yes = euler_yes(dummy_rhs, y0, t, dt)
    y1_no = euler_no(dummy_rhs, y0, t, dt)
    assert_allclose(y1_yes, y1_no, rtol=FLOAT_RTol, atol=FLOAT_ATol)
