import numpy as np

# === Constants ===
FLOAT_TOLERANCE = 1e-9

# === Core helpers ===

def compute_proportion_sums_exponents(
    states_current, transitions, proportion_info,
    transition_sum_compartments, parameters, today
):
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]
    n_props = proportion_info.shape[1]

    total_rates = np.ones((n_transitions, n_nodes))
    source_numbers = np.zeros((n_transitions, n_nodes))

    proportion_contribs = np.zeros((n_props, n_nodes))  # upper bound

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        n_p = p_stop - p_start

        first = True
        for i, p_idx in enumerate(range(p_start, p_stop)):
            sum_start = proportion_info[0, p_idx]
            sum_stop = proportion_info[1, p_idx]
            expnt = parameters[proportion_info[2, p_idx], today]

            comps = transition_sum_compartments[sum_start:sum_stop]
            summed = states_current[comps, :].sum(axis=0)
            summed_exp = summed ** expnt

            if first:
                source_numbers[t_idx] = summed
                safe_src = np.where(summed > 0, summed, 1.0)
                proportion_contribs[i, :] = summed_exp / safe_src
                first = False
            else:
                proportion_contribs[i, :] = summed_exp

        if n_p > 0:
            total_rates[t_idx, :] *= np.prod(proportion_contribs[p_start:p_stop, :], axis=0)

    return total_rates, source_numbers


def compute_transition_rates(
    total_rates_base, source_numbers, transitions, parameters, today,
    percent_day_away, proportion_who_move, mobility_data,
    mobility_data_indices, mobility_row_indices, population
):
    n_transitions, n_nodes = total_rates_base.shape
    total_rates = total_rates_base.copy()

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]

        if (p_stop - p_start) == 1:
            total_rates[t_idx] *= parameters[transitions[2, t_idx], today]
        else:
            for node in range(n_nodes):
                prop_keep = 1 - percent_day_away * proportion_who_move[node]
                visitors_idx = slice(
                    mobility_data_indices[node],
                    mobility_data_indices[node + 1]
                )
                visitors = mobility_row_indices[visitors_idx]
                prop_change = (percent_day_away *
                               mobility_data[visitors_idx] / population[node])

                rate_keep = (prop_keep *
                             source_numbers[t_idx, node] *
                             parameters[transitions[2, t_idx], today][node] /
                             population[node])

                rate_change = (prop_change *
                               source_numbers[t_idx, visitors] /
                               population[visitors] *
                               parameters[transitions[2, t_idx], today][visitors])

                total_rates[t_idx, node] *= rate_keep + rate_change.sum()

    return total_rates


def compute_transition_amounts(source_numbers, total_rates, method, dt):
    n_transitions, n_nodes = total_rates.shape
    amounts = np.zeros((n_transitions, n_nodes))

    for t_idx in range(n_transitions):
        if method == "rk4":
            amounts[t_idx] = source_numbers[t_idx] * total_rates[t_idx]
        elif method == "euler":
            rate = 1 - np.exp(-dt * total_rates[t_idx])
            amounts[t_idx] = source_numbers[t_idx] * rate
        elif method == "stochastic":
            rate = 1 - np.exp(-dt * total_rates[t_idx])
            amounts[t_idx] = np.array([
                np.random.binomial(int(source_numbers[t_idx, node]), rate[node])
                for node in range(n_nodes)
            ])
        else:
            raise ValueError(f"Unknown method: {method}")

    return amounts


def assemble_flux(amounts, transitions, ncompartments, nspatial_nodes):
    dy_dt = np.zeros((ncompartments, nspatial_nodes))

    for t_idx in range(amounts.shape[0]):
        src = transitions[0, t_idx]
        dst = transitions[1, t_idx]
        dy_dt[src] -= amounts[t_idx]
        dy_dt[dst] += amounts[t_idx]

    return dy_dt.flatten()

# === RHS builder ===

def build_rhs(
    ncompartments, nspatial_nodes, transitions, proportion_info,
    transition_sum_compartments, parameters, method, dt,
    percent_day_away, proportion_who_move,
    mobility_data, mobility_data_indices,
    mobility_row_indices, population
):
    def rhs(t, y):
        today = int(np.floor(t))
        states_current = y.reshape((ncompartments, nspatial_nodes))

        total_base, source_numbers = compute_proportion_sums_exponents(
            states_current, transitions, proportion_info,
            transition_sum_compartments, parameters, today
        )

        total_rates = compute_transition_rates(
            total_base, source_numbers, transitions, parameters, today,
            percent_day_away, proportion_who_move,
            mobility_data, mobility_data_indices,
            mobility_row_indices, population
        )

        amounts = compute_transition_amounts(
            source_numbers, total_rates, method, dt
        )

        dy_dt = assemble_flux(amounts, transitions, ncompartments, nspatial_nodes)
        return dy_dt

    return rhs
