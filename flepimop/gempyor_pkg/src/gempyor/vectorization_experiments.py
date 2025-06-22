# flepiMoP/flepimop/gempyor_pkg/src/gempyor/vectorization_experiments.py

import numpy as np

def compute_proportion_sums_exponents_vectorized(
    states_current,
    transitions,
    proportion_info,
    transition_sum_compartments,
    parameters,
    today
):
    n_transitions = transitions.shape[1]
    n_nodes = states_current.shape[1]
    n_proportions = proportion_info.shape[1]

    total_rates = np.ones((n_transitions, n_nodes))
    source_numbers = np.zeros((n_transitions, n_nodes))

    # NEW: Preallocate max possible proportion contribs to avoid dynamic lists
    max_props_per_trans = n_proportions  # safe upper bound
    proportion_contribs = np.zeros((max_props_per_trans, n_nodes))

    for t_idx in range(n_transitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]
        n_props = p_stop - p_start

        first_proportion = True

        for i, p_idx in enumerate(range(p_start, p_stop)):
            sum_start = proportion_info[0, p_idx]
            sum_stop = proportion_info[1, p_idx]
            exponent = parameters[proportion_info[2, p_idx], today]

            comp_indices = transition_sum_compartments[sum_start:sum_stop]
            # CHANGED: sum across compartments vectorized (no inner loop over compartments)
            summed = states_current[comp_indices, :].sum(axis=0)
            # CHANGED: exponentiation vectorized
            summed_exp = summed ** exponent

            if first_proportion:
                source_numbers[t_idx, :] = summed
                safe_source = np.where(summed > 0, summed, 1.0)
                # CHANGED: first proportion contribution vectorized
                proportion_contribs[i, :] = summed_exp / safe_source
                first_proportion = False
            else:
                # CHANGED: store contribution in preallocated array instead of appending list
                proportion_contribs[i, :] = summed_exp

        if n_props > 0:
            # CHANGED: use preallocated array + np.prod for contribution combination
            total_rates[t_idx, :] *= np.prod(proportion_contribs[:n_props, :], axis=0)

    return total_rates, source_numbers




def rhs_stage1_vectored(
    t,
    x,
    ncompartments,
    nspatial_nodes,
    ntransitions,
    transitions,
    proportion_info,
    transition_sum_compartments,
    parameters,
    today,
    method,
    dt,
    percent_day_away,
    proportion_who_move,
    mobility_data,
    mobility_data_indices,
    mobility_row_indices,
    population
):
    states_current = np.reshape(x, (2, ncompartments, nspatial_nodes))[0]
    transition_amounts = np.zeros((ntransitions, nspatial_nodes))

    if (x < 0).any():
        print("Integration error: rhs got a negative x (pos, time)", np.where(x < 0), t)

    # CHANGED: calls vectorized Stage 1 function instead of legacy inner loops
    total_rates, source_numbers = compute_proportion_sums_exponents_vectorized(
        states_current,
        transitions,
        proportion_info,
        transition_sum_compartments,
        parameters,
        today
    )

    # Note: rest of mobility + rate assembly is unchanged from legacy
    for t_idx in range(ntransitions):
        p_start = transitions[3, t_idx]
        p_stop = transitions[4, t_idx]

        if (p_stop - p_start) == 1:
            # CHANGED: total_rates already includes contribution; just apply rate
            total_rates[t_idx, :] *= parameters[transitions[2, t_idx], today]
        else:
            # legacy mobility logic remains
            for node in range(nspatial_nodes):
                prop_keep = 1 - percent_day_away * proportion_who_move[node]
                prop_change = (percent_day_away *
                               mobility_data[mobility_data_indices[node]:mobility_data_indices[node + 1]]
                               / population[node])

                rate_keep = (prop_keep *
                             source_numbers[t_idx, node] *
                             parameters[transitions[2, t_idx], today][node] /
                             population[node])

                visitors = mobility_row_indices[
                    mobility_data_indices[node]:mobility_data_indices[node + 1]
                ]
                rate_change = (prop_change *
                               source_numbers[t_idx, visitors] /
                               population[visitors] *
                               parameters[transitions[2, t_idx], today][visitors])

                total_rates[t_idx, node] *= (rate_keep + rate_change.sum())

    # method logic remains same, but applies to vectorized Stage 1 results
    for t_idx in range(ntransitions):
        if method == "rk4":
            number_move = source_numbers[t_idx, :] * total_rates[t_idx, :]
        elif method in ("euler", "stochastic"):
            compound_rate = 1 - np.exp(-dt * total_rates[t_idx, :])
            number_move = source_numbers[t_idx, :] * compound_rate
            if method == "stochastic":
                number_move = np.array([
                    np.random.binomial(int(source_numbers[t_idx, node]), compound_rate[node])
                    for node in range(nspatial_nodes)
                ])
        else:
            raise ValueError(f"Unknown method: {method}")

        transition_amounts[t_idx, :] = number_move

    return transition_amounts

