"""
Oversees use of the rk4 algorithim for simulation.
"""


import logging
import numpy as np
from numba import jit
import tqdm, scipy
from .utils import Timer

(
    transition_source_col,
    transition_destination_col,
    transition_rate_col,
    transition_proportion_start_col,
    transition_proportion_stop_col,
) = np.arange(5)

proportion_sum_starts_col = 0
proportion_sum_stops_col = 1
proportion_exponent_col = 2

float_tolerance = 1e-9


def rk4_integration(
    *,
    ncompartments,  # 1
    nspatial_nodes,  # 2
    ndays,  # 3
    parameters,  # 4
    dt,  # 5
    transitions,  # 6
    proportion_info,  # 7
    transition_sum_compartments,  # 8
    initial_conditions,  # 9
    seeding_data,  # 10
    seeding_amounts,  # 11
    mobility_data,  # 12
    mobility_row_indices,  # 13
    mobility_data_indices,  # 14
    population,  # 15
    method="rk4",
    silent=False
):
    states = np.zeros((ndays, ncompartments, nspatial_nodes))
    states_daily_incid = np.zeros((ndays, ncompartments, nspatial_nodes))
    states_current = np.zeros((ncompartments, nspatial_nodes))
    states_next = np.zeros((ncompartments, nspatial_nodes))

    ntransitions = transitions.shape[1]

    ## Setting values
    states_current = np.copy(initial_conditions)
    states_next = states_current.copy()

    proportion_who_move = np.zeros((nspatial_nodes))
    percent_day_away = 0.5
    for spatial_node in range(nspatial_nodes):
        proportion_who_move[spatial_node] = min(
            mobility_data[mobility_data_indices[spatial_node] : mobility_data_indices[spatial_node + 1]].sum()
            / population[spatial_node],
            1,
        )

    def rhs(t, x, today):
        states_current = np.reshape(x, (2, ncompartments, nspatial_nodes))[0]
        st_next = states_current.copy()  # this is used to make sure stochastic integration never goes below zero
        transition_amounts = np.zeros((ntransitions, nspatial_nodes))  # keep track of the transitions

        if (x < 0).any():
            print("Integration error: rhs got a negative x (pos, time)", np.where(x < 0), t)

        for transition_index in range(ntransitions):
            total_rate = np.ones((nspatial_nodes))
            first_proportion = True

            # Each transition may have several proportional_to factors
            for proportion_index in range(
                transitions[transition_proportion_start_col][transition_index],
                transitions[transition_proportion_stop_col][transition_index],
            ):
                # Compute the number of individuals in the compartments for this proportions
                relevant_number_in_comp = np.zeros((nspatial_nodes))
                relevant_exponent = np.ones((nspatial_nodes))
                for proportion_sum_index in range(
                    proportion_info[proportion_sum_starts_col][proportion_index],
                    proportion_info[proportion_sum_stops_col][proportion_index],
                ):
                    relevant_number_in_comp += states_current[transition_sum_compartments[proportion_sum_index]]

                # exponents should not be a proportion, since we don't sum them over sum compartments
                relevant_exponent = parameters[proportion_info[proportion_exponent_col][proportion_index]][today]

                # chadi: i believe what this mean that the first proportion is always the
                # source compartment. That's why there is nothing with n_spatial node here.
                # but (TODO) we should enforce that ?
                if first_proportion:
                    only_one_proportion = (
                        transitions[transition_proportion_start_col][transition_index] + 1
                    ) == transitions[transition_proportion_stop_col][transition_index]
                    first_proportion = False
                    source_number = relevant_number_in_comp  # does this mean we need the first to be "source" ??? yes !
                    if source_number.max() > 0:
                        total_rate[source_number > 0] *= (
                            source_number[source_number > 0] ** relevant_exponent[source_number > 0]
                            / source_number[source_number > 0]
                        )
                    if only_one_proportion:
                        total_rate *= parameters[transitions[transition_rate_col][transition_index]][today]
                else:
                    for spatial_node in range(nspatial_nodes):
                        proportion_keep_compartment = 1 - percent_day_away * proportion_who_move[spatial_node]
                        proportion_change_compartment = (
                            percent_day_away
                            * mobility_data[
                                mobility_data_indices[spatial_node] : mobility_data_indices[spatial_node + 1]
                            ]
                            / population[spatial_node]
                        )
                        rate_keep_compartment = (
                            proportion_keep_compartment
                            * relevant_number_in_comp[spatial_node] ** relevant_exponent[spatial_node]
                            / population[spatial_node]
                            * parameters[transitions[transition_rate_col][transition_index]][today][spatial_node]
                        )

                        visiting_subpop = mobility_row_indices[
                            mobility_data_indices[spatial_node] : mobility_data_indices[spatial_node + 1]
                        ]

                        rate_change_compartment = proportion_change_compartment * (
                            relevant_number_in_comp[visiting_subpop] ** relevant_exponent[visiting_subpop]
                        )
                        rate_change_compartment /= population[visiting_subpop]
                        rate_change_compartment *= parameters[transitions[transition_rate_col][transition_index]][
                            today
                        ][visiting_subpop]
                        total_rate[spatial_node] *= (rate_keep_compartment + rate_change_compartment.sum())

            # compute the number of individual transitioning from source to destination from the total rate
            # number_move has shape (nspatial_nodes)
            if method == "rk4":
                number_move = source_number * total_rate
            elif method == "euler" or method == "stochastic":
                compound_adjusted_rate = 1.0 - np.exp(-dt * total_rate)
                number_move = source_number * compound_adjusted_rate  ## to initialize typ
                if method == "stochastic":
                    for spatial_node in range(nspatial_nodes):
                        number_move[spatial_node] = np.random.binomial(
                            # number_move[spatial_node] = random.binomial(
                            source_number[spatial_node],
                            compound_adjusted_rate[spatial_node],
                        )
                elif method == "euler":
                    pass
                else:
                    raise ValueError(f"Did not understand method == {method}")

            transition_amounts[transition_index] = number_move

        return transition_amounts
        # for spatial_node in range(nspatial_nodes):
        #    if number_move[spatial_node] > states_current[transitions[transition_source_col][transition_index]][spatial_node]:
        #        number_move[spatial_node] = states_current[transitions[transition_source_col][transition_index]][spatial_node]

    if method == "rk4" or method == "euler":
        rhs = jit(nopython=True)(rhs)

    @jit(nopython=True)
    def update_states(states, delta_t, transition_amounts):
        states_diff = np.zeros((2, ncompartments, nspatial_nodes))  # first dim: 0 -> states_diff, 1: states_cum
        st_next = states.copy()
        st_next = np.reshape(st_next, (2, ncompartments, nspatial_nodes))
        if method == "rk4":
            # we move by delta_t * transitions, in case of rk4
            # when we use euler, the compound_adjusted_rate  already
            # includes the time step
            transition_amounts = transition_amounts.copy() * delta_t

        for transition_index in range(ntransitions):
            for spatial_node in range(nspatial_nodes):
                if transition_amounts[transition_index][spatial_node] < 0:
                    print(
                        "Integration error: transition amounts negative (trans_idx, node)",
                        transition_index,
                        spatial_node,
                    )
                if (
                    transition_amounts[transition_index][spatial_node]
                    >= st_next[0][transitions[transition_source_col][transition_index]][spatial_node] - float_tolerance
                ):
                    transition_amounts[transition_index][spatial_node] = max(
                        st_next[0][transitions[transition_source_col][transition_index]][spatial_node]
                        - float_tolerance,
                        0,
                    )
            st_next[0][transitions[transition_source_col][transition_index]] -= transition_amounts[transition_index]
            st_next[0][transitions[transition_destination_col][transition_index]] += transition_amounts[
                transition_index
            ]

            states_diff[0, transitions[transition_source_col][transition_index]] -= transition_amounts[transition_index]
            states_diff[0, transitions[transition_destination_col][transition_index]] += transition_amounts[
                transition_index
            ]
            states_diff[1, transitions[transition_destination_col][transition_index], :] += transition_amounts[
                transition_index
            ]  # Cumumlative

        return states + np.reshape(states_diff, states_diff.size)

    @jit(nopython=True, fastmath=True)
    def rk4_integrate(t, x, today):
        k1 = rhs(t, x, today)
        k2 = rhs(t + dt / 2, update_states(x, dt / 2, k1), today)
        k3 = rhs(t + dt / 2, update_states(x, dt / 2, k2), today)
        k4 = rhs(t + dt, update_states(x, dt, k3), today)
        return update_states(x, dt / 6, (k1 + 2 * k2 + 2 * k3 + k4))

    yesterday = -1
    times = np.arange(0, (ndays - 1) + 1e-7, dt)

    for time_index, time in tqdm.tqdm(enumerate(times), disable=silent):  # , total=len(times)
        today = int(np.floor(time))
        is_a_new_day = today != yesterday
        yesterday = today

        if is_a_new_day:
            # Prevalence is saved at the begining of the day, while incidence is during the day
            states[today, :, :] = states_next
            for seeding_instance_idx in range(
                seeding_data["day_start_idx"][today],
                seeding_data["day_start_idx"][min(today + int(np.ceil(dt)), len(seeding_data["day_start_idx"]) - 1)],
            ):
                this_seeding_amounts = seeding_amounts[seeding_instance_idx]
                seeding_subpops = seeding_data["seeding_subpops"][seeding_instance_idx]
                seeding_sources = seeding_data["seeding_sources"][seeding_instance_idx]
                seeding_destinations = seeding_data["seeding_destinations"][seeding_instance_idx]
                # this_seeding_amounts = this_seeding_amounts < states_next[seeding_sources] ?  this_seeding_amounts : states_next[seeding_instance_idx]
                states_next[seeding_sources][seeding_subpops] -= this_seeding_amounts
                states_next[seeding_sources][seeding_subpops] = states_next[seeding_sources][seeding_subpops] * (
                    states_next[seeding_sources][seeding_subpops] > 0
                )
                states_next[seeding_destinations][seeding_subpops] += this_seeding_amounts

                # ADD TO cumulative, this is debatable,
                states_daily_incid[today][seeding_destinations][seeding_subpops] += this_seeding_amounts

        x_ = np.zeros((2, ncompartments, nspatial_nodes))
        x_[0] = states_next
        x_ = np.reshape(x_, x_.size)
        if method == "rk4":
            sol = rk4_integrate(time, x_, today)
        elif method == "euler" or method == "stochastic":
            sol = update_states(x_, dt, rhs(time, x_, today))
        x_ = np.reshape(sol, (2, ncompartments, nspatial_nodes))
        states_daily_incid[today] += x_[1]
        states_next = x_[0]

    if dt == 2.0:  # smooth prevalence:
        states_f = scipy.interpolate.interp1d(
            np.arange(ndays, step=2),
            states[::2, :, :],
            axis=0,
            kind="linear",
            bounds_error=False,  # necessary in some case, for some ti and tf
            fill_value="extrapolate",  # necessary in some case, for some ti and tf
        )
        states = states_f(np.arange(ndays))

        # states_i_f = scipy.interpolate.interp1d(np.arange(ndays, step=2), states_daily_incid[::2, :, :], axis=0, kind="linear")
        # states_daily_incid = states_i_f(np.arange(ndays)) / 2
        ## error is smaller with this bellow, but there should be even smarter ways of doing this TODO
        states_daily_incid = states_daily_incid / 2
        states_daily_incid[1::2, :, :] = states_daily_incid[:-1:2, :, :]

    error = False
    ## Perform some checks:
    if np.isnan(states_daily_incid).any() or np.isnan(states).any():
        logging.critical("Integration error: NaN detected in epidemic integration result. Failing...")
        error = True
    if not (np.isfinite(states_daily_incid).all() and np.isfinite(states).all()):
        logging.critical("Integration error: Inf detected in epidemic integration result. Failing...")
        error = True
    if (states_daily_incid < 0).any() or (states < 0).any():
        logging.critical("Integration error: negative values detected in epidemic integration result. Failing...")
        # todo: this, but smart so it doesn't fail if empty array
        # print(
        #    f"STATES: NNZ:{states[states < 0].size}/{states.size}, max:{np.max(states[states < 0])}, min:{np.min(states[states < 0])}, mean:{np.mean(states[states < 0])} median:{np.median(states[states < 0])}"
        # )
        # print(
        #    f"STATES_incid: NNZ:{states_daily_incid[states_daily_incid < 0].size}/{states_daily_incid.size}, max:{np.max(states_daily_incid[states_daily_incid < 0])}, min:{np.min(states_daily_incid[states_daily_incid < 0])}, mean:#{np.mean(states_daily_incid[states_daily_incid < 0])} median:{np.median(states_daily_incid[states_daily_incid < 0])}"
        # )
        error = True
    if error:
        logging.critical("Saving run configuration due to integration error")
        import pickle

        with open("integration_dump.pkl", "wb") as fn_dump:
            pickle.dump(
                [
                    states,
                    states_daily_incid,
                    ncompartments,
                    nspatial_nodes,
                    ndays,
                    parameters,
                    dt,
                    transitions,
                    proportion_info,
                    transition_sum_compartments,
                    initial_conditions,
                    dict(seeding_data),
                    seeding_amounts,
                    mobility_data,
                    mobility_row_indices,
                    mobility_data_indices,
                    population,
                    method,
                ],
                fn_dump,
            )
        print(
            "load the name space with: \nwith open('integration_dump.pkl','rb') as fn_dump:\n    states, states_daily_incid, ncompartments, nspatial_nodes, ndays, parameters, dt, transitions, proportion_info,  transition_sum_compartments, initial_conditions, seeding_data, seeding_amounts, mobility_data, mobility_row_indices, mobility_data_indices, population, method = pickle.load(fn_dump)"
        )
        print("/!\\ Invalid integration, will cause problems for downstream users /!\\ ")
        # raise ValueError("Invalid Integration...")
    return states, states_daily_incid
