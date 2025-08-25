# tests/vectorized_outcomes/test_vectorized_outcomes.py
import numpy as np
import pandas as pd
import pytest

from gempyor.vectorized_outcomes import VectorizedOutcomes


# -------------------------------
# Fixtures: tiny deterministic toy
# -------------------------------


@pytest.fixture
def toy_compartments():
    """
    Build a tiny compartments table with two ages, two vax strata,
    and stages S, I1, I2. Vaccination is encoded inside infection_stage,
    e.g., 'I1_v0'. VectorizedOutcomes will auto-derive a 'vaccination_stage'
    column from this (or fill 'NA' if absent).
    """
    rows = []
    ages = ["A", "B"]
    vax = ["v0", "v1"]
    stages = ["S", "I1", "I2"]
    for age in ages:
        for vx in vax:
            for st in stages:
                rows.append(
                    {
                        "infection_stage": f"{st}_{vx}",
                        "age_strata": age,
                    }
                )
    comp = pd.DataFrame(rows)
    return comp


@pytest.fixture
def toy_states_constant(toy_compartments):
    """
    states: (T=6, C, N=2) with constant S and I per age across time,
    making closed-form checks trivial.
    - For age A: S_total=100 per node, I_total=10 per node
    - For age B: S_total=200 per node, I_total=5  per node
    Two vax rows per stage/age, so we split them evenly.
    """
    T = 6
    N = 2
    C = len(toy_compartments)

    S_A_each_v = 50.0
    S_B_each_v = 100.0
    I_A_each_v = 5.0  # I1 only (I2=0)
    I_B_each_v = 2.5

    states = np.zeros((T, C, N), dtype=float)

    # locate row indices
    s_idx = toy_compartments["infection_stage"].str.startswith("S").to_numpy()
    i1_idx = toy_compartments["infection_stage"].str.startswith("I1").to_numpy()
    # i2_idx = toy_compartments["infection_stage"].str.startswith("I2").to_numpy()  # unused

    age = toy_compartments["age_strata"].to_numpy()

    # fill S
    for a in ["A", "B"]:
        mask_a = age == a
        # per-vax rows for S
        mask_s_a = s_idx & mask_a
        each = S_A_each_v if a == "A" else S_B_each_v
        states[:, mask_s_a, :] = each  # constant over time, both nodes

    # fill I1 (I2 left at 0)
    for a in ["A", "B"]:
        mask_a = age == a
        mask_i1_a = i1_idx & mask_a
        each = I_A_each_v if a == "A" else I_B_each_v
        states[:, mask_i1_a, :] = each

    return states


@pytest.fixture
def toy_params_broadcast():
    """
    Parameter names and arrays (P, T, N) or (P, T) for broadcasting tests.
    Here we give 3 rows: 'r0', 'gamma', 'theta1' as time-constant scalars.
    """
    T = 6
    names = np.array(["r0", "gamma", "theta1"], dtype=str)
    # rate = r0 * gamma * theta1 = 2.0 * 0.5 * 1.0 = 1.0
    vals = np.array([2.0, 0.5, 1.0], dtype=float)[:, None]  # (P, 1)
    params = np.repeat(vals, T, axis=1)  # (P, T)
    return names, params


@pytest.fixture
def toy_params_age_specific():
    """
    Age-specific parameters for r0 and gamma, global theta1.
    Names encode age tokens 'A'/'B' so VectorizedOutcomes can align them.
    """
    T = 6
    names = np.array(["r0_A", "r0_B", "gamma_A", "gamma_B", "theta1"], dtype=str)
    r0_A = np.full(T, 2.0)
    r0_B = np.full(T, 3.0)
    g_A = np.full(T, 0.5)
    g_B = np.full(T, 0.5)
    th1 = np.full(T, 1.0)
    params = np.stack([r0_A, r0_B, g_A, g_B, th1], axis=0)  # (P=5, T)
    return names, params


@pytest.fixture
def times_numeric():
    """Strictly increasing numeric times (days since t0)."""
    # T=6 -> 0,1,2,3,4,5
    return np.arange(6, dtype=float)


@pytest.fixture
def population_two_nodes():
    """Population for two locations."""
    return np.array([1000.0, 2000.0], dtype=float)


# -------------------------------
# Helpers for label indexing
# -------------------------------


def _age_index(labels, age_value):
    for i, d in enumerate(labels):
        if d.get("age_strata") == age_value:
            return i
    raise AssertionError(f"Age {age_value} not in labels: {labels}")


# -------------------------------
# Core incidence (numeric times)
# -------------------------------


def test_incidence_numeric_no_agg(
    toy_states_constant,
    times_numeric,
    toy_compartments,
    toy_params_broadcast,
    population_two_nodes,
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    states = toy_states_constant
    T, C, N = states.shape

    # Compute aggregated=None => (Ages, K=T-1, N) left-edge intervals
    inc, labels, idx = X.incidence_by_location_agg(
        states=states,
        times=times_numeric,
        compartments=toy_compartments,
        param_names=names,
        params=params,
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=None,
    )
    # Two ages "A","B"
    assert inc.shape == (2, T - 1, N)
    assert len(labels) == 2
    assert len(idx) == T - 1

    iA = _age_index(labels, "A")
    iB = _age_index(labels, "B")

    # With our toy: rate = 1.0; S_A=100, I_A=10; S_B=200, I_B=5
    # per-interval counts (dt=1): A: 100*10/Pop, B: 200*5/Pop
    expected_A = (100.0 * 10.0) / population_two_nodes
    expected_B = (200.0 * 5.0) / population_two_nodes

    # All intervals identical due to constants
    assert np.allclose(inc[iA], expected_A[None, :])
    assert np.allclose(inc[iB], expected_B[None, :])


def test_incidence_population_toggle(
    toy_states_constant,
    times_numeric,
    toy_compartments,
    toy_params_broadcast,
    population_two_nodes,
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    states = toy_states_constant

    inc_norm, labels, _ = X.incidence_by_location_agg(
        states=states,
        times=times_numeric,
        compartments=toy_compartments,
        param_names=names,
        params=params,
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=None,
    )
    inc_raw, _, _ = X.incidence_by_location_agg(
        states=states,
        times=times_numeric,
        compartments=toy_compartments,
        param_names=names,
        params=params,
        population=population_two_nodes,
        normalize_by_population=False,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=None,
    )
    iA = _age_index(labels, "A")
    iB = _age_index(labels, "B")
    # Ratio between raw and normalized should be population
    ratio_A = inc_raw[iA] / inc_norm[iA]
    ratio_B = inc_raw[iB] / inc_norm[iB]
    assert np.allclose(ratio_A, population_two_nodes[None, :])
    assert np.allclose(ratio_B, population_two_nodes[None, :])


def test_rate_prefix_age_specific_mapping(
    toy_states_constant,
    times_numeric,
    toy_compartments,
    toy_params_age_specific,
    population_two_nodes,
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_age_specific
    states = toy_states_constant

    inc, labels, _ = X.incidence_by_location_agg(
        states=states,
        times=times_numeric,
        compartments=toy_compartments,
        param_names=names,
        params=params,
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=None,
    )

    iA = _age_index(labels, "A")
    iB = _age_index(labels, "B")
    # Here: rate_A = 2.0*0.5*1.0 = 1.0, rate_B = 3.0*0.5*1.0 = 1.5
    ratio = inc[iB] / inc[iA]
    assert np.allclose(ratio, 1.5)


def test_numeric_window_aggregation(
    toy_states_constant, toy_compartments, toy_params_broadcast, population_two_nodes
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    # Use half-day spacing to test window aggregation to 1.0 day
    times = np.arange(0.0, 6.0, 0.5)  # T=12
    # Repeat states along time to fit T
    states = np.repeat(toy_states_constant[:1], len(times), axis=0)

    inc_day, labels, periods = X.incidence_by_location_agg(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=np.repeat(params[:, :1], len(times), axis=1),
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=1.0,  # numeric window=1.0 (same units as `times`)
    )
    # Should produce roughly len(times)-1 intervals aggregated into ~6 daily windows
    assert inc_day.shape[1] == 6  # 0-1,1-2,...,5-6

    inc_raw, _, _ = X.incidence_by_location_agg(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=np.repeat(params[:, :1], len(times), axis=1),
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=None,
    )
    # Sum conservation across windows
    assert np.allclose(inc_day.sum(axis=1), inc_raw.sum(axis=1))


# --------------------------------
# Incidence (datetime & epiweeks)
# --------------------------------


def test_incidence_datetime_weekly_epiweeks(
    toy_states_constant, toy_compartments, toy_params_broadcast, population_two_nodes
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    # Build 14 daily timestamps for T=14
    times = pd.date_range("2025-01-01", periods=14, freq="D")
    # Expand states/params to T=14
    states = np.repeat(toy_states_constant[:1], len(times), axis=0)
    params14 = np.repeat(params[:, :1], len(times), axis=1)

    inc_w, labels, periods = X.incidence_by_location_agg(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=params14,
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate="W-SAT",  # epiweeks anchored on Saturday
    )
    # 14 days → 2 epiweekly bins (sometimes 3 depending on anchor/day-of-week)
    assert inc_w.shape[1] in (2, 3)

    inc_raw, _, _ = X.incidence_by_location_agg(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=params14,
        population=population_two_nodes,
        normalize_by_population=True,
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
        aggregate=None,
    )
    assert np.allclose(inc_w.sum(axis=1), inc_raw.sum(axis=1))


# ---------------------------------------
# Hospitalizations: constant / exp / gamma
# ---------------------------------------


def _spike_states_for_day(compartments, day_idx: int, T: int = 7, N: int = 1):
    """
    Build states such that infections spike on the interval [day_idx, day_idx+1]
    for both ages (S constant, I non-zero only at that left edge).
    """
    C = len(compartments)
    states = np.zeros((T, C, N), dtype=float)

    # S rows: set to 100 per age (split across two vax rows -> each 50)
    s_mask = compartments["infection_stage"].str.startswith("S").to_numpy()
    age = compartments["age_strata"].to_numpy()
    for a in ["A", "B"]:
        mask = s_mask & (age == a)
        states[:, mask, :] = 50.0  # each of the two vax rows → total S per age = 100

    # I1 rows: set to 10 per age only at left edge time=day_idx
    i_mask = compartments["infection_stage"].str.startswith("I1").to_numpy()
    for a in ["A", "B"]:
        mask = i_mask & (age == a)
        states[:, mask, :] = 0.0
        states[day_idx, mask, :] = (
            5.0  # two vax rows → total I per age at t=day_idx is 10
        )

    return states


def test_hospitalizations_constant_delay(
    toy_compartments, toy_params_broadcast, population_two_nodes
):
    X = VectorizedOutcomes(use_fft_convolution=False)

    names, params = toy_params_broadcast
    T = 7
    times = pd.date_range("2025-01-01", periods=T, freq="D")
    # make spike at day index 2
    states = _spike_states_for_day(toy_compartments, day_idx=2, T=T, N=2)
    prob_by_age = [0.1, 0.2]  # age specific

    hosp_ATN, labels, idx = X.hospitalizations_by_location(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=np.repeat(params[:, :1], T, axis=1),  # time-constant
        population=population_two_nodes,
        prob_spec=prob_by_age,
        delay_type="constant",
        delay_mean=1.0,  # deterministic 1-day shift
        aggregate="D",
        group_keys=("age_strata",),  # compute by age
        return_group_keys=("age_strata",),  # return by age
    )
    iA = _age_index(labels, "A")
    iB = _age_index(labels, "B")

    # Expect a single-day spike at day_idx+1, magnitude = infections(day_idx)*p_age
    # Infections = S*I/pop*rate*dt; here S=100, I=10, rate=1, dt=1
    expected_inf = np.array(
        [100.0 * 10.0 / population_two_nodes[0], 100.0 * 10.0 / population_two_nodes[1]]
    )
    # Spike should appear at index corresponding to (day_idx+1) in daily bins
    spike_pos = 3  # interval [2,3) with +1-day delay → bucket for day 3 (0-based)
    assert np.allclose(hosp_ATN[iA, spike_pos, :], 0.1 * expected_inf)  # age A
    assert np.allclose(hosp_ATN[iB, spike_pos, :], 0.2 * expected_inf)  # age B
    # All other days near zero
    mask_other = np.ones(hosp_ATN.shape[1], dtype=bool)
    mask_other[spike_pos] = False
    assert np.allclose(hosp_ATN[:, mask_other, :], 0.0)


def test_hospitalizations_mass_conservation_gamma_exp(
    toy_compartments, toy_params_broadcast, population_two_nodes
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast

    # make the horizon long so the convolution tail isn't cropped
    T = 128
    times = pd.date_range("2025-01-01", periods=T, freq="D")

    # spike at day index 5 (well before the end now)
    states = _spike_states_for_day(toy_compartments, day_idx=5, T=T, N=1)
    prob_by_age = [0.3, 0.4]

    params_T = np.repeat(params[:, :1], T, axis=1)

    # gamma kernel
    hosp_gamma, labels, idx = X.hospitalizations_by_location(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=params_T,
        population=np.array([1000.0]),
        prob_spec=prob_by_age,
        delay_type="gamma",
        delay_mean=7.0,
        delay_cv=0.5,
        delay_keep_mass=0.9999,
        aggregate="D",
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
    )
    # exponential kernel
    hosp_exp, _, _ = X.hospitalizations_by_location(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=params_T,
        population=np.array([1000.0]),
        prob_spec=prob_by_age,
        delay_type="exponential",
        delay_mean=7.0,
        delay_keep_mass=0.9999,
        aggregate="D",
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
    )

    iA = _age_index(labels, "A")
    iB = _age_index(labels, "B")

    # infections per age at spike = S*I/pop = 100*10/1000 = 1.0
    expected_tot_A = 1.0 * 0.3
    expected_tot_B = 1.0 * 0.4

    # Totals should be ~exact (up to keep_mass numeric error)
    assert abs(hosp_gamma[iA].sum() - expected_tot_A) < 1e-3
    assert abs(hosp_gamma[iB].sum() - expected_tot_B) < 1e-3
    assert abs(hosp_exp[iA].sum() - expected_tot_A) < 1e-3
    assert abs(hosp_exp[iB].sum() - expected_tot_B) < 1e-3


# -------------------------
# Deaths: API + mass checks
# -------------------------


def test_deaths_shapes_and_aggregation(
    toy_states_constant, toy_compartments, toy_params_broadcast, population_two_nodes
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    times = pd.date_range("2025-02-01", periods=14, freq="D")
    states = np.repeat(toy_states_constant[:1], len(times), axis=0)
    death_prob = [0.001, 0.01]  # IFR for ages A,B

    deaths_w, labels, periods = X.deaths_by_location(
        states=states,
        times=times,
        compartments=toy_compartments,
        param_names=names,
        params=np.repeat(params[:, :1], len(times), axis=1),
        population=population_two_nodes,
        prob_spec=death_prob,
        delay_type="gamma",
        delay_mean=10.0,
        delay_cv=0.6,
        aggregate="W-SAT",
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
    )
    assert deaths_w.ndim == 3
    assert deaths_w.shape[0] == 2  # ages
    assert deaths_w.shape[2] == 2  # locations
    assert len(periods) in (2, 3)  # ~2 epiweeks in 14d
    assert (deaths_w >= 0).all()


# --------------------------------
# Convolution path: FFT vs direct
# --------------------------------


def test_fft_vs_direct_convolution_equivalence(toy_compartments, toy_params_broadcast):
    # Long series to trigger FFT path
    T = 4096
    times = pd.date_range("2025-01-01", periods=T, freq="D")
    # Build a simple small random incidence via states (single node)
    rng = np.random.default_rng(123)
    comp = toy_compartments
    N = 1
    C = len(comp)
    states = np.zeros((T, C, N), dtype=float)
    # Put all mass in S and small random in I1 (both ages)
    s_mask = comp["infection_stage"].str.startswith("S").to_numpy()
    i_mask = comp["infection_stage"].str.startswith("I1").to_numpy()
    states[:, s_mask, :] = 50.0
    states[:, i_mask, :] = rng.uniform(0, 1, size=(T, i_mask.sum(), N))

    names, params = toy_params_broadcast
    params_T = np.repeat(params[:, :1], T, axis=1)

    pop = np.array([1000.0])

    X_direct = VectorizedOutcomes(use_fft_convolution=False)
    hosp_direct, labels, idx = X_direct.hospitalizations_by_location(
        states=states,
        times=times,
        compartments=comp,
        param_names=names,
        params=params_T,
        population=pop,
        prob_spec=[0.1, 0.1],
        delay_type="gamma",
        delay_mean=7.0,
        delay_cv=0.5,
        aggregate="D",
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
    )
    X_fft = VectorizedOutcomes(use_fft_convolution=True, fft_threshold=64)
    hosp_fft, _, _ = X_fft.hospitalizations_by_location(
        states=states,
        times=times,
        compartments=comp,
        param_names=names,
        params=params_T,
        population=pop,
        prob_spec=[0.1, 0.1],
        delay_type="gamma",
        delay_mean=7.0,
        delay_cv=0.5,
        aggregate="D",
        group_keys=("age_strata",),
        return_group_keys=("age_strata",),
    )
    assert np.allclose(hosp_direct, hosp_fft, rtol=1e-10, atol=1e-10)


# ----------------
# Error conditions
# ----------------


def test_errors_invalid_times_nonmonotonic(
    toy_states_constant, toy_compartments, toy_params_broadcast, population_two_nodes
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    times_bad = np.array([0, 1, 1, 2, 3, 4], dtype=float)  # not strictly increasing

    with pytest.raises(ValueError):
        X.incidence_by_location_agg(
            states=toy_states_constant,
            times=times_bad,
            compartments=toy_compartments,
            param_names=names,
            params=params,
            population=population_two_nodes,
            normalize_by_population=True,
            group_keys=("age_strata",),
            return_group_keys=("age_strata",),
            aggregate=None,
        )


def test_errors_missing_param_prefix(
    toy_states_constant, times_numeric, toy_compartments, population_two_nodes
):
    # Supply wrong prefixes via the CONSTRUCTOR
    X = VectorizedOutcomes(
        rate_prefixes=("alphaDoesNotExist",), use_fft_convolution=False
    )
    names = np.array(["r0", "gamma", "theta1"])
    params = np.repeat(np.array([[1.0], [1.0], [1.0]]), len(times_numeric), axis=1)

    with pytest.raises(ValueError):
        X.incidence_by_location_agg(
            states=toy_states_constant,
            times=times_numeric,
            compartments=toy_compartments,
            param_names=names,
            params=params,
            population=population_two_nodes,
            normalize_by_population=True,
            group_keys=("age_strata",),
            return_group_keys=("age_strata",),
            aggregate=None,
        )


def test_errors_bad_aggregate_type_with_numeric_times(
    toy_states_constant,
    times_numeric,
    toy_compartments,
    toy_params_broadcast,
    population_two_nodes,
):
    X = VectorizedOutcomes(use_fft_convolution=False)
    names, params = toy_params_broadcast
    # Passing string aggregate with numeric times should raise
    with pytest.raises(ValueError):
        X.incidence_by_location_agg(
            states=toy_states_constant,
            times=times_numeric,
            compartments=toy_compartments,
            param_names=names,
            params=params,
            population=population_two_nodes,
            normalize_by_population=True,
            group_keys=("age_strata",),
            return_group_keys=("age_strata",),
            aggregate="W-SAT",
        )
