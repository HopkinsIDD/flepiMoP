import numpy as np
import numpy.typing as npt
from numba import njit, prange
from collections.abc import Callable


# === Constants ===
_PARALLEL_THRESHOLD = 1e7


# === Parameter time slicing (for solve_ivp's continuous t) ===


def param_slice(
    parameters: npt.NDArray[np.float64], t: float, mode: str = "linear"
) -> npt.NDArray[np.float64]:
    """
    Slice parameters at a given time.

    Args:
        parameters (npt.NDArray[np.float64]):
            Array of shape (P, T, N) or (P, T).
            - P: number of parameters.
            - T: unit-spaced discrete time points (0, 1, 2, ...).
            - N: number of spatial nodes (optional).
            If shape is (P, T), it is broadcast to (P, T, 1).
        t (float):
            Continuous time used by the solver.
        mode (str):
            Interpolation mode. One of:
            - `"step"`: piecewise-constant in time using ``floor(t)``.
            - `"linear"`: linear interpolation between ``floor(t)`` and ``ceil(t)``.

    Returns:
        npt.NDArray[np.float64]:
            Array of shape (P, N) representing the parameter values at time ``t``.
    """

    if parameters.ndim == 2:  # (P, T) -> (P, T, 1)
        parameters = parameters[:, :, None]

    P, T, N = parameters.shape

    if mode == "step":
        i = int(np.clip(np.floor(t), 0, T - 1))
        out = parameters[:, i, :]
        return out

    elif mode == "linear":
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, T - 1)
        alpha = float(np.clip(t - i0, 0.0, 1.0))
        out = (1.0 - alpha) * parameters[:, i0, :] + alpha * parameters[:, i1, :]

        return out

    else:
        raise ValueError(f"Unknown param_time_mode: {mode}")


# === Helper for symbolic expression resolution (from a time slice) ===


def resolve_param_expr_from_slice(
    expr: str,
    param_t: npt.NDArray[np.float64],  # (P, N)
    param_name_to_row: dict[str, int],  # "beta" -> row index
) -> npt.NDArray[np.float64]:
    """
    Resolve a product expression like ``'beta * gamma'`` against a (P, N) parameter slice.

    Args:
        param_slice (npt.NDArray[np.float64]):
            Array of shape (P, N) containing parameter values.
        expr (str):
            A product expression consisting of parameter names joined by `*`,
            for example ``"beta * gamma"``.
        name_to_row (dict[str, int]):
            Mapping from parameter name to its corresponding row index in ``param_slice``.

    Returns:
        npt.NDArray[np.float64]:
            Array of shape (N,) containing the evaluated product.
    """

    terms = [s.strip() for s in expr.split("*")]
    out = None
    for name in terms:
        row = param_name_to_row[name]
        vec = param_t[row, :]  # (N,)
        out = vec if out is None else out * vec

    return out


# === 1. Core Proportion Logic ===


@njit(fastmath=True)
def prod_along_axis0(arr_2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the product along the first axis of a 2D array.

    This function is implemented manually instead of using ``np.prod`` to
    remain compatible with Numba's ``nopython`` mode.

    Args:
        arr_2d (npt.NDArray[np.float64]):
            A 2D NumPy array of shape (M, N).

    Returns:
        npt.NDArray[np.float64]:
            A 1D NumPy array of shape (N,) containing the product of values
            along axis 0.
    """

    n_cols = arr_2d.shape[1]
    result = np.ones(n_cols, dtype=arr_2d.dtype)
    for i in range(arr_2d.shape[0]):
        for j in range(n_cols):
            result[j] *= arr_2d[i, j]
    return result


@njit(fastmath=True)
def compute_proportion_sums_exponents(
    states_current: npt.NDArray[np.float64],  # (C, N)
    transitions: npt.NDArray[np.float64],  # (5, Tn)
    proportion_info: npt.NDArray[np.float64],  # (3, Pk)
    transition_sum_compartments: npt.NDArray[np.float64],  # (S,)
    param_t: npt.NDArray[np.float64],  # (P, N)  <-- time-sliced parameters
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute multiplicative proportion terms, source sizes, and a flag for ``n_p == 1``.

    Matches legacy ``rk4_integration`` math:

    * If only one proportion term (``n_p == 1``), normalize the first proportion term
    and multiply by the rate parameter immediately.
    * Otherwise, compute proportion contributions for all terms and leave parameter
    multiplication and mobility mixing to later.

    Args:
        states_current (npt.NDArray[np.float64]):
            Array of shape (C, N) containing the current compartment states.
        transitions (npt.NDArray[np.int64]):
            Transition definition array of shape (5, Tn) describing source/destination compartments
            and proportion term indices.
        proportion_info (npt.NDArray[np.int64]):
            Array containing proportion term configuration.
        transition_sum_compartments (npt.NDArray[np.int64]):
            Compartments summed to form proportion denominators.
        parameters_t (npt.NDArray[np.float64]):
            Parameter values for the current time step, shape (P, N).

    Returns:
        tuple:
            total_rates (npt.NDArray[np.float64]):
                Array of shape (Tn, N) containing proportion products (includes parameter if ``n_p == 1``).
            source_numbers (npt.NDArray[np.float64]):
                Array of shape (Tn, N) containing summed counts for the first proportion term.
            single_prop_mask (npt.NDArray[np.bool_]):
                Boolean array of shape (Tn,) where True indicates transitions with ``n_p == 1``.
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


def compute_transition_rates(
    total_rates_base: npt.NDArray[np.float64],  # (Tn, N) e.g., I for S->E
    source_numbers: npt.NDArray[np.float64],  # (Tn, N) e.g., S for S->E (not used here)
    transitions: npt.NDArray[np.float64],  # (5, Tn)
    param_t: npt.NDArray[np.float64],  # (P, N)
    percent_day_away: float,
    proportion_who_move: npt.NDArray[np.float64],  # (N,)
    mobility_data: npt.NDArray[np.float64],
    mobility_data_indices: npt.NDArray[np.float64],
    mobility_row_indices: npt.NDArray[np.float64],
    population: npt.NDArray[np.float64],  # (N,)
    single_prop_mask: npt.NDArray[np.float64],  # (Tn,)
    param_expr_lookup: dict[int, str] | None = None,
    param_name_to_row: dict[str, int] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Compute total transition rates for each transition and node, applying parameter
    scaling and mobility mixing according to model configuration.

    If a transition has exactly one proportion term (``n_p == 1``), the associated
    parameter is assumed to have been applied earlier in
    ``compute_proportion_sums_exponents``, and no mobility mixing is performed.

    Args:
        total_rates_base (npt.NDArray[np.float64]):
            Base rates for each transition and node, shape (Tn, N).
        source_numbers (npt.NDArray[np.float64]):
            Source compartment sizes for each transition and node, shape (Tn, N).
            Not used in this function.
        transitions (npt.NDArray[np.float64]):
            Transition definition array of shape (5, Tn).
        param_t (npt.NDArray[np.float64]):
            Parameter values at the current time step, shape (P, N).
        percent_day_away (float):
            Fraction of the day individuals spend away from their home node.
        proportion_who_move (npt.NDArray[np.float64]):
            Fraction of individuals in each node who move, shape (N,).
        mobility_data (npt.NDArray[np.float64]):
            CSR-format data array for mobility between nodes.
        mobility_data_indices (npt.NDArray[np.float64]):
            CSR-format index pointer array for mobility data.
        mobility_row_indices (npt.NDArray[np.float64]):
            CSR-format row indices array for mobility data.
        population (npt.NDArray[np.float64]):
            Population counts per node, shape (N,).
        single_prop_mask (npt.NDArray[np.float64]):
            Boolean mask of shape (Tn,) where True indicates transitions
            with exactly one proportion term.
        param_expr_lookup (dict[int, str] | None):
            Optional mapping from parameter index to an expression string
            (e.g., ``"beta * gamma"``). If None, parameters are taken directly.
        param_name_to_row (dict[str, int] | None):
            Mapping from parameter names to their corresponding row indices
            in ``param_t``. Required if ``param_expr_lookup`` is provided.

    Returns:
        npt.NDArray[np.float64]:
            Total rates for each transition and node after parameter scaling and
            mobility mixing, shape (Tn, N).
    """
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

    return total_rates


# === 3. Binomial Stochastic (NumPy) ===
# (Kept for optional tau-leaping outside solve_ivp; not used in deterministic RHS.)


def compute_transition_amounts_numpy_binomial(
    source_numbers: npt.NDArray[np.float64],
    total_rates: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """
    Perform binomial draws for stochastic updates over a time step.

    Uses per-node, per-transition binomial sampling to determine the number of
    individuals transitioning within the interval ``dt``, based on the total
    transition rates. Returns per-step counts (not ``dy/dt``).

    Args:
        source_numbers (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the number of individuals
            in the source compartment for each transition and node.
        total_rates (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the per-capita transition
            rates for each transition and node.
        dt (float):
            Duration of the time step.

    Returns:
        npt.NDArray[np.float64]:
            Array of shape (Tn, N) containing the integer number of individuals
            who transitioned in the given time step, as 64-bit floats.
    """
    probs = 1.0 - np.exp(-dt * total_rates)
    draws = np.random.binomial(source_numbers, probs)
    return draws


# === 4. Transition Amounts (Deterministic dy/dt only) ===


@njit(fastmath=True)
def compute_transition_amounts_serial(
    source_numbers: npt.NDArray[np.float64], total_rates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute deterministic instantaneous flux for each transition and node (serial).

    Calculates the rate of change (``dy/dt``) as the product of the source compartment
    size and the per-capita transition rate, without any stochasticity. This is a
    serial implementation optimized with Numba.

    Args:
        source_numbers (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the number of individuals
            in the source compartment for each transition and node.
        total_rates (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the per-capita transition
            rates for each transition and node.

    Returns:
        npt.NDArray[np.float64]:
            Array of shape (Tn, N) containing the deterministic instantaneous
            flux (``dy/dt`` contribution) for each transition and node.
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
    source_numbers: npt.NDArray[np.float64], total_rates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute deterministic instantaneous flux for each transition and node (parallel).

    Calculates the rate of change (``dy/dt``) as the product of the source compartment
    size and the per-capita transition rate, without any stochasticity. This
    implementation uses Numba's parallelization over transitions for improved
    performance on larger workloads.

    Args:
        source_numbers (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the number of individuals
            in the source compartment for each transition and node.
        total_rates (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the per-capita transition
            rates for each transition and node.

    Returns:
        npt.NDArray[np.float64]:
            Array of shape (Tn, N) containing the deterministic instantaneous
            flux (``dy/dt`` contribution) for each transition and node.
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
    source_numbers: npt.NDArray[np.float64], total_rates: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Select and execute the appropriate transition amount computation method.

    Chooses between the serial and parallel deterministic flux computations
    based on the total workload size. Returns the instantaneous flux (``dy/dt``)
    without scaling by ``dt`` or applying any integration method.

    Args:
        source_numbers (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the number of individuals
            in the source compartment for each transition and node.
        total_rates (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the per-capita transition
            rates for each transition and node.

    Returns:
        npt.NDArray[np.float64]:
            Array of shape (Tn, N) containing the deterministic instantaneous
            flux (``dy/dt`` contribution) for each transition and node.
    """
    workload = source_numbers.shape[0] * source_numbers.shape[1]
    if workload >= _PARALLEL_THRESHOLD:
        return compute_transition_amounts_parallel(source_numbers, total_rates)
    else:
        return compute_transition_amounts_serial(source_numbers, total_rates)


# === 5. Assemble Flux Vector ===


@njit(fastmath=True)
def assemble_flux(
    amounts: npt.NDArray[np.float64],  # (Tn, N), instantaneous flux
    transitions: npt.NDArray[np.float64],  # (5, Tn)
    ncompartments: int,
    nspatial_nodes: int,
) -> npt.NDArray[np.float64]:
    """
    Assemble the net rate of change (``dy/dt``) from individual transition fluxes.

    For each transition, subtracts the instantaneous flux from the source compartment
    and adds it to the destination compartment for every spatial node. The resulting
    2D array is then flattened for use in solvers like ``solve_ivp``.

    Args:
        amounts (npt.NDArray[np.float64]):
            Array of shape (Tn, N) containing the instantaneous flux
            (e.g., from ``compute_transition_amounts_*``) for each transition and node.
        transitions (npt.NDArray[np.float64]):
            Array of shape (5, Tn) defining transitions, where
            ``transitions[0, t_idx]`` is the source compartment index and
            ``transitions[1, t_idx]`` is the destination compartment index.
        ncompartments (int):
            Total number of compartments in the model.
        nspatial_nodes (int):
            Total number of spatial nodes or subpopulations.

    Returns:
        npt.NDArray[np.float64]:
            Flattened array of shape (ncompartments * nspatial_nodes,) containing
            the net rate of change (``dy/dt``) for all compartments and nodes.
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
    states_current: npt.NDArray[np.float64],  # (C, N), modified in place
    today: int,
    seeding_data: dict[str, npt.NDArray[np.float64]],
    seeding_amounts: npt.NDArray[np.float64],
    daily_incidence: npt.NDArray[np.float64] | None = None,  # (D, C, N) if provided
) -> None:
    """
    Apply legacy-style seeding to the model state for a given day.

    This replicates the seeding logic from the original ``rk4_integration``:
    individuals are subtracted from source compartments and added to destination
    compartments for specific subpopulations, based on seeding events scheduled
    for the current day. Negative compartment counts are prevented by clamping
    to zero.

    Optionally, daily incidence counts can be updated to include seeded amounts.

    Args:
        states_current (npt.NDArray[np.float64]):
            Array of shape (C, N) containing the current model state
            (compartments × subpopulations). Modified in place.
        today (int):
            Current simulation day index.
        seeding_data (dict[str, npt.NDArray[np.float64]]):
            Dictionary containing seeding metadata with keys:
              - ``"day_start_idx"``: 1D array of indices into the seeding arrays
              - ``"seeding_subpops"``: 1D array of subpopulation indices
              - ``"seeding_sources"``: 1D array of source compartment indices
              - ``"seeding_destinations"``: 1D array of destination compartment indices
        seeding_amounts (npt.NDArray[np.float64]):
            1D array of seeding amounts corresponding to the events.
        daily_incidence (npt.NDArray[np.float64], optional):
            Array of shape (D, C, N) containing daily incidence counts.
            If provided, seeded amounts are added to the relevant entries.

    Returns:
        None
    """
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
    Wraps precomputed model structures and builds a right-hand-side (RHS)
    function suitable for SciPy's ``solve_ivp`` integrator.

    The factory stores simulation constants, parameter mappings, and optional
    seeding information. It produces a callable of the form:
    ``rhs(t, y, parameters) -> dy/dt``.

    Required keys in ``precomputed`` define the model structure
    (compartments, mobility, transitions, etc.). Optional keys provide
    seeding metadata and incidence tracking.

    Attributes:
        REQUIRED_KEYS (list[str]):
            Names of keys that must be present in ``precomputed`` for
            successful initialization.
        OPTIONAL_KEYS (list[str]):
            Keys in ``precomputed`` that, if present, enable optional features
            such as seeding and daily incidence tracking.
        precomputed (dict):
            Precomputed model structures passed during initialization.
        param_expr_lookup (dict[int, str] | None):
            Mapping from parameter row index to expression string (e.g.,
            ``"beta * gamma"``). Optional.
        param_name_to_row (dict[str, int] | None):
            Mapping from parameter names to their row indices. Required if
            ``param_expr_lookup`` is provided.
        param_time_mode (str):
            Parameter time-dependence mode. Either:
              - ``"step"``: piecewise constant in time
              - ``"linear"``: linearly interpolated in time
        _last_day_applied (dict):
            Tracks the last simulation day for which seeding was applied.
        _rhs (Callable | None):
            The cached RHS function built from the precomputed structures.

    Args:
        precomputed (dict):
            Dictionary containing required and optional precomputed model
            structures (see ``REQUIRED_KEYS`` and ``OPTIONAL_KEYS``).
        param_expr_lookup (dict[int, str], optional):
            Mapping from parameter row index to expression string.
        param_name_to_row (dict[str, int], optional):
            Mapping from parameter names to row indices, required if
            ``param_expr_lookup`` is given.
        param_time_mode (str, optional):
            Parameter time interpolation mode. Defaults to ``"linear"``.

    Raises:
        KeyError: If any of the required keys are missing from ``precomputed``.
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
        self._rhs = None

        # build once using defaults present in precomputed (if any)
        self.build_rhs()

    def reset_seeding_tracker(self) -> None:
        """
        Reset the internal seeding day-change tracker.

        This clears the stored "last day applied" value used to prevent
        multiple seeding applications within the same simulation day.
        Call this before starting a new ``solve_ivp`` run or restarting
        a simulation.

        Returns:
            None
        """
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
        Construct and store an ODE right-hand-side (RHS) function for ``solve_ivp``.

        The generated RHS computes compartmental derivatives (dy/dt) for a spatial SEIR-type
        model, including parameter time-slicing, proportion term calculation, mobility mixing,
        and optional legacy seeding applied once per simulation day.

        This method can be called again to rebuild the RHS with different parameters or seeding data.

        Args:
            param_time_mode (str | None, optional):
                Parameter time interpolation mode. If ``None``, uses the factory's
                default. Must be one of:
                * ``"step"`` — piecewise-constant parameters based on floor(t).
                * ``"linear"`` — linear interpolation between discrete time steps.
            seeding_data (dict[str, npt.NDArray[np.float64]] | None, optional):
                Legacy-style seeding specification. Expected keys include:
                * ``"day_start_idx"`` — int array of start indices for each day.
                * ``"seeding_subpops"`` — int array of subpopulation indices.
                * ``"seeding_sources"`` — int array of source compartment indices.
                * ``"seeding_destinations"`` — int array of destination compartment indices.
                If ``None``, seeding is skipped.
            seeding_amounts (npt.NDArray[np.float64] | None, optional):
                Amounts to transfer for each seeding event, aligned with ``seeding_data``.
                If ``None``, seeding is skipped.
            daily_incidence (npt.NDArray[np.float64] | None, optional):
                Optional array of shape ``(D, C, N)`` to accumulate seeded cases
                per day, compartment, and node.

        Returns:
            Callable[[float, npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
                A function ``rhs(t, y, parameters) -> dy/dt`` suitable for passing to SciPy's
                ``solve_ivp``. This function:
                1. Optionally applies seeding at the start of each day.
                2. Slices parameters for time ``t``.
                3. Computes proportion terms and source sizes.
                4. Applies mobility mixing and parameter scaling.
                5. Assembles the net flux ``dy/dt`` as a flattened array.
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
        def rhs(
            t: float, y: npt.NDArray[np.float64], parameters: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
            """
            Compute the derivative dy/dt for the SEIR-style model at time ``t``.

            This function is intended for use with SciPy's ``solve_ivp`` and
            incorporates optional once-per-day seeding, parameter time-slicing,
            proportion term computation, mobility mixing, and flux assembly.

            Args:
                t (float):
                    Continuous simulation time in days.
                y (npt.NDArray[np.float64]):
                    Flattened state vector of shape ``(C*N,)`` where ``C`` is the
                    number of compartments and ``N`` is the number of spatial nodes.
                parameters (npt.NDArray[np.float64]):
                    Parameter array of shape ``(P, T, N)`` or ``(P, T)`` giving
                    time-dependent parameter values.

            Returns:
                npt.NDArray[np.float64]:
                    Flattened array ``dy/dt`` of shape ``(C*N,)``.
            """

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

    def create_rhs(
        self,
    ) -> Callable[
        [float, npt.NDArray[np.float64], npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]:
        """
        Return the currently built ODE right-hand-side function.

        If no RHS has been built yet, this method calls ``build_rhs()`` with
        default parameters from the factory's precomputed data.

        Returns:
            Callable[[float, npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
                A function ``rhs(t, y, parameters) -> dy/dt`` suitable for passing
                to SciPy's ``solve_ivp``.
        """
        if self._rhs is None:
            return self.build_rhs()
        return self._rhs

    def solve(
        self,
        y0: npt.NDArray[np.float64],
        parameters: npt.NDArray[np.float64],
        t_span: tuple[float, float],
        t_eval: npt.NDArray[np.float64] | None = None,
        **solve_ivp_kwargs,
    ):
        """
        Integrate the model forward in time using the stored RHS.

        This is a convenience wrapper around ``scipy.integrate.solve_ivp`` that
        automatically resets the seeding tracker before each run and passes the
        factory's currently built RHS function to the solver.

        Args:
            y0 (npt.NDArray[np.float64]):
                Flattened initial state vector of shape ``(C*N,)``.
            parameters (npt.NDArray[np.float64]):
                Parameter array of shape ``(P, T, N)`` or ``(P, T)`` giving
                time-dependent parameter values.
            t_span (tuple[float, float]):
                Start and end times for integration ``(t0, tf)``.
            t_eval (npt.NDArray[np.float64] | None, optional):
                Times at which to store the computed solution. If ``None``,
                solver chooses its own step locations.
            **solve_ivp_kwargs:
                Additional keyword arguments passed to ``solve_ivp`` (e.g., ``method``,
                ``rtol``, ``atol``, ``vectorized``).

        Returns:
            scipy.integrate.OdeResult:
                The result object returned by ``solve_ivp``, with attributes
                like ``t``, ``y``, and ``success``.
        """
        self.reset_seeding_tracker()  # fresh run, re-apply day 0 seeding as needed
        rhs = self.create_rhs()
        return solve_ivp(
            fun=lambda t, y: rhs(t, y, parameters),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            **solve_ivp_kwargs,
        )
