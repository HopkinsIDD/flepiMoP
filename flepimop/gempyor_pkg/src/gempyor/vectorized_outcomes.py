# vectorized_outcomes.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import numpy.typing as npt

# Optional: SciPy is NOT required. We use closed-form for exponential,
# and an Erlang approximation for gamma (integer shape via repeated conv).
# If you prefer exact non-integer gamma CDFs, swap in scipy.special.gammainc.


# ---------------------------------------------------------------------------
# Utilities: discrete delay kernels on a generic bin grid
# ---------------------------------------------------------------------------

def _ensure_prob_vector(prob: float | npt.NDArray[np.float64], N: int) -> npt.NDArray[np.float64]:
    """Broadcast scalar -> vector(N), ensure contiguous float64."""
    if np.isscalar(prob):
        v = np.full(N, float(prob), dtype=np.float64)
    else:
        v = np.asarray(prob, dtype=np.float64)
        if v.shape != (N,):
            raise ValueError(f"prob vector must have shape (N,), got {v.shape}")
    return np.ascontiguousarray(v, dtype=np.float64)


def make_fixed_delay_kernel(delay_days: float, bin_width_days: float, max_bins: int | None = None) -> npt.NDArray[np.float64]:
    """
    Discretize a fixed delay into the bin grid.
    If delay is not an integer multiple of the bin width, split mass across the two adjacent bins.
    """
    d = float(delay_days) / float(bin_width_days)
    i0 = int(math.floor(d))
    frac = d - i0
    # kernel length: either 1 or 2 (split) unless user enforces a max
    if frac < 1e-12:
        k = np.zeros(i0 + 1, dtype=np.float64)
        k[i0] = 1.0
    else:
        k = np.zeros(i0 + 2, dtype=np.float64)
        k[i0] = 1.0 - frac
        k[i0 + 1] = frac

    if max_bins is not None and k.size > max_bins:
        # Truncate tail if user forces limit (mass loss warning)
        k = k[:max_bins]
        s = k.sum()
        if s > 0:
            k /= s
    return k


def make_exponential_delay_kernel(mean_days: float, bin_width_days: float, tail_mass: float = 1e-8) -> npt.NDArray[np.float64]:
    """
    Discrete geometric-like kernel from an exponential delay (mean_days).
    pmf[k] = exp(-λ*kΔ) - exp(-λ*(k+1)Δ), with λ = 1/mean, Δ = bin_width_days.
    Truncate when remaining tail mass < tail_mass.
    """
    mean = float(mean_days)
    if mean <= 0:
        # Degenerates to fixed zero delay
        return np.array([1.0], dtype=np.float64)

    lam = 1.0 / mean
    delta = float(bin_width_days)
    # per-bin survival factor
    r = math.exp(-lam * delta)
    p = 1.0 - r  # first-bin mass multiplier

    # Build until tail < tail_mass
    vals = []
    surv = 1.0
    while surv > tail_mass:
        vals.append(surv * p)
        surv *= r

        # Safety cap
        if len(vals) > 1_000_000:
            break

    k = np.asarray(vals, dtype=np.float64)
    s = k.sum()
    if s > 0:
        k /= s
    return k


def _convolve_discrete(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Small 1D 'valid' convolution (full support). For short kernels this is fast in NumPy.
    """
    la, lb = a.size, b.size
    out = np.zeros(la + lb - 1, dtype=np.float64)
    # Manual conv (short kernels typical); avoids FFT overhead
    for i in range(la):
        out[i:i + lb] += a[i] * b
    return out


def make_gamma_delay_kernel(mean_days: float, shape_k: float, bin_width_days: float, tail_mass: float = 1e-8) -> npt.NDArray[np.float64]:
    """
    Discrete kernel approximating a Gamma(k, θ) delay via Erlang (integer shape) convolution of exponentials.
    - mean = k * θ  => λ = k / mean
    - If k is non-integer, we round to nearest integer (Erlang approximation), preserving mean.
    This is numerically robust and fast; for exact non-integer k, substitute SciPy CDF differencing if desired.
    """
    mean = float(mean_days)
    k = int(round(float(shape_k)))
    if k <= 0:
        # Fallback to fixed: 0 delay
        return np.array([1.0], dtype=np.float64)
    # Erlang: sum of k exponentials with common rate λ = k / mean
    exp_kernel = make_exponential_delay_kernel(mean_days=mean / k, bin_width_days=bin_width_days, tail_mass=tail_mass / k)
    out = exp_kernel
    for _ in range(k - 1):
        out = _convolve_discrete(out, exp_kernel)
        # Optional truncation of tiny tails for memory
        # Keep cumulative tail small to control size
        csum = out[::-1].cumsum()[::-1]
        keep = csum > tail_mass
        if keep.any():
            last = np.nonzero(keep)[0][-1]
            out = out[: last + 1]
    s = out.sum()
    if s > 0:
        out /= s
    return out


# ---------------------------------------------------------------------------
# Compiled spec data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutcomeLeaf:
    """
    A leaf outcome is defined by a fixed set of transition rows and (optional) node filtering.
    per-node probability is applied, then delayed by a discrete kernel.
    """
    name: str
    transition_indices: npt.NDArray[np.int64]    # shape (R,), rows into (Tn, N)
    node_mask_num: npt.NDArray[np.float64]      # shape (N,), 0/1 numeric mask
    prob_vec: npt.NDArray[np.float64]           # shape (N,), per-node probability
    kernel: npt.NDArray[np.float64]             # shape (L,), discrete delay pmf

    # Working buffers (allocated by observer): not part of identity
    # (We will attach external buffers in the observer for speed; kept here for clarity)


@dataclass(frozen=True)
class OutcomeSum:
    """A sum node aggregates child outcomes by name."""
    name: str
    children: Tuple[str, ...]


@dataclass(frozen=True)
class CompiledOutcomes:
    """
    Fully compiled outcomes ready for high-speed accumulation.
    """
    leaves: Dict[str, OutcomeLeaf]
    sums: Dict[str, OutcomeSum]
    # Precomputed properties
    names_topo: Tuple[str, ...]                 # topological order for finalize (leaves -> sums)
    max_kernel_len: int


# ---------------------------------------------------------------------------
# Observer: fast, allocation-free per-step accumulation
# ---------------------------------------------------------------------------

class OutcomeObserver:
    """
    High-performance online accumulator for outcomes during simulation.

    Usage:
      obs = OutcomeObserver(spec, n_nodes=N, t0=t0, tf=tf, bin_width_days=1.0)
      ...
      # For each solver step with integrated transition counts over [t0_step, t1_step):
      obs.on_step(t0_step, t1_step, step_counts_tn_by_node)  # (Tn, N)

      # At the end:
      times, series_by_name = obs.finalize(names=None)  # default: all outcomes (leaves + sums)

    Notes:
      - Input 'step_counts' MUST be counts integrated over the step (not rates).
      - Binning splits each step proportionally by overlap with bin intervals.
      - Delay kernels schedule mass into future bins (no dt assumptions).
    """

    __slots__ = (
        "spec", "N", "t0", "tf", "bin_width", "nbins", "bin_edges",
        "leaf_names", "sum_names",
        "_leaf_sched",     # dict name -> ndarray (N, nbins + maxK)
        "_leaf_work",      # dict name -> ndarray (N,) reusable buffer
        "_bin_overlap_buf" # ndarray (nbins,) for overlap fractions
    )

    def __init__(self,
                 spec: CompiledOutcomes,
                 n_nodes: int,
                 t0: float,
                 tf: float,
                 bin_width_days: float = 1.0):
        self.spec = spec
        self.N = int(n_nodes)
        self.t0 = float(t0)
        self.tf = float(tf)
        self.bin_width = float(bin_width_days)

        if self.tf <= self.t0:
            raise ValueError("tf must be > t0")

        # Determine binning
        self.nbins = int(math.ceil((self.tf - self.t0) / self.bin_width))
        self.bin_edges = self.t0 + np.arange(self.nbins + 1, dtype=np.float64) * self.bin_width

        # Prepare schedules for leaves (we materialize only leaves; sums will be built at finalize)
        self.leaf_names = tuple(spec.leaves.keys())
        self.sum_names = tuple(spec.sums.keys())
        horizon = self.nbins + int(spec.max_kernel_len)
        self._leaf_sched: Dict[str, npt.NDArray[np.float64]] = {
            name: np.zeros((self.N, horizon), dtype=np.float64) for name in self.leaf_names
        }

        # Reusable per-leaf work vectors to avoid allocations
        self._leaf_work: Dict[str, npt.NDArray[np.float64]] = {
            name: np.zeros(self.N, dtype=np.float64) for name in self.leaf_names
        }

        # Reusable bin-overlap scratch
        self._bin_overlap_buf = np.zeros(self.nbins, dtype=np.float64)

    # ----------------- core accumulation -----------------

    def on_step(self,
                t_start: float,
                t_end: float,
                step_counts_tn_by_node: npt.NDArray[np.float64]) -> None:
        """
        Accumulate one solver step worth of integrated transition counts.

        Parameters
        ----------
        t_start, t_end : float
            Step interval [t_start, t_end). Must satisfy t0 <= t_start < t_end <= tf.
        step_counts_tn_by_node : (Tn, N) float64
            Counts integrated over this step for each transition row (Tn) and node (N).
        """
        ts = float(t_start)
        te = float(t_end)
        if not (self.t0 - 1e-12 <= ts < te <= self.tf + 1e-12):
            raise ValueError("Step times out of bounds or inverted.")

        counts = np.ascontiguousarray(step_counts_tn_by_node, dtype=np.float64)
        if counts.shape[1] != self.N:
            raise ValueError(f"counts second dim must be N={self.N}, got {counts.shape}")

        # --- 1) Determine overlap with bins (proportional split) ---
        # Compute overlap length with each bin; we only act on bins with nonzero overlap
        # Efficient vector formulation: overlap = max(0, min(bin_right, te) - max(bin_left, ts))
        be = self.bin_edges
        lefts = np.maximum(be[:-1], ts)
        rights = np.minimum(be[1:], te)
        np.subtract(rights, lefts, out=self._bin_overlap_buf)  # rights - lefts
        self._bin_overlap_buf[self._bin_overlap_buf < 0.0] = 0.0

        step_duration = te - ts
        if step_duration <= 0:
            return  # nothing to add

        # Convert to fractions per bin
        self._bin_overlap_buf /= step_duration

        # Indices of bins spanned by the step
        spanned_bins = np.nonzero(self._bin_overlap_buf > 0.0)[0]
        if spanned_bins.size == 0:
            return

        # --- 2) For each leaf outcome: aggregate transitions -> per-node counts ---
        for name in self.leaf_names:
            leaf = self.spec.leaves[name]
            work = self._leaf_work[name]     # (N,), reused
            sched = self._leaf_sched[name]   # (N, nbins + maxK)

            # work[:] = sum(counts[rows, :])  without allocating a new array
            work.fill(0.0)
            rows = leaf.transition_indices
            for r in rows:
                work += counts[r, :]   # (N,)

            # Apply node mask and per-node probability (both length N, contiguous)
            # Note: elementwise multiply is faster than boolean masking here.
            # work *= mask * prob
            np.multiply(work, leaf.node_mask_num, out=work)
            np.multiply(work, leaf.prob_vec, out=work)

            # --- 3) Distribute across spanned bins and apply delay kernel via outer product ---
            ker = leaf.kernel
            L = ker.size
            # For each overlapped bin b, add: sched[:, b:b+L] += (work * frac_b)[:, None] * ker[None, :]
            for b in spanned_bins:
                frac = self._bin_overlap_buf[b]
                if frac <= 0.0:
                    continue
                # portion = work * frac  (reuse work by scaling + restore)
                # To avoid mutating 'work', make a scaled view via broadcasting in the outer:
                # sched[:, b:b+L] += work[:,None] * (ker * frac)[None,:]
                # Pre-scale kernel once
                # (allocates a small length-L temp; far cheaper than copying N)
                ker_scaled = ker * frac
                start = b
                stop = b + L
                # Bounds check (if kernel runs past horizon, safely truncate)
                if stop > sched.shape[1]:
                    # Truncate and renormalize lost tail proportionally (we simply drop it)
                    # This is acceptable if tf is chosen with margin; else warn externally.
                    slice_L = sched.shape[1] - start
                    if slice_L <= 0:
                        continue
                    # Add truncated outer product
                    sched[:, start:sched.shape[1]] += work[:, None] * ker_scaled[:slice_L][None, :]
                else:
                    sched[:, start:stop] += work[:, None] * ker_scaled[None, :]

    # ----------------- finalize and aggregation -----------------

    def finalize(self,
                 names: Optional[Iterable[str]] = None
                 ) -> Tuple[npt.NDArray[np.float64], Dict[str, npt.NDArray[np.float64]]]:
        """
        Return bin-left times and a dict of series:
          name -> (nbins, N) float64, time-major.
        Includes requested names (defaults to all: leaves + sums).

        Sum nodes are materialized here by adding their children's arrays.
        """
        # Base timeline (bin left edges)
        times = self.bin_edges[:-1].copy()

        # Materialize all leaves first
        out: Dict[str, npt.NDArray[np.float64]] = {}
        for name in self.leaf_names:
            sched = self._leaf_sched[name]              # (N, nbins + maxK)
            series = sched[:, :self.nbins].T.copy()     # -> (nbins, N), copy to detach
            out[name] = series

        # Topologically add sums
        for sname in self.spec.names_topo:
            if sname in out:
                continue  # leaf already present
            # It's a sum node
            children = self.spec.sums[sname].children
            acc = None
            for child in children:
                arr = out[child]
                if acc is None:
                    acc = arr.copy()    # (nbins, N)
                else:
                    acc += arr
            out[sname] = acc if acc is not None else np.zeros((self.nbins, self.N), dtype=np.float64)

        # Filter to requested names
        if names is not None:
            pick = tuple(names)
            out = {k: out[k] for k in pick}

        return times, out

    # ---- flexible aggregations (optional helpers) ----

    @staticmethod
    def aggregate_nodes(series: npt.NDArray[np.float64],
                        groups: Dict[str, npt.NDArray[np.int64]]) -> Dict[str, npt.NDArray[np.float64]]:
        """
        Sum over node sets. Input series shape (T, N). Returns dict group -> (T,) sum.
        groups: mapping name -> indices (1D int array). No allocations per group beyond the result.
        """
        T, N = series.shape
        out: Dict[str, npt.NDArray[np.float64]] = {}
        for gname, idx in groups.items():
            idx = np.asarray(idx, dtype=np.int64)
            v = series[:, idx].sum(axis=1)
            out[gname] = v
        return out

    @staticmethod
    def resample_time_sum(series: npt.NDArray[np.float64],
                          assign: npt.NDArray[np.int64],
                          n_bins_out: int) -> npt.NDArray[np.float64]:
        """
        Generic time aggregation by bin assignment.
        - series: (T_in, N)
        - assign: (T_in,) integer mapping each input bin i -> output bin j in [0, n_bins_out)
        Output: (T_out, N) with sums over assigned bins.
        This lets you support MMWR epiweeks or any calendar by providing 'assign'.
        """
        T_in, N = series.shape
        if assign.shape != (T_in,):
            raise ValueError("assign must have shape (T_in,)")
        out = np.zeros((n_bins_out, N), dtype=np.float64)
        # Vectorized scatter-add: per output bin, sum rows where assign==j
        # Efficient approach: sort by assign and cumulative sum
        order = np.argsort(assign, kind="stable")
        s_sorted = series[order, :]              # (T_in, N)
        a_sorted = assign[order]                 # (T_in,)
        # Find segment boundaries
        bounds = np.flatnonzero(np.diff(a_sorted, prepend=a_sorted[0]-1))  # start indices
        bounds = np.append(bounds, T_in)  # add sentinel end
        # Accumulate each segment
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            j = int(a_sorted[b0])
            out[j, :] += s_sorted[b0:b1, :].sum(axis=0)
        return out


# ---------------------------------------------------------------------------
# Spec compiler helpers (from your YAML-like structure)
# ---------------------------------------------------------------------------

def compile_outcomes(
    outcomes_cfg: Dict,
    *,
    # Required resolvers provided by caller (precomputed from model metadata):
    resolve_incidence_to_transition_rows: callable,
    # Example signature:
    #   resolve_incidence_to_transition_rows(
    #       infection_stage: str,
    #       vaccination_stage: str,
    #       variant_type: str,
    #       age_strata: str
    #   ) -> npt.NDArray[np.int64]    # transition row indices (R,)
    node_mask_from_labels: callable,
    # Example signature:
    #   node_mask_from_labels(incidence_dict_or_none) -> npt.NDArray[np.float64]  # length N (0/1)
    prob_vector_from_cfg: callable,
    # Example signature:
    #   prob_vector_from_cfg(outcome_name: str, base_prob: float|None) -> npt.NDArray[np.float64]
    N_nodes: int,
    bin_width_days: float,
) -> CompiledOutcomes:
    """
    Compile a YAML-like outcomes dict into fast arrays.
    You provide the resolvers so this stays decoupled from your model internals.
    """
    leaves: Dict[str, OutcomeLeaf] = {}
    sums: Dict[str, OutcomeSum] = {}

    maxK = 1

    # 1) First pass: build leaves and sum placeholders
    for name, spec in outcomes_cfg.items():
        if "sum" in spec:
            # Will fill later with children tuple
            continue

        # Leaf with structure:
        # source: incidence: {infection_stage, vaccination_stage, variant_type, age_strata}
        # probability: {value: {distribution: fixed, value: ...}}
        # delay: {value: {distribution: fixed|exponential|gamma, value: ...}}
        src = spec.get("source", {})
        inc = src.get("incidence", None)
        if inc is None:
            # It might be an alias 'source: <other outcome name>' (for H from I)
            # Treat it as a sum of that single child. We'll resolve sums next pass.
            continue

        # Resolve transitions
        rows = resolve_incidence_to_transition_rows(
            inc.get("infection_stage", None),
            inc.get("vaccination_stage", None),
            inc.get("variant_type", None),
            inc.get("age_strata", None),
        )
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError(f"Transition resolver must return 1D indices, got {rows.shape}")

        # Node mask (optional finer filtering; your resolver can return all-ones if not used)
        mask_num = node_mask_from_labels(inc)  # (N,) numeric 0/1
        mask_num = np.ascontiguousarray(mask_num, dtype=np.float64)
        if mask_num.shape != (N_nodes,):
            raise ValueError(f"Node mask must be (N,), got {mask_num.shape}")

        # Probability
        prob_cfg = spec.get("probability", {}).get("value", {})
        base_prob = prob_cfg.get("value", 1.0)
        prob_vec = prob_vector_from_cfg(name, base_prob)  # (N,)
        prob_vec = _ensure_prob_vector(prob_vec, N_nodes)

        # Delay kernel
        delay_cfg = spec.get("delay", {}).get("value", {})
        dkind = str(delay_cfg.get("distribution", "fixed")).lower()
        kval: npt.NDArray[np.float64]
        if dkind == "fixed":
            kval = make_fixed_delay_kernel(delay_cfg.get("value", 0.0), bin_width_days)
        elif dkind == "exponential":
            kval = make_exponential_delay_kernel(delay_cfg.get("value", 1.0), bin_width_days)
        elif dkind == "gamma":
            # Support inputs like {"k": 3, "mean": 5} or {"shape": 3, "mean": 5} or {"value": {"mean":..,"shape":..}}
            mean = float(delay_cfg.get("mean", delay_cfg.get("value", 1.0)))
            shape_k = float(delay_cfg.get("shape", delay_cfg.get("k", 1.0)))
            kval = make_gamma_delay_kernel(mean_days=mean, shape_k=shape_k, bin_width_days=bin_width_days)
        else:
            raise ValueError(f"Unknown delay distribution: {dkind}")

        maxK = max(maxK, int(kval.size))

        leaves[name] = OutcomeLeaf(
            name=name,
            transition_indices=rows,
            node_mask_num=mask_num,
            prob_vec=prob_vec,
            kernel=kval,
        )

    # 2) Second pass: fill sums (including aliases like 'source: other_outcome_name')
    for name, spec in outcomes_cfg.items():
        if "sum" in spec:
            children = tuple(spec["sum"])
            sums[name] = OutcomeSum(name=name, children=children)
            continue

        # alias pattern: { source: "<other_outcome_name>", probability:..., delay:... }
        src = spec.get("source", None)
        if isinstance(src, str):
            # Create a sum node referencing that leaf/aggregate
            sums[name] = OutcomeSum(name=name, children=(src,))
            continue
        elif isinstance(src, dict) and "incidence" not in src and "sum" not in spec:
            # If not incidence and not sum, but 'source' is present and refers to another outcome
            # e.g., H from I: "source: incidI_..." + prob+delay
            # Implement by creating a synthetic leaf "name" that uses the child's already-defined leaf stream
            # BUT with a new prob/delay. For performance and simplicity we treat it as sum over one child
            # and let the 'child' carry its own prob/delay. If you truly need per-outcome prob/delay chaining,
            # resolve it externally and pass as incidence leaves.
            # Here we interpret as simple alias (sum over child).
            maybe_child = src
            if isinstance(maybe_child, str):
                sums[name] = OutcomeSum(name=name, children=(maybe_child,))
                continue

    # 3) Topological order (leaves first, then sums resolving dependencies)
    # Simple Kahn-like: we assume no cycles (YAML shouldn't have).
    all_names = set(leaves.keys()) | set(sums.keys())
    deps = {sname: set(sums[sname].children) for sname in sums}
    produced = set(leaves.keys())
    order: List[str] = list(leaves.keys())  # start with leaves in insertion order
    added = True
    while added:
        added = False
        for sname, ch in list(deps.items()):
            if sname in produced:
                continue
            if ch.issubset(produced):
                order.append(sname)
                produced.add(sname)
                added = True
    if produced != all_names:
        missing = ", ".join(sorted(all_names - produced))
        raise ValueError(f"Cycle or missing children in outcome sums; unresolved: {missing}")

    return CompiledOutcomes(
        leaves=leaves,
        sums=sums,
        names_topo=tuple(order),
        max_kernel_len=maxK,
    )
