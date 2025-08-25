# gempyor/vectorized_outcomes.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import (
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Literal,
)
from scipy.stats import gamma as gamma_dist
from numpy.fft import rfft, irfft

DelayType = Literal["gamma", "exponential", "constant"]


class VectorizedOutcomes:
    """
    Vectorized post-simulation outcomes with flexible stratification.

    Highlights
    ----------
    • Compute infection incidence from S * I * Π(rate_prefixes) with optional population normalization.
    • Disaggregate by arbitrary group_keys (e.g., ("age_strata","vaccination_stage")).
    • Apply group-specific probabilities (e.g., hospitalization, death) using dict/array specs.
    • Convolve with delay kernels (gamma, exponential, constant) using direct or FFT path.
    • Aggregate to a requested return_group_keys (e.g., aggregate over vaccination to return Age×Time×Loc).

    Shapes
    ------
    states: (T, C, N)
    params: (P, T, N) or (P, T)
    Returns (by default): (Ages, Periods, Locations)

    Notes
    -----
    - If compartments lacks a 'vaccination_stage' column, we derive it from infection_stage
      by splitting on '_' (e.g., 'I1_v0' -> vaccination_stage='v0'). If absent, uses 'NA'.
    - Probabilities can be specified per (age_strata, vaccination_stage) group; the output can
      still be aggregated back to by-age with return_group_keys=("age_strata",).
    """

    # ------------------------
    # Construction & settings
    # ------------------------

    def __init__(
        self,
        *,
        rate_prefixes: Iterable[str] = ("r0", "gamma", "theta1"),
        use_fft_convolution: bool = True,
        fft_threshold: int = 1024,  # min series length to switch to FFT conv
    ):
        self.rate_prefixes = tuple(rate_prefixes)
        self.use_fft_convolution = bool(use_fft_convolution)
        self.fft_threshold = int(fft_threshold)

    # ------------------------
    # Public API: Incidence
    # ------------------------

    def incidence_by_location_agg(
        self,
        *,
        states: np.ndarray,  # (T, C, N)
        times: Sequence,  # length T; floats OR pandas.DatetimeIndex
        compartments: pd.DataFrame,  # rows align with C
        param_names: Sequence[str],  # (P,)
        params: np.ndarray,  # (P, T, N) or (P, T)
        population: Optional[np.ndarray] = None,  # (N,)
        normalize_by_population: bool = True,
        infected_stages: Iterable[str] = ("I1", "I2", "I3"),
        susceptible_stage: str = "S",
        # stratification controls:
        group_keys: Tuple[str, ...] = ("age_strata",),  # compute per these
        return_group_keys: Tuple[str, ...] = ("age_strata",),  # aggregate to these
        # aggregation in time:
        aggregate: Optional[
            float | str
        ] = "D",  # None | float window (numeric times) | pandas offset (datetime)
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Compute **incidence counts** per requested group(s), aggregated in time.

        Returns
        -------
        inc_agg : np.ndarray
            Shape (G_out, Periods, N) where G_out corresponds to unique combinations
            of `return_group_keys` (default: ages).
        labels_out : list[dict]
            Each dict has the keys in return_group_keys (e.g., {"age_strata": "A"}).
        period_index : np.ndarray or pandas.DatetimeIndex
            Period labels for the second axis.
        """
        comp = self._ensure_vax_column(compartments.copy())
        self._validate_inputs(
            states,
            times,
            comp,
            param_names,
            params,
            population,
            normalize_by_population,
        )

        # 1) Build groups to compute over
        labels_full = self._unique_group_labels(comp, group_keys, susceptible_stage)
        if len(labels_full) == 0:
            raise ValueError(f"No groups found for group_keys={group_keys}.")

        # 2) Build S_g(t,n), I_g(t,n) for each group g
        S_GTN, I_GTN, age_for_group = self._build_SI_by_groups(
            states, comp, labels_full, infected_stages, susceptible_stage
        )  # (G,T,N), (G,T,N), age index per group

        # 3) Rate tensor per *age* then mapped to groups
        ages = self._ordered_ages_from_S(comp, susceptible_stage)
        rate_ATN = self._rate_per_age(param_names, params, ages)  # (A,T,N_or_1)
        # broadcast N_if_1 to match states' N
        if rate_ATN.shape[2] == 1 and states.shape[2] > 1:
            rate_ATN = np.repeat(rate_ATN, states.shape[2], axis=2)
        rate_GTN = rate_ATN[age_for_group, :, :]  # (G,T,N)

        # 4) Normalize by population if requested
        if normalize_by_population:
            pop = np.asarray(population, dtype=float)
            pop[pop <= 0.0] = 1.0
            rate_GTN = rate_GTN / pop[None, None, :]

        # 5) Convert to per-interval counts via Δt (LEFT-EDGE alignment)
        dt, left_index, is_datetime = self._step_widths(
            times
        )  # dt length T-1, left_index = times[:-1]
        # interval counts aligned to LEFT edges (G, K=T-1, N)
        inc_interval = (S_GTN[:, :-1, :] * I_GTN[:, :-1, :] * rate_GTN[:, :-1, :]) * dt[
            None, :, None
        ]

        # 6) Aggregate in time
        inc_time_agg, period_index = self._aggregate_intervals(
            inc_interval, left_index, aggregate, is_datetime
        )

        # 7) Aggregate groups to the requested return_group_keys
        inc_out, labels_out = self._aggregate_groups(
            inc_time_agg, labels_full, group_keys, return_group_keys
        )

        return inc_out, labels_out, period_index

    # ---------------------------------
    # Public API: Hospitalizations/Death
    # ---------------------------------

    def hospitalizations_by_location(
        self,
        *,
        states: np.ndarray,  # (T, C, N)
        times: Sequence,
        compartments: pd.DataFrame,
        param_names: Sequence[str],
        params: np.ndarray,
        population: Optional[np.ndarray],
        # probability by strata (can be dict, array, scalar); see _resolve_probabilities
        prob_spec: Any,
        # delay (global/common kernel):
        delay_type: DelayType = "gamma",
        delay_mean: Optional[float] = None,
        delay_cv: Optional[float] = None,
        delay_shape: Optional[float] = None,
        delay_scale: Optional[float] = None,
        delay_keep_mass: float = 0.999,
        # stratification & output:
        infected_stages: Iterable[str] = ("I1", "I2", "I3"),
        susceptible_stage: str = "S",
        group_keys: Tuple[str, ...] = ("age_strata", "vaccination_stage"),
        return_group_keys: Tuple[str, ...] = ("age_strata",),
        aggregate: Optional[float | str] = "D",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Build hospitalization incidence:
          1) compute incidence by `group_keys`
          2) convolve each group with a common delay kernel
          3) multiply by group-specific probabilities
          4) aggregate groups to `return_group_keys` (default: age-only)

        prob_spec examples:
          - scalar (float): same for all groups
          - array-like length == #groups OR length == #ages (if group_keys include age)
          - dict with keys:
              ('age', 'vax') -> prob
              ('age',) or 'age' -> prob (fallback if exact combo missing)
              'default' -> prob (final fallback)
              or dicts as keys: {'age_strata':'A','vaccination_stage':'waned'} -> prob
        """
        # 1) incidence by full groups (keep high granularity to support per-strata probabilities)
        inc_GTN, labels_full, period_index = self.incidence_by_location_agg(
            states=states,
            times=times,
            compartments=compartments,
            param_names=param_names,
            params=params,
            population=population,
            normalize_by_population=True,
            infected_stages=infected_stages,
            susceptible_stage=susceptible_stage,
            group_keys=group_keys,
            return_group_keys=group_keys,  # keep full stratification here
            aggregate=None,  # need per-interval steps before delay
        )  # (G, K, N), K = T-1 (left-edge intervals)

        # 2) Build a DAILY kernel & resample to daily, then (optionally) to `aggregate`
        daily_index, daily_inc_GTN = self._to_daily_counts(
            times, inc_GTN
        )  # sum to daily buckets

        kernel = self._make_delay_kernel(
            delay_type=delay_type,
            mean=delay_mean,
            cv=delay_cv,
            shape=delay_shape,
            scale=delay_scale,
            keep_mass=delay_keep_mass,
        )

        # 3) Convolve per group/location
        conv_GTN = self._convolve_daily(daily_inc_GTN, kernel)  # (G, Td, N)

        # 4) Apply probabilities per group
        probs_G = self._resolve_probabilities(prob_spec, labels_full, group_keys)
        conv_GTN *= probs_G[:, None, None]

        # 5) Optionally aggregate time (weekly/monthly/epiweeks)
        conv_time_agg, period_idx = self._aggregate_daily(
            conv_GTN, daily_index, aggregate
        )

        # 6) Aggregate groups to requested return level (e.g., age-only)
        out, labels_out = self._aggregate_groups(
            conv_time_agg, labels_full, group_keys, return_group_keys
        )
        return out, labels_out, period_idx

    def deaths_by_location(
        self,
        *,
        states: np.ndarray,
        times: Sequence,
        compartments: pd.DataFrame,
        param_names: Sequence[str],
        params: np.ndarray,
        population: Optional[np.ndarray],
        prob_spec: Any,
        delay_type: DelayType = "gamma",
        delay_mean: Optional[float] = None,
        delay_cv: Optional[float] = None,
        delay_shape: Optional[float] = None,
        delay_scale: Optional[float] = None,
        delay_keep_mass: float = 0.999,
        infected_stages: Iterable[str] = ("I1", "I2", "I3"),
        susceptible_stage: str = "S",
        group_keys: Tuple[str, ...] = ("age_strata", "vaccination_stage"),
        return_group_keys: Tuple[str, ...] = ("age_strata",),
        aggregate: Optional[float | str] = "D",
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Deaths pipeline is identical to hospitalizations, differing only in prob_spec semantics.
        """
        return self.hospitalizations_by_location(
            states=states,
            times=times,
            compartments=compartments,
            param_names=param_names,
            params=params,
            population=population,
            prob_spec=prob_spec,
            delay_type=delay_type,
            delay_mean=delay_mean,
            delay_cv=delay_cv,
            delay_shape=delay_shape,
            delay_scale=delay_scale,
            delay_keep_mass=delay_keep_mass,
            infected_stages=infected_stages,
            susceptible_stage=susceptible_stage,
            group_keys=group_keys,
            return_group_keys=return_group_keys,
            aggregate=aggregate,
        )

    # ------------------------
    # Validation & utilities
    # ------------------------

    @staticmethod
    def _validate_inputs(
        states: np.ndarray,
        times: Sequence,
        compartments: pd.DataFrame,
        param_names: Sequence[str],
        params: np.ndarray,
        population: Optional[np.ndarray],
        normalize_by_population: bool,
    ) -> None:
        if states.ndim != 3:
            raise ValueError("states must be (T, C, N).")
        T, C, N = states.shape
        if len(times) != T:
            raise ValueError("`times` length must match T in states.")

        if (
            "infection_stage" not in compartments.columns
            or "age_strata" not in compartments.columns
        ):
            raise ValueError(
                "compartments must include 'infection_stage' and 'age_strata'."
            )
        if len(compartments) != C:
            raise ValueError("compartments rows must match C.")

        P = len(param_names)
        if params.shape[0] != P:
            raise ValueError("params first dim must match len(param_names).")
        if params.ndim not in (2, 3):
            raise ValueError("params must be (P,T) or (P,T,N).")
        if params.shape[1] != T:
            raise ValueError("params second dim must match T.")

        if normalize_by_population:
            if population is None or np.asarray(population).shape[0] != N:
                raise ValueError(
                    "population (N,) required when normalize_by_population=True."
                )

    @staticmethod
    def _ensure_vax_column(comp: pd.DataFrame) -> pd.DataFrame:
        if "vaccination_stage" in comp.columns:
            return comp
        # derive from infection_stage if present as suffix "_<vax>"
        # if no "_" present, fill as "NA"
        s = comp["infection_stage"].astype(str)
        v = np.where(s.str.contains("_"), s.str.split("_").str[-1], "NA")
        comp["vaccination_stage"] = v
        return comp

    @staticmethod
    def _ordered_ages_from_S(comp: pd.DataFrame, susceptible_stage: str) -> List[str]:
        mask = comp["infection_stage"].astype(str).str.startswith(susceptible_stage)
        ages = comp.loc[mask, "age_strata"].astype(str).tolist()
        # preserve first occurrence order
        return list(dict.fromkeys(ages).keys())

    # ------------------------
    # Build S/I for groups
    # ------------------------

    @staticmethod
    def _unique_group_labels(
        comp: pd.DataFrame,
        group_keys: Tuple[str, ...],
        susceptible_stage: str,
    ) -> List[Dict[str, Any]]:
        """Unique combinations of group_keys observed among S rows (stable order)."""
        mask_S = comp["infection_stage"].astype(str).str.startswith(susceptible_stage)
        df = comp.loc[mask_S, list(group_keys)].astype(object)
        # dropna for robustness; keep order of first occurrence
        seen = set()
        labels: List[Dict[str, Any]] = []
        for row in df.itertuples(index=False, name=None):
            if row not in seen:
                seen.add(row)
                labels.append(dict(zip(group_keys, row)))
        return labels

    @staticmethod
    def _indices_for_stage_and_label(
        comp: pd.DataFrame,
        stage_prefix: str,
        group_keys: Tuple[str, ...],
        label: Dict[str, Any],
    ) -> np.ndarray:
        s = comp["infection_stage"].astype(str)
        mask = s.str.startswith(stage_prefix)
        for k in group_keys:
            mask &= comp[k].astype(object) == label[k]
        return comp.index[mask].to_numpy()

    def _build_SI_by_groups(
        self,
        states: np.ndarray,  # (T,C,N)
        comp: pd.DataFrame,
        labels: List[Dict[str, Any]],
        infected_stages: Iterable[str],
        susceptible_stage: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return S_GTN, I_GTN, and age index per group.
        """
        T, C, N = states.shape
        ages = self._ordered_ages_from_S(comp, susceptible_stage)
        age_index = {a: i for i, a in enumerate(ages)}

        G = len(labels)
        S_GTN = np.zeros((G, T, N), dtype=states.dtype)
        I_GTN = np.zeros((G, T, N), dtype=states.dtype)
        age_for_group = np.empty(G, dtype=int)

        for g, lab in enumerate(labels):
            # track age for mapping age-specific rates
            a = str(lab.get("age_strata"))
            age_for_group[g] = age_index.get(a, 0)

            # S
            idx_S = self._indices_for_stage_and_label(
                comp, susceptible_stage, tuple(lab.keys()), lab
            )
            if idx_S.size:
                S_GTN[g] = states[:, idx_S, :].sum(axis=1)

            # I = sum over requested infected stage prefixes
            if infected_stages:
                acc = np.zeros((T, N), dtype=states.dtype)
                for stag in infected_stages:
                    idx_I = self._indices_for_stage_and_label(
                        comp, stag, tuple(lab.keys()), lab
                    )
                    if idx_I.size:
                        acc += states[:, idx_I, :].sum(axis=1)
                I_GTN[g] = acc

        return S_GTN, I_GTN, age_for_group

    # ------------------------
    # Rate per age
    # ------------------------

    @staticmethod
    def _param_slice_TN(params: np.ndarray, pidx: int, N: int) -> np.ndarray:
        arr = params[pidx]
        if arr.ndim == 1:
            return arr[:, None].repeat(N, axis=1)
        return arr

    def _rate_per_age(
        self,
        param_names: Sequence[str],
        params: np.ndarray,
        ages: Sequence[str],
    ) -> np.ndarray:
        """
        Build rate tensor per *age*: product of prefixes in self.rate_prefixes.
        - If only one name startswith prefix -> broadcast to all ages.
        - If multiple -> choose those containing the age token (first match per age).
        """
        names = np.asarray(param_names, dtype=str)
        T = params.shape[1]
        N = params.shape[2] if params.ndim == 3 else 1

        # Start with ones
        rate_ATN = np.ones((len(ages), T, N), dtype=float)

        for pref in self.rate_prefixes:
            hits = np.nonzero(np.char.startswith(names, pref))[0]
            if hits.size == 0:
                raise ValueError(f"No parameter starts with '{pref}'.")

            if hits.size == 1:
                ts = self._param_slice_TN(params, int(hits[0]), N)
                rate_ATN *= ts[None, :, :]
                continue

            # age-specific mapping: pick the one that contains the age token
            lower = [names[h].lower() for h in hits]
            for a_i, age in enumerate(ages):
                token = str(age).lower()
                chosen = None
                for j, nm in enumerate(lower):
                    if token in nm:
                        chosen = int(hits[j])
                        break
                if chosen is None:
                    # fallback: broadcast first hit
                    chosen = int(hits[0])
                ts = self._param_slice_TN(params, chosen, N)
                rate_ATN[a_i] *= ts

        return rate_ATN

    # ------------------------
    # Time handling & aggregation
    # ------------------------

    @staticmethod
    def _step_widths(times: Sequence) -> Tuple[np.ndarray, Any, bool]:
        """
        Return:
          dt: np.ndarray length T-1 of step widths
          left_edge_index: times[:-1] (float array OR DatetimeIndex)
          is_datetime: bool
        """
        if isinstance(times, pd.DatetimeIndex):
            tns = times.view("int64").astype(float)  # ns
            dt = np.diff(tns) / 1e9 / 86400.0
            if (dt <= 0).any():
                raise ValueError("`times` must be strictly increasing.")
            return dt, times[:-1], True

        arr = np.asarray(times, dtype=float)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("`times` must be 1-D with length >= 2.")
        dt = np.diff(arr)
        if (dt <= 0).any():
            raise ValueError("`times` must be strictly increasing.")
        return dt, arr[:-1], False

    @staticmethod
    def _aggregate_intervals(
        inc_interval: np.ndarray,  # (G, K, N)
        left_edges: Any,  # times[:-1] (float array OR DatetimeIndex)
        aggregate: Optional[float | str],
        is_datetime: bool,
    ) -> Tuple[np.ndarray, Any]:
        # No aggregation: return per-interval counts aligned to left edges
        if aggregate is None:
            return inc_interval, left_edges

        # numeric window (for numeric times)
        if isinstance(aggregate, (int, float)):
            if is_datetime:
                raise ValueError("Numeric `aggregate` requires numeric `times`.")
            window = float(aggregate)
            le = np.asarray(left_edges, dtype=float)
            t0 = float(le[0]) if le.size else 0.0
            gids = np.floor((le - t0) / window).astype(int)  # (K,)
            Gout = int(gids.max() + 1) if gids.size else 0
            G, K, N = inc_interval.shape
            out = np.zeros((G, Gout, N), dtype=float)
            for g in range(Gout):
                mask = gids == g
                if mask.any():
                    out[:, g, :] = inc_interval[:, mask, :].sum(axis=1)
            period_idx = t0 + np.arange(Gout) * window
            return out, period_idx

        # pandas offset (datetime)
        if isinstance(aggregate, str):
            if not is_datetime:
                raise ValueError("String `aggregate` requires datetime-like `times`.")
            le = pd.to_datetime(left_edges)
            keys = le.to_period(aggregate).to_timestamp()
            bins = pd.Index(keys).unique().sort_values()
            bin_map = {b: i for i, b in enumerate(bins)}
            gids = np.array([bin_map[k] for k in keys], dtype=int)

            G, K, N = inc_interval.shape
            out = np.zeros((G, len(bins), N), dtype=float)
            for i in range(len(bins)):
                mask = gids == i
                if mask.any():
                    out[:, i, :] = inc_interval[:, mask, :].sum(axis=1)
            return out, bins

        raise ValueError("Unsupported `aggregate` argument.")

    # ------------------------
    # Group aggregation helpers
    # ------------------------

    @staticmethod
    def _labels_to_frame(labels: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(labels)

    @staticmethod
    def _aggregate_groups(
        data_GTN: np.ndarray,  # (G, T, N)
        labels_full: List[Dict[str, Any]],
        group_keys: Tuple[str, ...],
        return_group_keys: Tuple[str, ...],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Sum across groups that share the same values for return_group_keys.
        """
        if tuple(return_group_keys) == tuple(group_keys):
            return data_GTN, labels_full

        df = VectorizedOutcomes._labels_to_frame(labels_full)
        # Build grouping key
        if len(return_group_keys) == 0:
            # aggregate everything into a single group
            gids = np.zeros(len(labels_full), dtype=int)
            out_labels = [dict()]
        else:
            key_tuples = [tuple(df[k].astype(object)) for k in return_group_keys]
            key_tuples = list(zip(*key_tuples))  # list of tuples per row
            uniq, inv = np.unique(
                np.array(key_tuples, dtype=object), return_inverse=True
            )
            gids = inv
            # reconstruct label dicts
            out_labels = []
            for ut in uniq:
                if not isinstance(ut, tuple):
                    ut = (ut,)
                lab = {k: v for k, v in zip(return_group_keys, ut)}
                out_labels.append(lab)

        Gout = int(gids.max() + 1) if len(gids) else 0
        G, T, N = data_GTN.shape
        out = np.zeros((Gout, T, N), dtype=data_GTN.dtype)
        for g in range(Gout):
            mask = gids == g
            if mask.any():
                out[g] = data_GTN[mask].sum(axis=0)
        return out, out_labels

    # ------------------------
    # Daily conversion & delay
    # ------------------------

    @staticmethod
    def _to_daily_counts(
        times: Sequence, inc_GTN: np.ndarray
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Convert per-interval counts to **daily** sums (summing into calendar-day bins).
        The per-interval counts are aligned to LEFT edges (times[:-1]).
        """
        # Build an index aligned to left edges (length K)
        if isinstance(times, pd.DatetimeIndex):
            left = times[:-1]
        else:
            t0 = pd.Timestamp("1970-01-01")
            left = t0 + pd.to_timedelta(np.asarray(times[:-1], dtype=float), unit="D")

        G, K, N = inc_GTN.shape
        df = pd.DataFrame(inc_GTN.reshape(G * N, K).T, index=left)
        daily = df.resample("D").sum()
        Td = daily.shape[0]
        daily_vals = (
            daily.to_numpy().T.reshape(G, N, Td).transpose(0, 2, 1)
        )  # (G, Td, N)
        return daily.index, daily_vals

    def _make_delay_kernel(
        self,
        *,
        delay_type: DelayType,
        mean: Optional[float],
        cv: Optional[float],
        shape: Optional[float],
        scale: Optional[float],
        keep_mass: float,
    ) -> np.ndarray:
        if delay_type == "constant":
            if mean is None:
                raise ValueError("constant delay requires `mean` (days).")
            shift = max(0, int(round(float(mean))))
            k = np.zeros(shift + 1, dtype=float)
            k[-1] = 1.0
            return k

        if delay_type == "exponential":
            if mean is None:
                raise ValueError("exponential delay requires `mean` (days).")
            lam = 1.0 / float(mean)
            # discrete daily mass: P(d <= t < d+1) for t~Exp(lam)
            # = exp(-lam*d) - exp(-lam*(d+1))
            L = int(np.ceil(-np.log(1 - keep_mass) / lam)) + 4
            grid = np.arange(L, dtype=float)
            w = np.exp(-lam * grid) - np.exp(-lam * (grid + 1))
            w /= w.sum()
            return w

        if delay_type == "gamma":
            if shape is None or scale is None:
                if mean is None or cv is None:
                    raise ValueError("gamma delay requires (shape,scale) or (mean,cv).")
                shape = 1.0 / (cv * cv)
                scale = float(mean) / shape
            dist = gamma_dist(a=shape, scale=scale)
            L = int(np.ceil(dist.ppf(keep_mass)))
            if L < 1:
                L = 1
            edges = np.arange(L + 1, dtype=float)
            cdf = dist.cdf(edges)
            w = np.diff(cdf)
            if w.sum() <= 0:
                w = np.array([1.0], dtype=float)
            w /= w.sum()
            return w

        raise ValueError(f"Unsupported delay_type: {delay_type}")

    def _convolve_daily(self, X_GTN: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolve along the time axis with either direct or FFT path.
        X_GTN: (G, Td, N)
        """
        G, T, N = X_GTN.shape
        L = kernel.size
        out = np.zeros_like(X_GTN, dtype=float)

        if not self.use_fft_convolution or (T + L) < self.fft_threshold:
            # direct
            for g in range(G):
                for n in range(N):
                    out[g, :, n] = np.convolve(X_GTN[g, :, n], kernel, mode="full")[:T]
            return out

        # FFT path (real FFT per (g,n) series)
        fft_len = int(2 ** int(np.ceil(np.log2(T + L - 1))))
        K = np.zeros(fft_len, dtype=float)
        K[:L] = kernel
        Kf = rfft(K)
        buf = np.zeros(fft_len, dtype=float)
        for g in range(G):
            for n in range(N):
                buf[:] = 0.0
                series = X_GTN[g, :, n]
                buf[:T] = series
                yf = rfft(buf)
                conv = irfft(yf * Kf, n=fft_len)[:T]
                out[g, :, n] = conv
        return out

    @staticmethod
    def _aggregate_daily(
        daily_GTN: np.ndarray,  # (G, Td, N)
        daily_index: pd.DatetimeIndex,
        aggregate: Optional[str | float],
    ) -> Tuple[np.ndarray, Any]:
        if aggregate is None or aggregate == "D":
            return daily_GTN, daily_index
        if isinstance(aggregate, (int, float)):
            raise ValueError(
                "Numeric aggregate not supported after daily conversion; use a pandas offset string."
            )
        # pandas resample path
        G, Td, N = daily_GTN.shape
        df = pd.DataFrame(
            daily_GTN.transpose(1, 0, 2).reshape(Td, G * N), index=daily_index
        )
        agg = df.resample(aggregate).sum()
        T2 = agg.shape[0]
        arr = agg.to_numpy().reshape(T2, G, N).transpose(1, 0, 2)
        return arr, agg.index

    # ------------------------
    # Probability resolution
    # ------------------------

    @staticmethod
    def _resolve_probabilities(
        prob_spec: Any,
        labels_full: List[Dict[str, Any]],
        group_keys: Tuple[str, ...],
    ) -> np.ndarray:
        """
        Produce a length-G vector of probabilities aligned with labels_full.

        Accepted forms:
          - scalar
          - array-like with length == G
          - array-like with length == #unique ages (if 'age_strata' in group_keys)
          - dict with keys:
              * tuple in the exact order of group_keys, e.g., ('age','vax')
              * tuple of just ('age',) -> fallback if exact combo missing
              * string age -> fallback
              * 'default' -> final fallback
              * OR dict keys: {'age_strata':'A','vaccination_stage':'waned'}
        """
        G = len(labels_full)

        # scalar
        if np.isscalar(prob_spec):
            p = float(prob_spec)
            return np.full(G, p, dtype=float)

        # dict-based flexible mapping (check **before** coercing to array)
        if isinstance(prob_spec, Mapping):

            def fetch(lab: Dict[str, Any]) -> float:
                tup_exact = tuple(lab[k] for k in group_keys)
                if tup_exact in prob_spec:
                    return float(prob_spec[tup_exact])

                # dict-key match
                for k in prob_spec.keys():
                    if isinstance(k, dict):
                        if all(lab.get(kk) == kv for kk, kv in k.items()):
                            return float(prob_spec[k])

                # age-only fallbacks
                a = lab.get("age_strata", None)
                if a is not None:
                    if (a,) in prob_spec:
                        return float(prob_spec[(a,)])
                    if a in prob_spec:
                        return float(prob_spec[a])

                if "default" in prob_spec:
                    return float(prob_spec["default"])
                raise KeyError(
                    f"No probability found for group {lab} and no suitable fallback."
                )

            return np.array([fetch(lab) for lab in labels_full], dtype=float)

        # array-like length == G
        arr = np.asarray(prob_spec, dtype=float)
        if arr.ndim == 1 and arr.size == G:
            return arr

        # array-like by age-only if available
        ages = [lab.get("age_strata", None) for lab in labels_full]
        if all(a is not None for a in ages):
            uniq_ages = list(dict.fromkeys(ages))
            if arr.size == len(uniq_ages):
                age_to_p = {age: float(v) for age, v in zip(uniq_ages, arr)}
                return np.array(
                    [age_to_p[lab["age_strata"]] for lab in labels_full], dtype=float
                )

        raise ValueError(
            "Unrecognized prob_spec shape or keys; supply scalar, length-G array, age-length array, or a dict keyed by group labels."
        )
