# src/gempyor/vectorized_modifiers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Union
import datetime as dt
import numpy as np
import numpy.typing as npt

# ---------- Small helpers ----------

def _parse_date(s: Union[str, dt.date]) -> dt.date:
    if isinstance(s, dt.date):
        return s
    return dt.date.fromisoformat(str(s))

def _clamp(a: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, a))

def _subpop_to_indices(spec: Union[str, Iterable[int], Iterable[str]],
                       n_loc: int,
                       name_to_idx: dict[str, int] | None = None) -> npt.NDArray[np.int64]:
    """
    Turn subpop spec into integer indices.
    - "all" -> all locations
    - sequence[int] -> used as-is (validated and deduped)
    - sequence[str] -> resolved via name_to_idx (required)
    """
    if isinstance(spec, str):
        if spec.lower() == "all":
            return np.arange(n_loc, dtype=np.int64)
        # else treat as single named subpop
        if name_to_idx is None:
            raise ValueError("subpop names given but name_to_idx is None")
        return np.array([name_to_idx[spec]], dtype=np.int64)

    # iterable provided
    items = list(spec)
    if not items:
        return np.zeros(0, dtype=np.int64)

    if isinstance(items[0], str):
        if name_to_idx is None:
            raise ValueError("subpop names given but name_to_idx is None")
        idx = [name_to_idx[x] for x in items]
    else:
        idx = [int(x) for x in items]

    idx = np.array(sorted(set(idx)), dtype=np.int64)
    if idx.min(initial=0) < 0 or idx.max(initial=-1) >= n_loc:
        raise ValueError(f"subpop indices out of bounds for n_loc={n_loc}: {idx}")
    return idx

def _extract_scalar_value(block, default: float = 1.0) -> float:
    """
    Normalize common YAML shapes for a numeric modifier into a float.

    Accepts:
      - number
      - {"value": number}
      - {"distribution": "...", "value": number}
      - {"value": {"distribution": "...", "value": number}}
    """
    if block is None:
        return float(default)
    # direct scalar
    if isinstance(block, (int, float, np.floating)):
        return float(block)
    # dict-like
    if isinstance(block, dict):
        if "value" in block:
            return _extract_scalar_value(block["value"], default=default)
        if "distribution" in block and "value" in block:
            return _extract_scalar_value(block["value"], default=default)
    # last resort
    try:
        return float(block)
    except Exception:
        return float(default)

# ---------- Data structures ----------

@dataclass(frozen=True)
class SinglePeriodSpec:
    name: str                # e.g., "Seas_dec"
    parameter: str           # e.g., "r0"
    subpop_idx: npt.NDArray[np.int64]   # (L_sel,)
    day0: int                # inclusive day index in [0, T-1]
    day1: int                # inclusive day index in [0, T-1]
    base_value: float        # default multiplier from config (value.value)

@dataclass(frozen=True)
class StackedSpec:
    name: str
    children: tuple[str, ...]   # names of leaf or other stacked

@dataclass
class CompiledModifiers:
    # All leaves by name
    leaves: dict[str, SinglePeriodSpec]
    # Stacks by name
    stacks: dict[str, StackedSpec]
    # Scenario name -> list root modifiers (usually a single stack like "none")
    scenarios: dict[str, tuple[str, ...]]
    # Deterministic traversal order (leaf/stacks) for reproducible overrides
    order_leaves: tuple[str, ...]
    order_stacks: tuple[str, ...]

# ---------- Compiler ----------

def compile_seir_modifiers(
    seir_modifiers_cfg: dict,
    *,
    start_date: dt.date,
    n_days: int,
    n_loc: int,
    param_names: Iterable[str],
    subpop_name_to_idx: dict[str, int] | None = None,
) -> "ModifierApplier":
    """
    Compile the seir_modifiers block into an efficient applier.

    Parameters
    ----------
    seir_modifiers_cfg : dict
        The 'seir_modifiers' section from the config.
    start_date : date
        Model start date (day 0).
    n_days : int
        Number of model days (length of time axis).
    n_loc : int
        Number of locations.
    param_names : iterable of str
        Parameter names in the parameter tensor's first axis order (used for mapping).
    subpop_name_to_idx : dict[str,int], optional
        If provided, resolves subpop names; otherwise subpop must be "all" or integer indices.

    Returns
    -------
    ModifierApplier
        Object with .apply_to_params(...)
    """
    # Pull sections
    mods_cfg = seir_modifiers_cfg.get("modifiers", {})
    scen_list = seir_modifiers_cfg.get("scenarios", []) or []
    # normalize scenarios into dict; if it's a list, we still need their definitions in 'modifiers'
    scenarios: dict[str, tuple[str, ...]] = {}
    for s in scen_list:
        # The scenario name must exist under modifiers as a StackedModifier
        scenarios[str(s)] = (str(s),)

    # Build leaf specs
    leaves: dict[str, SinglePeriodSpec] = {}
    stacks: dict[str, StackedSpec] = {}

    # Helper: convert date to day index (inclusive)
    def _to_day_idx(d: Union[str, dt.date]) -> int:
        dd = _parse_date(d)
        return (dd - start_date).days

    for name, spec in mods_cfg.items():
        m = str(spec.get("method", "")).strip()
        if m == "SinglePeriodModifier":
            param = str(spec["parameter"])
            # periods are inclusive; clamp to [0, n_days-1]
            d0 = _clamp(_to_day_idx(spec["period_start_date"]), 0, n_days - 1)
            d1 = _clamp(_to_day_idx(spec["period_end_date"]),   0, n_days - 1)
            if d1 < d0:
                # out of horizon -> skip
                continue

            subpop_spec = spec.get("subpop", "all")
            sub_idx = _subpop_to_indices(subpop_spec, n_loc, subpop_name_to_idx)

            # robust extraction (handles scalar or nested dicts)
            base_val = _extract_scalar_value(spec.get("value"), default=1.0)

            leaves[name] = SinglePeriodSpec(
                name=name,
                parameter=param,
                subpop_idx=sub_idx,
                day0=int(d0),
                day1=int(d1),
                base_value=base_val,
            )

        elif m == "StackedModifier":
            children = tuple(spec.get("modifiers", []) or [])
            stacks[name] = StackedSpec(name=name, children=children)

        else:
            # Unknown method or scenario alias (which should be a StackedModifier defined above)
            continue

    # Allow direct use of stacked modifier names as scenarios later
    for name, sp in mods_cfg.items():
        if str(sp.get("method", "")) == "StackedModifier" and name not in scenarios:
            # not adding here to scenarios dict to avoid changing semantics,
            # but users may pass scenario=name to apply it directly.
            pass

    order_leaves = tuple(leaves.keys())
    order_stacks = tuple(stacks.keys())

    compiled = CompiledModifiers(
        leaves=leaves,
        stacks=stacks,
        scenarios=scenarios,
        order_leaves=order_leaves,
        order_stacks=order_stacks,
    )

    # param mapping for (P,T,L) tensors
    param_name_to_idx = {name: i for i, name in enumerate(param_names)}

    return ModifierApplier(compiled, start_date, n_days, n_loc, param_name_to_idx)


# ---------- Applier ----------

class ModifierApplier:
    """
    Applies compiled SEIR modifiers to parameter arrays.
    """

    def __init__(self,
                 compiled: CompiledModifiers,
                 start_date: dt.date,
                 n_days: int,
                 n_loc: int,
                 param_name_to_idx: dict[str, int]):
        self.compiled = compiled
        self.start_date = start_date
        self.n_days = int(n_days)
        self.n_loc = int(n_loc)
        self.param_name_to_idx = dict(param_name_to_idx)

    # ---- public API ----

    def list_leaf_modifiers(self) -> tuple[str, ...]:
        """Deterministic order of leaf modifiers (for array overrides)."""
        return self.compiled.order_leaves

    def list_stack_modifiers(self) -> tuple[str, ...]:
        return self.compiled.order_stacks

    def list_scenarios(self) -> tuple[str, ...]:
        """Scenario names declared under seir_modifiers.scenarios (or empty)."""
        return tuple(self.compiled.scenarios.keys())

    def apply_to_params(
        self,
        params: Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]],
        *,
        scenario: str | None = None,
        # Override values for leaf modifiers; either dict by name, or array aligned to list_leaf_modifiers()
        leaf_value_overrides: dict[str, float] | None = None,
        leaf_value_array: npt.NDArray[np.float64] | None = None,
        # If True, also return the composite scales used per parameter
        return_scales: bool = False,
    ) -> Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]], tuple[Union[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]], dict[str, npt.NDArray[np.float64]]]]:
        """
        Apply modifiers (for the given scenario or explicit set) to the parameter arrays.

        params:
          - Either a tensor of shape (P,T,L) with parameters in self.param_name_to_idx order
          - Or a dict {param_name: (T,L)}. This is convenient if you only carry the few that get modified.

        Overlapping modifiers multiply. Non-mentioned parameters are untouched.

        Returns updated params (same type/shape), and optionally a dict of scales per-parameter (T,L).
        """
        scales_by_param = self._build_composite_scales(
            scenario=scenario,
            leaf_value_overrides=leaf_value_overrides,
            leaf_value_array=leaf_value_array,
        )

        # Apply to params
        if isinstance(params, dict):
            out: dict[str, npt.NDArray[np.float64]] = {}
            for pname, arr in params.items():
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape != (self.n_days, self.n_loc):
                    raise ValueError(f"param '{pname}' must be shape (T,L)=({self.n_days},{self.n_loc}), got {arr.shape}")
                scale = scales_by_param.get(pname, None)
                if scale is not None:
                    out[pname] = arr * scale
                else:
                    out[pname] = arr.copy()
            return (out, scales_by_param) if return_scales else out

        # 3D tensor path
        tensor = np.asarray(params, dtype=np.float64)
        if tensor.ndim != 3 or tensor.shape[1] != self.n_days or tensor.shape[2] != self.n_loc:
            raise ValueError(f"params tensor must be (P,T,L) with T={self.n_days}, L={self.n_loc}, got {tensor.shape}")

        out_tensor = tensor.copy()
        for pname, scale in scales_by_param.items():
            pidx = self.param_name_to_idx.get(pname, None)
            if pidx is None:
                # Parameter not present in this tensor—ignore
                continue
            out_tensor[pidx, :, :] *= scale
        return (out_tensor, scales_by_param) if return_scales else out_tensor

    # ---- internals ----

    def _leaf_value(self,
                    name: str,
                    overrides: dict[str, float] | None,
                    array_override:npt.NDArray[np.float64] | None) -> float:
        leaf = self.compiled.leaves[name]
        if overrides and name in overrides:
            return float(overrides[name])
        if array_override is not None:
            order = self.compiled.order_leaves
            if array_override.shape != (len(order),):
                raise ValueError(f"leaf_value_array must be shape ({len(order)},)")
            i = order.index(name)
            return float(array_override[i])
        return float(leaf.base_value)

    def _collect_active_roots(self, scenario: str | None) -> tuple[str, ...]:
        """
        Return the set of top-level modifiers to apply.
        If scenario is provided and exists in config 'scenarios', we resolve it to its root(s).
        Otherwise, if scenario names a StackedModifier directly, use it.
        If None, we try 'none' if available; else apply all stacks defined.
        """
        if scenario is not None:
            # 1) scenario declared under scenarios
            if scenario in self.compiled.scenarios:
                return self.compiled.scenarios[scenario]
            # 2) direct use of a stacked modifier as a scenario name
            if scenario in self.compiled.stacks:
                return (scenario,)
            # 3) direct use of a leaf (rare but allowed)
            if scenario in self.compiled.leaves:
                return (scenario,)
            raise ValueError(f"Unknown scenario/modifier: {scenario}")

        # Default preference
        if "none" in self.compiled.scenarios:
            return self.compiled.scenarios["none"]
        if "none" in self.compiled.stacks:
            return ("none",)
        # else: apply all stacks (unlikely desired, but keeps behavior defined)
        if self.compiled.stacks:
            return tuple(self.compiled.stacks.keys())
        # else: if only leaves exist, apply all leaves
        return tuple(self.compiled.leaves.keys())

    def _expand_stack(self, root: str) -> list[str]:
        """
        Depth-first expand a stack into a flat list of leaf names and/or other stacks,
        but we only *apply* leaf names (stacks are just containers).
        """
        if root in self.compiled.leaves:
            return [root]
        if root not in self.compiled.stacks:
            # Unknown; ignore silently to be forgiving
            return []
        out: list[str] = []
        for child in self.compiled.stacks[root].children:
            out.extend(self._expand_stack(child))
        return out

    def _build_composite_scales(
        self,
        *,
        scenario: str | None,
        leaf_value_overrides: dict[str, float] | None,
        leaf_value_array: npt.NDArray[np.float64] | None,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Produce per-parameter scale arrays (T,L) with all applicable leaves multiplied.
        """
        # Initialize param-wise scale to ones lazily
        scales_by_param: dict[str, npt.NDArray[np.float64]] = {}

        # Which roots are active?
        roots = self._collect_active_roots(scenario)
        # Expand to leaves
        leaves_to_apply: list[str] = []
        for r in roots:
            leaves_to_apply.extend(self._expand_stack(r))
        # Dedupe while keeping order
        seen = set()
        ordered_leaves = []
        for nm in leaves_to_apply:
            if nm not in seen:
                seen.add(nm)
                ordered_leaves.append(nm)

        # Apply each leaf
        for leaf_name in ordered_leaves:
            leaf = self.compiled.leaves.get(leaf_name, None)
            if leaf is None:
                continue

            val = self._leaf_value(leaf_name, leaf_value_overrides, leaf_value_array)

            # Skip identity multipliers to avoid touching arrays
            if val == 1.0:
                continue

            # Get or create the (T,L) scale for this parameter
            scale = scales_by_param.get(leaf.parameter)
            if scale is None:
                scale = np.ones((self.n_days, self.n_loc), dtype=np.float64)
                scales_by_param[leaf.parameter] = scale

            # Apply multiplicative factor on slice: days [d0..d1], locations subpop_idx
            d0, d1 = leaf.day0, leaf.day1
            loc = leaf.subpop_idx
            # Use a single advanced index to avoid creating a temporary copy
            scale[d0:d1+1, loc] *= val

        return scales_by_param
