# tests/vectorized_inference/test_outcomes_parsing.py
# Requires env:
#   PROJECT_PATH=/abs/path/to/Flu_USA
#   CONFIG_PATH=smh_calibrate_season_2012.yml

from __future__ import annotations

import os
import platform
from ctypes.util import find_library

# ---- threading env (set before imports that touch BLAS/Numba) ----
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYMC_PROGRESSBAR", "1")

def _choose_numba_layer() -> str:
    forced = os.environ.get("NUMBA_THREADING_LAYER")
    if forced:
        return forced
    if find_library("tbb") or find_library("tbb12"):
        return "tbb"
    if find_library("omp") or find_library("gomp"):
        return "omp"
    return "workqueue"

def _maybe_patch_dylib_path(layer: str) -> None:
    if platform.system() != "Darwin":
        return
    extra = []
    if layer == "tbb":
        extra = ["/opt/homebrew/opt/tbb/lib", "/usr/local/opt/tbb/lib"]
    elif layer == "omp":
        extra = ["/opt/homebrew/opt/libomp/lib", "/usr/local/opt/libomp/lib"]
    if not extra:
        return
    cur = os.environ.get("DYLD_LIBRARY_PATH", "")
    add = [p for p in extra if os.path.exists(p) and (p not in cur)]
    if add:
        os.environ["DYLD_LIBRARY_PATH"] = (":".join(add) + (":" + cur if cur else ""))

_layer = _choose_numba_layer()
_maybe_patch_dylib_path(_layer)
os.environ.setdefault("NUMBA_THREADING_LAYER", _layer)
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(4, max(1, os.cpu_count() or 1))))


import os
from pathlib import Path
import copy
import re
import json

import numpy as np
import pytest
import confuse
import yaml


from gempyor.hosp_weekly_pipeline import (  # type: ignore
    WeeklyHospPipeline,
    _compile_outcomes_from_config,
)

# ---------- helpers & fixtures ----------

def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        pytest.skip(
            f"Set {name}. Example:\n"
            f"  PROJECT_PATH=/.../Flu_USA\n"
            f"  CONFIG_PATH=smh_calibrate_season_2012.yml"
        )
    return v


@pytest.fixture(autouse=True)
def chdir_project_root(monkeypatch):
    """
    Function-scoped (works with function-scoped monkeypatch).
    Ensures relative YAML paths like model_input/... resolve just like your normal runs.
    """
    proj = Path(_require_env("PROJECT_PATH")).expanduser().resolve()
    if not proj.exists():
        pytest.skip(f"PROJECT_PATH does not exist: {proj}")
    monkeypatch.chdir(proj)


def _cfg_path_or_skip() -> Path:
    proj = Path(_require_env("PROJECT_PATH")).expanduser().resolve()
    conf = _require_env("CONFIG_PATH")
    cfg_path = (proj / conf).resolve()
    if not cfg_path.exists():
        pytest.skip(f"CONFIG_PATH not found: {cfg_path}")
    return cfg_path


def _load_confuse(cfg_path: Path) -> confuse.Configuration:
    c = confuse.Configuration("FluUSA_TestOutcomes_Pipeline", __name__)
    c.set_file(str(cfg_path))
    return c


def _yaml_prob(spec) -> float:
    """Mirror the probability extraction used by the pipeline helpers."""
    if spec is None:
        return 1.0
    if isinstance(spec, (int, float)):
        return float(spec)
    v = spec.get("value", None)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        inner = v.get("value", v.get("val", v.get("mult", None)))
        if isinstance(inner, (int, float)):
            return float(inner)
    return 1.0


def _build_pipeline(cfg_path: Path, *, dt_days: float = 1.0) -> WeeklyHospPipeline:
    # Build the pipeline (this invokes ModelInfo under-the-hood with the proper path_prefix)
    return WeeklyHospPipeline(str(cfg_path), dt_days=dt_days)


# ---------- tests ----------

def test_outcomes_tree_structure():
    cfg_path = _cfg_path_or_skip()
    pipe = _build_pipeline(cfg_path)

    # Pull the already-compiled maps from the pipeline
    resolve_map = pipe._out_resolve_rows
    prob_map = pipe._out_prob_map
    delay_steps = pipe._out_delay_steps
    sum_map = pipe._out_sum_map
    order = pipe._out_all_names

    assert isinstance(order, list) and len(order) > 0
    keys = set(order)
    assert set(resolve_map) >= keys or set(resolve_map) == keys  # tolerate extras if any
    assert set(prob_map) >= keys
    assert set(delay_steps) >= keys

    # All delays non-negative integers
    assert all(isinstance(v, int) and v >= 0 for v in delay_steps.values())

    # Leaves (those with concrete transition rows) must not be sum nodes
    leaf_names = [nm for nm, rows in resolve_map.items() if isinstance(rows, np.ndarray) and rows.size > 0]
    for nm in leaf_names:
        assert nm not in sum_map or len(sum_map.get(nm, [])) == 0

    # Spot check your structure:
    # incidH is a sum that should include incidH_AllFlu, which itself is a sum of leaf age×vax nodes
    if "incidH" in sum_map:
        assert "incidH_AllFlu" in sum_map["incidH"]
    if "incidH_AllFlu" in sum_map:
        kids = sum_map["incidH_AllFlu"]
        assert isinstance(kids, list) and len(kids) >= 3
        # its own resolve rows should be empty (pure sum/alias)
        assert pipe._out_resolve_rows["incidH_AllFlu"].size == 0


def test_probability_injection_matches_yaml():
    cfg_path = _cfg_path_or_skip()
    conf = _load_confuse(cfg_path)
    outcomes_cfg = conf["outcomes"]["outcomes"].get()

    pipe = _build_pipeline(cfg_path)

    # Compare each *leaf* probability between YAML and the pipeline’s compiled prob_map
    mismatches = []
    NL = pipe.NL
    for name, rows in pipe._out_resolve_rows.items():
        if isinstance(rows, np.ndarray) and rows.size > 0:
            spec = outcomes_cfg.get(name, {})
            p_yaml = _yaml_prob(spec.get("probability"))
            p_vec = pipe._out_prob_map.get(name, None)
            if p_vec is None or not isinstance(p_vec, np.ndarray) or p_vec.shape != (NL,):
                mismatches.append((name, f"prob_map missing or wrong shape {getattr(p_vec, 'shape', None)}"))
                continue
            if not np.allclose(p_vec, p_yaml, rtol=0, atol=0):
                mismatches.append((name, f"expected {p_yaml}, got first entries {p_vec[:3]}"))

    assert not mismatches, "Probability mismatches:\n" + "\n".join(f"  {n}: {m}" for n, m in mismatches)


def test_tweak_yaml_probability_is_reflected(tmp_path: Path):
    """
    Write a patched copy of the YAML *in the same directory as the original* so relative
    paths (model_input/...) remain valid, then confirm the pipeline picks up the new prob.
    """
    cfg_path = _cfg_path_or_skip()
    conf = _load_confuse(cfg_path)
    outcomes_cfg = conf["outcomes"]["outcomes"].get()

    # Choose a leaf to tweak
    # Prefer one of the incidH_(unvaccinated|waned|1dose)_* leaves.
    def _pick_leaf_name():
        # Build a tiny “probe” pipeline to get resolve rows
        p = _build_pipeline(cfg_path)
        leafs = [nm for nm, rows in p._out_resolve_rows.items() if isinstance(rows, np.ndarray) and rows.size > 0]
        for pat in (r"^incidH_unvaccinated_.*", r"^incidH_waned_.*", r"^incidH_1dose_.*"):
            for nm in leafs:
                if re.match(pat, nm):
                    return nm
        return leafs[0] if leafs else None

    target_leaf = _pick_leaf_name()
    if not target_leaf:
        pytest.skip("No leaf outcomes found to tweak in this config.")

    # Load YAML text and patch that one node’s probability
    text = cfg_path.read_text()
    y = yaml.safe_load(text)
    node = y["outcomes"]["outcomes"][target_leaf]
    old_p = _yaml_prob(node.get("probability"))
    new_p = float(min(old_p * 1.25, 0.5))
    if "probability" not in node or not isinstance(node["probability"], dict):
        node["probability"] = {"value": {"distribution": "fixed", "value": new_p}}
    else:
        if "value" not in node["probability"] or not isinstance(node["probability"]["value"], dict):
            node["probability"]["value"] = {"distribution": "fixed", "value": new_p}
        else:
            node["probability"]["value"]["value"] = new_p

    # Write a neighbor file next to the original so relative paths still work
    patched = cfg_path.with_name(cfg_path.stem + "__pytest_patch.yml")
    patched.write_text(yaml.safe_dump(y, sort_keys=False))

    try:
        pipe0 = _build_pipeline(cfg_path)
        pipe1 = _build_pipeline(patched)

        NL = pipe0.NL
        assert target_leaf in pipe0._out_prob_map and target_leaf in pipe1._out_prob_map

        p0 = pipe0._out_prob_map[target_leaf]
        p1 = pipe1._out_prob_map[target_leaf]
        assert p0.shape == (NL,) and p1.shape == (NL,)

        # The patched probability should propagate into the compiled map
        assert np.allclose(p1, new_p)
        # And it should differ from the original unless old==new by clipping
        if not np.isclose(old_p, new_p):
            assert not np.allclose(p0, p1)
    finally:
        # Clean up the patched file
        try:
            patched.unlink()
        except Exception:
            pass


def test_delays_nonnegative_and_reasonable():
    cfg_path = _cfg_path_or_skip()
    pipe = _build_pipeline(cfg_path)

    delay_steps = pipe._out_delay_steps
    assert delay_steps, "No delays parsed"

    # Reasonable upper bound: twice the number of solver steps in the horizon
    T_days = int(pipe.T)
    steps = int(np.ceil(T_days / max(pipe.dt, 1e-6)))
    max_ok = 2 * steps

    for name, d in delay_steps.items():
        assert isinstance(d, int) and d >= 0, f"{name} has invalid delay {d}"
        assert d <= max_ok, f"{name} delay {d} too large for horizon {T_days}d (dt={pipe.dt})"


def test_every_sum_eventually_resolves_to_leaves():
    cfg_path = _cfg_path_or_skip()
    pipe = _build_pipeline(cfg_path)

    resolve_map = pipe._out_resolve_rows
    sum_map = pipe._out_sum_map
    order = pipe._out_all_names

    visiting = set()
    memo: dict[str, bool] = {}

    def _has_leaf(name: str) -> bool:
        if name in memo:
            return memo[name]
        rows = resolve_map.get(name, np.array([], dtype=np.int64))
        if isinstance(rows, np.ndarray) and rows.size > 0:
            memo[name] = True
            return True
        kids = sum_map.get(name, [])
        if not kids:
            memo[name] = False
            return False
        visiting.add(name)
        ok = False
        for k in kids:
            if k in visiting:  # cycle guard
                ok = False
                break
            if _has_leaf(k):
                ok = True
        visiting.discard(name)
        memo[name] = ok
        return ok

    failures = [nm for nm in order if nm in sum_map and not _has_leaf(nm)]
    assert not failures, f"Sum nodes without any leaf descendants: {failures}"
