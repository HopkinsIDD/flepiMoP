# tests/modifiers/test_vectorized_modifiers_r0.py
import os
# keep BLAS runtimes from fighting with anything else
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import shutil
import datetime as dt

import numpy as np
import pytest
import confuse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gempyor.model_info import ModelInfo
from gempyor.vectorized_modifiers import compile_seir_modifiers


# ----------------------------- helpers ---------------------------------
def _load_structured_example(tmp_path_factory):
    """Load examples/tutorials/Structured_Example.yml into a temp root with absolute paths."""
    tmp_root = tmp_path_factory.mktemp("modifiers_case")

    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

    # Inputs
    src_struct = tutorial_dir / "model_input" / "Structured_Example"
    dst_struct = tmp_root / "model_input" / "Structured_Example"
    shutil.copytree(src_struct, dst_struct, dirs_exist_ok=True)

    src_ic = tutorial_dir / "model_input" / "initial_condition"
    dst_ic = tmp_root / "model_input" / "initial_condition"
    shutil.copytree(src_ic, dst_ic, dirs_exist_ok=True)

    # Config
    cfg_name = "Structured_Example.yml"
    cfg_path = tmp_root / cfg_name
    shutil.copyfile(tutorial_dir / cfg_name, cfg_path)

    # Patch relative -> absolute
    text = cfg_path.read_text()
    text = text.replace("model_input/", str(tmp_root / "model_input") + "/")
    cfg_path.write_text(text)

    conf = confuse.Configuration("TestModel", __name__)
    conf.set_file(str(cfg_path))

    model = ModelInfo(
        config=conf,
        config_filepath=str(cfg_path),
        path_prefix=str(tmp_root),
        setup_name="Structured_Example",
        seir_modifiers_scenario="none",
    )

    # Build parameter cube
    _ = model.initial_conditions.get_from_config(sim_id=0, modinf=model)
    unique_strings, transitions, transition_sum_compartments, proportion_info = (
        model.compartments.get_transition_array()
    )

    param_defs = conf["seir"]["parameters"].get()
    base_params = model.parameters.parameters_quick_draw(model.n_days, model.nsubpops)
    params = model.compartments.parse_parameters(base_params, param_defs, unique_strings)  # (P,T,L)

    # parameter axis names/order come from param_defs
    param_names = np.array(list(param_defs.keys()))
    param_name_to_row = {name: i for i, name in enumerate(param_names)}

    # (kept for completeness, but unused by the tests now)
    unique_name_to_row = {name: i for i, name in enumerate(unique_strings)}

    return {
        "model": model,
        "config": conf,
        "params": params,                      # (P,T,L)
        "param_names": param_names,            # parameter names in P axis order
        "param_name_to_row": param_name_to_row,
        "unique_strings": unique_strings,      # transition/compartment strings
        "unique_name_to_row": unique_name_to_row,
        "n_days": model.n_days,
        "n_locs": model.nsubpops,
        "start_date": model.ti,
        "end_date": model.tf,
    }


def _factor_from_yaml_for_param(config, start_date: dt.date, n_days: int, param: str) -> np.ndarray:
    """
    Independent ground-truth: build a daily multiplicative factor timeline for `param`
    from seir_modifiers YAML (scenario 'none'). Overlaps multiply; subpop handling is
    ignored here (we’ll broadcast across locations in the test — Structured_Example
    uses 'all' anyway).
    """
    sm = config["seir_modifiers"].get()
    modifiers = sm["modifiers"]

    # recursively flatten a stack into leaf names
    def flatten(name):
        spec = modifiers[name]
        if spec.get("method") == "StackedModifier":
            out = []
            for child in spec.get("modifiers", []) or []:
                out.extend(flatten(child))
            return out
        return [name]

    roots_cfg = sm.get("scenarios", ["none"]) or ["none"]
    roots = ["none"] if "none" in roots_cfg else list(roots_cfg)
    leaves = []
    for r in roots:
        # scenario name may also be a StackedModifier at the same level
        if r in modifiers and modifiers[r].get("method") == "StackedModifier":
            leaves.extend(flatten(r))
        else:
            # fallback: treat as a single leaf if given that way
            leaves.append(r)

    factor = np.ones(n_days, dtype=np.float64)

    for nm in leaves:
        spec = modifiers.get(nm)
        if not spec or spec.get("method") != "SinglePeriodModifier":
            continue
        if str(spec.get("parameter")).lower() != param.lower():
            continue

        # dates
        d0 = dt.date.fromisoformat(str(spec["period_start_date"]))
        d1 = dt.date.fromisoformat(str(spec["period_end_date"]))
        # robust nested access for value.value.value OR value.value OR value
        v = spec.get("value", {})
        if isinstance(v, dict):
            v1 = v.get("value", v)
            if isinstance(v1, dict):
                mult = float(v1.get("value", 1.0))
            else:
                mult = float(v1)
        else:
            mult = float(v)

        # apply inclusive
        left = max(d0, start_date)
        right = min(d1, start_date + dt.timedelta(days=n_days - 1))
        if right < left:
            continue
        i0 = (left - start_date).days
        i1 = (right - start_date).days
        factor[i0:i1 + 1] *= mult

    return factor


def _param_lookup(param: str, param_names: np.ndarray):
    if param in param_names:
        return int(np.where(param_names == param)[0][0])
    return None


# ------------------------------ tests -----------------------------------
@pytest.mark.slow
def test_apply_r0_modifiers_matches_yaml(tmp_path_factory):
    """
    Correctness: applying compile_seir_modifiers().apply_to_params() for scenario='none'
    must match an independently computed factor timeline from the YAML.
    """
    case = _load_structured_example(tmp_path_factory)
    config = case["config"]
    params = case["params"]
    param_names = case["param_names"]
    param_name_to_row = case["param_name_to_row"]
    T, L = case["n_days"], case["n_locs"]

    # compile applier
    applier = compile_seir_modifiers(
        seir_modifiers_cfg=config["seir_modifiers"].get(),
        start_date=case["start_date"],
        n_days=T,
        n_loc=L,
        param_names=param_names,
        subpop_name_to_idx=None,  # YAML uses "all"
    )

    # apply to the full tensor (P,T,L)
    out_tensor = applier.apply_to_params(params, scenario="none")
    assert out_tensor.shape == params.shape

    # pull baseline and modified r0 using parameter axis names
    r0_idx = _param_lookup("r0", param_names)
    assert r0_idx is not None, "Expected 'r0' in param_names from config['seir']['parameters']"
    baseline = params[r0_idx, :, :]           # (T,L)
    modified = out_tensor[r0_idx, :, :]       # (T,L)

    # ground-truth factor(t) from YAML (broadcast across locations)
    f = _factor_from_yaml_for_param(config, case["start_date"], T, "r0")  # (T,)
    expected = baseline * f[:, None]

    np.testing.assert_allclose(modified, expected, rtol=1e-12, atol=0.0)


@pytest.mark.slow
def test_plot_r0_panel_first4(tmp_path_factory):
    """
    Visualization: save a 2x2 panel plot of baseline r0(t) and modified r0(t) for first 4 locations.
    """
    case = _load_structured_example(tmp_path_factory)
    config = case["config"]
    params = case["params"]
    param_names = case["param_names"]
    T, L = case["n_days"], case["n_locs"]

    applier = compile_seir_modifiers(
        seir_modifiers_cfg=config["seir_modifiers"].get(),
        start_date=case["start_date"],
        n_days=T,
        n_loc=L,
        param_names=param_names,
        subpop_name_to_idx=None,
    )
    out_tensor = applier.apply_to_params(params, scenario="none")

    r0_idx = _param_lookup("r0", param_names)
    assert r0_idx is not None, "Expected 'r0' in param_names"
    baseline = params[r0_idx, :, :]           # (T,L)
    modified = out_tensor[r0_idx, :, :]       # (T,L)

    # figure path in the same directory as this test file
    outdir = Path(__file__).parent
    outpath = outdir / "r0_panel_first4.png"

    K = min(4, L)
    locs = list(range(K))
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), dpi=120, sharex=True)
    axes = axes.ravel()
    for i, loc in enumerate(locs):
        ax = axes[i]
        ax.plot(t, baseline[:, loc], label="baseline r0", linewidth=1.4)
        ax.plot(t, modified[:, loc], label="modified r0", linestyle="--", linewidth=1.4)
        ax.set_title(f"Location {loc}")
        ax.grid(True, alpha=0.3)
        if i in (2, 3):
            ax.set_xlabel("day")
        if i in (0, 2):
            ax.set_ylabel("r0")
    # handle fewer than 4 locations gracefully
    for j in range(K, 4):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Baseline vs Modified r0(t) — first 4 locations", y=0.98)
    fig.tight_layout(rect=[0, 0, 0.96, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

    print(f"[artifact] panel saved to: {outpath}")
    assert outpath.exists()


def test_overlap_and_subpop_selection_unit():
    """
    Unit test on a tiny synthetic case to verify:
      - overlapping modifiers multiply
      - subpop selection by indices works
    """
    # tiny (P=1 named r0, T=5, L=3)
    param_names = ["r0"]
    T, L = 5, 3
    start = dt.date(2025, 1, 1)
    params = np.ones((1, T, L), dtype=np.float64) * 2.0  # baseline r0=2 everywhere

    cfg = {
        "scenarios": ["none"],
        "modifiers": {
            # two overlapping leaves on days 1..3 and 2..4; product applies on overlap (2..3)
            "m1": {
                "method": "SinglePeriodModifier",
                "parameter": "r0",
                "subpop": [0, 2],  # two locations only
                "period_start_date": "2025-01-02",
                "period_end_date": "2025-01-04",
                "value": {"value": {"distribution": "fixed", "value": 0.5}},
            },
            "m2": {
                "method": "SinglePeriodModifier",
                "parameter": "r0",
                "subpop": [2],  # only loc 2
                "period_start_date": "2025-01-03",
                "period_end_date": "2025-01-05",
                "value": {"value": {"distribution": "fixed", "value": 0.8}},
            },
            "none": {  # stack
                "method": "StackedModifier",
                "modifiers": ["m1", "m2"],
            },
        },
    }

    applier = compile_seir_modifiers(
        seir_modifiers_cfg=cfg,
        start_date=start,
        n_days=T,
        n_loc=L,
        param_names=param_names,
    )
    out = applier.apply_to_params(params, scenario="none")
    r0 = out[0]

    # expected per day:
    # day0 (2025-01-01): no modifiers => 2.0 for all
    # day1 (2025-01-02): m1 on loc 0 & 2 => 2*0.5 for 0 & 2; loc1 unchanged
    # day2 (2025-01-03): m1 on 0 & 2, m2 on 2 => 2*0.5 for loc0; 2*0.5*0.8 for loc2
    # day3 (2025-01-04): m1 on 0 & 2, m2 on 2 => same as day2
    # day4 (2025-01-05): m2 on 2 only => 2*0.8 for loc2
    exp = np.array([
        [2.0, 2.0, 2.0],
        [1.0, 2.0, 1.0],
        [1.0, 2.0, 0.8],
        [1.0, 2.0, 0.8],
        [2.0, 2.0, 1.6],
    ])
    np.testing.assert_allclose(r0, exp, rtol=0, atol=0)
