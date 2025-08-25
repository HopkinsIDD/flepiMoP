# tests/outcomes/test_outcomes_integration.py
import numpy as np
import pytest
import confuse
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gempyor.model_info import ModelInfo
from gempyor.vectorization_experiments import RHSfactory
from gempyor.vectorized_outcomes import compile_outcomes, OutcomeObserver

# --- helpers: incidence/prob/node resolvers -----------------
def build_safe_param_expr_lookup(
    unique_strings: list[str],
) -> tuple[dict[int, str] | None, dict[str, int]]:
    name_to_row = {name: i for i, name in enumerate(unique_strings)}
    expr_lookup: dict[int, str] = {}
    for idx, s in enumerate(unique_strings):
        if "*" in s:
            terms = [t.strip() for t in s.split("*")]
            if all(term in name_to_row for term in terms):
                expr_lookup[idx] = s
    return (expr_lookup if expr_lookup else None, name_to_row)


def param_lookup(param, param_names):
    if param in param_names:
        return np.where(param_names == param)[0][0]
    return None


def compartment_lookup(compartment: str, compartments_df):
    mask = compartments_df["infection_stage"].astype(str).str.startswith(compartment)
    return compartments_df[mask].index


def make_incidence_resolver(compartments_df, transitions_arr):
    dst_row = transitions_arr[1, :].astype(np.int64)
    def resolve_incidence(infection_stage, vaccination_stage, variant_type, age_strata):
        mask = np.ones(len(compartments_df), dtype=bool)
        if infection_stage is not None:
            cs = compartments_df["infection_stage"].astype(str)
            mask &= (cs == infection_stage) | cs.str.startswith(str(infection_stage))
        if vaccination_stage is not None:
            mask &= compartments_df["vaccination_stage"].astype(str).eq(str(vaccination_stage))
        if variant_type is not None:
            mask &= compartments_df["variant_type"].astype(str).eq(str(variant_type))
        if age_strata is not None:
            mask &= compartments_df["age_strata"].astype(str).eq(str(age_strata))
        dst_comp_indices = compartments_df.index.values[mask]
        tidx = np.nonzero(np.isin(dst_row, dst_comp_indices))[0]
        return tidx
    return resolve_incidence

def make_node_mask_from_labels(N):
    mask = np.ones(N, dtype=np.float64)
    return lambda inc: mask

def make_prob_vector_from_cfg(N):
    return lambda name, base_prob: np.full(N, float(base_prob if base_prob is not None else 1.0))

@pytest.fixture(scope="module")
def modelinfo_from_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("model_input")

    # Resolve tutorials dir relative to this test file
    repo_root = Path(__file__).resolve().parents[4]  # flepiMoP/
    tutorial_dir = repo_root / "examples" / "tutorials"

    # --- Copy Structured_Example inputs ---
    src_structured = tutorial_dir / "model_input" / "Structured_Example"
    dst_structured = tmp_path / "model_input" / "Structured_Example"
    shutil.copytree(src_structured, dst_structured, dirs_exist_ok=True)

    # --- Copy initial_condition directory (plugin + CSVs) ---
    src_ic = tutorial_dir / "model_input" / "initial_condition"
    dst_ic = tmp_path / "model_input" / "initial_condition"
    shutil.copytree(src_ic, dst_ic, dirs_exist_ok=True)

    # --- Copy config file ---
    config_file = "Structured_Example.yml"
    config_path = tmp_path / config_file
    shutil.copyfile(tutorial_dir / config_file, config_path)

    # --- Workaround: patch YAML to absolute paths ---
    cfg_text = config_path.read_text()
    cfg_text = cfg_text.replace("model_input/", str(tmp_path / "model_input") + "/")
    config_path.write_text(cfg_text)

    # --- Load config ---
    config = confuse.Configuration("TestModel", __name__)
    config.set_file(str(config_path))

    # --- Build ModelInfo ---
    model = ModelInfo(
        config=config,
        config_filepath=str(config_path),
        path_prefix=str(tmp_path),  # unused due to bug, but keep for API consistency
        setup_name="Structured_Example",
        seir_modifiers_scenario="none",
    )

    return model, config


@pytest.fixture
def model_and_inputs(modelinfo_from_config):
    model, config = modelinfo_from_config

    initial_array = model.initial_conditions.get_from_config(sim_id=0, modinf=model)
    unique_strings, transitions, transition_sum_compartments, proportion_info = (
        model.compartments.get_transition_array()
    )

    param_defs = config["seir"]["parameters"].get()
    base_params = model.parameters.parameters_quick_draw(model.n_days, model.nsubpops)
    parsed_params = model.compartments.parse_parameters(
        base_params, param_defs, unique_strings
    )

    param_expr_lookup, param_name_to_row = build_safe_param_expr_lookup(unique_strings)

    mobility_csr: csr_matrix = model.mobility
    population = model.subpop_pop

    mobility_data = mobility_csr.data
    mobility_data_indices = mobility_csr.indptr
    mobility_row_indices = mobility_csr.indices

    # Safe fraction who move (avoid divide-by-zero)
    proportion_who_move = np.zeros(model.nsubpops, dtype=np.float64)
    for i in range(model.nsubpops):
        pop_i = float(population[i])
        total_flux = float(
            mobility_data[mobility_data_indices[i] : mobility_data_indices[i + 1]].sum()
        )
        if pop_i > 0.0:
            proportion_who_move[i] = min(total_flux / pop_i, 1.0)
        else:
            proportion_who_move[i] = 0.0

    offset = model.ti.toordinal()
    t0 = 0.0
    t1 = float(model.tf.toordinal() - offset)
    dt = 0.1
    time_grid = np.linspace(t0, t1, int((t1 - t0) / dt) + 1).astype(np.float64)

    # Build the precomputed dict for RHSfactory (seeding intentionally omitted: OFF)
    precomputed = {
        "ncompartments": initial_array.shape[0],
        "nspatial_nodes": initial_array.shape[1],
        "transitions": transitions,
        "proportion_info": proportion_info,
        "transition_sum_compartments": transition_sum_compartments,
        "percent_day_away": 0.5,  # same as below
        "proportion_who_move": proportion_who_move,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_data_indices,
        "mobility_row_indices": mobility_row_indices,
        "population": population,
        # NOTE: no 'seeding_data' / 'seeding_amounts' keys => seeding OFF
    }

    return {
        "model": model,
        "config": config,
        "initial_array": initial_array,
        "params": parsed_params,
        "transitions": transitions,
        "proportion_info": proportion_info,
        "transition_sum_compartments": transition_sum_compartments,
        "mobility_data": mobility_data,
        "mobility_data_indices": mobility_data_indices,
        "mobility_row_indices": mobility_row_indices,
        "proportion_who_move": proportion_who_move,
        "population": population,
        "time_grid": time_grid,
        "dt": dt,
        "percent_day_away": 0.5,
        "param_expr_lookup": param_expr_lookup,
        "param_name_to_row": param_name_to_row,
        "precomputed": precomputed,
        "param_names": np.array(list(param_defs.keys())),
        "compartments": model.compartments.compartments,
    }

# ------------------------------------------------------------
@pytest.mark.slow
def test_outcomes_with_vectorized_solver(model_and_inputs, tmp_path):
    out = model_and_inputs
    ncomp, nloc = out["initial_array"].shape

    # RHSfactory (vectorized, no seeding for this test)
    factory = RHSfactory(
        precomputed=out["precomputed"],
        param_expr_lookup=out["param_expr_lookup"],
        param_name_to_row=out["param_name_to_row"],
        param_time_mode="step",
    )

    ndays = out["model"].n_days
    t_eval = np.arange(0.0, float(ndays), 1.0, dtype=np.float64)

    # --- Outcomes setup -------------------------------------
    outcomes_cfg = out["config"]["outcomes"]["outcomes"].get()
    compiled = compile_outcomes(
        outcomes_cfg,
        resolve_incidence_to_transition_rows=make_incidence_resolver(out["compartments"], out["transitions"]),
        node_mask_from_labels=make_node_mask_from_labels(nloc),
        prob_vector_from_cfg=make_prob_vector_from_cfg(nloc),
        N_nodes=nloc,
        bin_width_days=1.0,
    )
    obs = OutcomeObserver(compiled, n_nodes=nloc, t0=t_eval[0], tf=t_eval[-1], bin_width_days=1.0)

    # --- Run integration (RK45 via solve_ivp) ----------------
    res = factory.solve(
        y0=out["initial_array"].ravel(),
        parameters=out["params"],
        t_span=(t_eval[0], t_eval[-1]),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-3,
        atol=1e-6,
    )
    assert res.success, f"Integration failed: {res.message}"

    # Postprocess: approximate step counts by diff of states
    states = res.y.T.reshape(len(t_eval), ncomp, nloc)
    diffs = np.diff(states, axis=0)  # (T-1,C,N)

    # crude: treat negative of delta susceptible -> new infections, etc.
    # here we just feed dummy totals to check observer wiring
    dummy_counts = np.abs(diffs).sum(axis=1)  # (T-1,N) aggregate
    for i in range(len(t_eval)-1):
        obs.on_step(t_eval[i], t_eval[i+1], np.tile(dummy_counts[i], (out["transitions"].shape[1],1)))

    times, series = obs.finalize()

    # --- Basic checks ---------------------------------------
    assert "incidI" in series
    assert series["incidI"].shape[0] == len(times)
    # nontrivial signal
    assert series["incidI"].sum() > 0.0

    # --- Quick plot -----------------------------------------
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(times, series["incidI"].sum(axis=1), label="incidI")
    if "incidH" in series:
        ax.plot(times, series["incidH"].sum(axis=1), label="incidH")
    ax.set_xlabel("Day")
    ax.set_ylabel("Incidence (total)")
    ax.legend()
    fig.savefig(tmp_path / "outcomes_overlay.png", dpi=120)
    plt.close(fig)
