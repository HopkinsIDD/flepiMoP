import multiprocessing as mp
import os
from pathlib import Path
import sys

from gempyor.model_info import ModelInfo
from gempyor.seir import run_parallel_SEIR
from gempyor.shared_cli import parse_config_files


def main(setup_sample_2pop_vaccine_scenarios, n_jobs):
    cfg = parse_config_files(
        config_filepath=setup_sample_2pop_vaccine_scenarios
        / "config_sample_2pop_vaccine_scenarios.yml",
        id_run_id=None,
        out_run_id=None,
        seir_modifiers_scenarios=[],
        outcome_modifiers_scenarios=[],
        in_prefix=None,
        nslots=None,
        jobs=n_jobs,
        write_csv=False,
        write_parquet=True,
        first_sim_index=1,
        verbose=True,
    )

    seir_modifiers_scenario, outcome_modifiers_scenario = "no_vax", None
    nchains = cfg["nslots"].as_number()
    assert nchains == 10

    modinf = ModelInfo(
        config=cfg,
        nslots=nchains,
        seir_modifiers_scenario=seir_modifiers_scenario,
        outcome_modifiers_scenario=outcome_modifiers_scenario,
        write_csv=cfg["write_csv"].get(bool),
        write_parquet=cfg["write_parquet"].get(bool),
        first_sim_index=cfg["first_sim_index"].get(int),
        in_run_id=cfg["in_run_id"].get(str) if cfg["in_run_id"].exists() else None,
        out_run_id=cfg["out_run_id"].get(str) if cfg["out_run_id"].exists() else None,
        config_filepath=cfg["config_src"].as_str_seq(),
    )

    assert run_parallel_SEIR(modinf, config=cfg, n_jobs=cfg["jobs"].get(int)) is None


if __name__ == "__main__":
    setup_sample_2pop_vaccine_scenarios = Path(sys.argv[1])
    start_method = sys.argv[2]
    n_jobs = int(sys.argv[3])
    os.chdir(setup_sample_2pop_vaccine_scenarios)
    mp.set_start_method(start_method, force=True)
    main(setup_sample_2pop_vaccine_scenarios, n_jobs)
