import gempyor
import numpy as np
import pandas as pd
import datetime
import pytest

from gempyor.utils import config

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import glob, os, sys, shutil
from pathlib import Path

# import seaborn as sns
import pyarrow.parquet as pq
import pyarrow as pa
from gempyor import file_paths, outcomes, seir

config_path_prefix = ""

os.chdir(os.path.dirname(__file__))


def test_full_npis_read_write():
    os.chdir(os.path.dirname(__file__))

    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config_npi.yml",
        run_id=105,
        prefix="",
        first_sim_index=1,
        seir_modifiers_scenario="inference",
        outcome_modifiers_scenario="outcome_interventions",
        stoch_traj_flag=False,
        out_run_id=105,
    )
    # inference_simulator.one_simulation(sim_id2write=1,load_ID=False)

    # outcomes.onerun_delayframe_outcomes(
    #    sim_id2write=1, modinf=inference_simulator.s, load_ID=False, sim_id2load=1
    # )

    npi_outcomes = outcomes.build_outcome_modifiers(
        inference_simulator.modinf, load_ID=False, sim_id2load=None, config=config
    )
    # npi_seir = seir.build_npi_SEIR(
    #    inference_simulator.s, load_ID=False, sim_id2load=None, config=config
    # )

    inference_simulator.modinf.write_simID(ftype="hnpi", sim_id=1, df=npi_outcomes.getReductionDF())

    hnpi_read = pq.read_table(f"{config_path_prefix}model_output/hnpi/000000001.105.hnpi.parquet").to_pandas()
    hnpi_read["value"] = np.random.random(len(hnpi_read)) * 2 - 1
    out_hnpi = pa.Table.from_pandas(hnpi_read, preserve_index=False)
    pa.parquet.write_table(out_hnpi, file_paths.create_file_name(105, "", 1, "hnpi", "parquet"))
    import random

    random.seed(10)

    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config_npi.yml",
        run_id=105,
        prefix="",
        first_sim_index=1,
        seir_modifiers_scenario="inference",
        outcome_modifiers_scenario="outcome_interventions",
        stoch_traj_flag=False,
        out_run_id=106,
    )
    # shutil.move('model_output/seir/000000001.105.seir.parquet', 'model_output/seir/000000001.106.seir.parquet')

    # outcomes.onerun_delayframe_outcomes(
    #    sim_id2write=1, modinf=inference_simulator.s, load_ID=True, sim_id2load=1
    # )

    npi_outcomes = outcomes.build_outcome_modifiers(
        inference_simulator.modinf, load_ID=True, sim_id2load=1, config=config
    )
    inference_simulator.modinf.write_simID(ftype="hnpi", sim_id=1, df=npi_outcomes.getReductionDF())

    hnpi_read = pq.read_table(f"{config_path_prefix}model_output/hnpi/000000001.105.hnpi.parquet").to_pandas()
    hnpi_wrote = pq.read_table(f"{config_path_prefix}model_output/hnpi/000000001.106.hnpi.parquet").to_pandas()
    assert (hnpi_read == hnpi_wrote).all().all()

    # runs with the new, random NPI
    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config_npi.yml",
        run_id=106,
        prefix="",
        first_sim_index=1,
        seir_modifiers_scenario="inference",
        outcome_modifiers_scenario="outcome_interventions",
        stoch_traj_flag=False,
        out_run_id=107,
    )
    # shutil.move('model_output/seir/000000001.106.seir.parquet', 'model_output/seir/000000001.107.seir.parquet')

    # outcomes.onerun_delayframe_outcomes(
    #    sim_id2write=1, modinf=inference_simulator.s, load_ID=True, sim_id2load=1
    # )

    npi_outcomes = outcomes.build_outcome_modifiers(
        inference_simulator.modinf, load_ID=True, sim_id2load=1, config=config
    )
    inference_simulator.modinf.write_simID(ftype="hnpi", sim_id=1, df=npi_outcomes.getReductionDF())

    hnpi_read = pq.read_table(f"{config_path_prefix}model_output/hnpi/000000001.106.hnpi.parquet").to_pandas()
    hnpi_wrote = pq.read_table(f"{config_path_prefix}model_output/hnpi/000000001.107.hnpi.parquet").to_pandas()
    assert (hnpi_read == hnpi_wrote).all().all()


def test_spatial_groups():
    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config_test_spatial_group_npi.yml",
        run_id=105,
        prefix="",
        first_sim_index=1,
        seir_modifiers_scenario="inference",
        stoch_traj_flag=False,
        out_run_id=105,
    )

    # Test build from config, value of the reduction array
    npi = seir.build_npi_SEIR(inference_simulator.modinf, load_ID=False, sim_id2load=None, config=config)

    # all independent: r1
    assert len(npi.getReduction("r1")["2021-01-01"].unique()) == inference_simulator.modinf.nsubpops
    assert npi.getReduction("r1").isna().sum().sum() == 0

    # all the same: r2
    assert len(npi.getReduction("r2")["2021-01-01"].unique()) == 1
    assert npi.getReduction("r2").isna().sum().sum() == 0

    # two groups: r3
    assert len(npi.getReduction("r3")["2020-04-15"].unique()) == inference_simulator.modinf.nsubpops - 2
    assert npi.getReduction("r3").isna().sum().sum() == 0
    assert len(npi.getReduction("r3").loc[["01000", "02000"], "2020-04-15"].unique()) == 1
    assert len(npi.getReduction("r3").loc[["04000", "06000"], "2020-04-15"].unique()) == 1

    # one group: r4
    assert (
        len(npi.getReduction("r4")["2020-04-15"].unique()) == 4
    )  # 0 for these not included, 1 unique for the group, and two for the rest
    assert npi.getReduction("r4").isna().sum().sum() == 0
    assert len(npi.getReduction("r4").loc[["01000", "02000"], "2020-04-15"].unique()) == 1
    assert len(npi.getReduction("r4").loc[["04000", "06000"], "2020-04-15"].unique()) == 2
    assert (npi.getReduction("r4").loc[["05000", "08000"], "2020-04-15"] == 0).all()

    # mtr group: r5
    assert npi.getReduction("r5").isna().sum().sum() == 0
    assert len(npi.getReduction("r5")["2020-12-15"].unique()) == 2
    assert len(npi.getReduction("r5")["2020-10-15"].unique()) == 4
    assert len(npi.getReduction("r5").loc[["01000", "04000"], "2020-10-15"].unique()) == 1
    assert len(npi.getReduction("r5").loc[["02000", "06000"], "2020-10-15"].unique()) == 2

    # test the dataframes that are wrote.
    npi_df = npi.getReductionDF()

    # all independent: r1
    df = npi_df[npi_df["modifier_name"] == "all_independent"]
    assert len(df) == inference_simulator.modinf.nsubpops
    for g in df["subpop"]:
        assert "," not in g

    # all the same: r2
    df = npi_df[npi_df["modifier_name"] == "all_together"]
    assert len(df) == 1
    assert set(df["subpop"].iloc[0].split(",")) == set(inference_simulator.modinf.subpop_struct.subpop_names)
    assert len(df["subpop"].iloc[0].split(",")) == inference_simulator.modinf.nsubpops

    # two groups: r3
    df = npi_df[npi_df["modifier_name"] == "two_groups"]
    assert len(df) == inference_simulator.modinf.nsubpops - 2
    for g in ["01000", "02000", "04000", "06000"]:
        assert g not in df["subpop"]
    assert len(df[df["subpop"] == "01000,02000"]) == 1
    assert len(df[df["subpop"] == "04000,06000"]) == 1

    # mtr group: r5
    df = npi_df[npi_df["modifier_name"] == "mt_reduce"]
    assert len(df) == 4
    assert df.subpop.to_list() == ["09000,10000", "02000", "06000", "01000,04000"]
    assert df[df["subpop"] == "09000,10000"]["start_date"].iloc[0] == "2020-12-01,2021-12-01"
    assert (
        df[df["subpop"] == "01000,04000"]["start_date"].iloc[0]
        == df[df["subpop"] == "06000"]["start_date"].iloc[0]
        == "2020-10-01,2021-10-01"
    )


def test_spatial_groups():
    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config_test_spatial_group_npi.yml",
        run_id=105,
        prefix="",
        first_sim_index=1,
        seir_modifiers_scenario="inference",
        stoch_traj_flag=False,
        out_run_id=105,
    )

    # Test build from config, value of the reduction array
    npi = seir.build_npi_SEIR(inference_simulator.modinf, load_ID=False, sim_id2load=None, config=config)
    npi_df = npi.getReductionDF()

    inference_simulator.modinf.write_simID(ftype="snpi", sim_id=1, df=npi_df)

    snpi_read = pq.read_table(f"{config_path_prefix}model_output/snpi/000000001.105.snpi.parquet").to_pandas()
    snpi_read["value"] = np.random.random(len(snpi_read)) * 2 - 1
    out_snpi = pa.Table.from_pandas(snpi_read, preserve_index=False)
    pa.parquet.write_table(out_snpi, file_paths.create_file_name(106, "", 1, "snpi", "parquet"))

    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config_test_spatial_group_npi.yml",
        run_id=106,
        prefix="",
        first_sim_index=1,
        seir_modifiers_scenario="inference",
        stoch_traj_flag=False,
        out_run_id=107,
    )

    npi_seir = seir.build_npi_SEIR(inference_simulator.modinf, load_ID=True, sim_id2load=1, config=config)
    inference_simulator.modinf.write_simID(ftype="snpi", sim_id=1, df=npi_seir.getReductionDF())

    snpi_read = pq.read_table(f"{config_path_prefix}model_output/snpi/000000001.106.snpi.parquet").to_pandas()
    snpi_wrote = pq.read_table(f"{config_path_prefix}model_output/snpi/000000001.107.snpi.parquet").to_pandas()

    # now the order can change, so we need to sort by subpop and start_date
    snpi_wrote = snpi_wrote.sort_values(by=["subpop", "start_date"]).reset_index(drop=True)
    snpi_read = snpi_read.sort_values(by=["subpop", "start_date"]).reset_index(drop=True)
    assert (snpi_read == snpi_wrote).all().all()

    npi_read = seir.build_npi_SEIR(
        inference_simulator.modinf, load_ID=False, sim_id2load=1, config=config, bypass_DF=snpi_read
    )
    npi_wrote = seir.build_npi_SEIR(
        inference_simulator.modinf, load_ID=False, sim_id2load=1, config=config, bypass_DF=snpi_wrote
    )

    assert (npi_read.getReductionDF() == npi_wrote.getReductionDF()).all().all()

    assert (npi_wrote.getReduction("r1") == npi_read.getReduction("r1")).all().all()
    assert (npi_wrote.getReduction("r2") == npi_read.getReduction("r2")).all().all()
    assert (npi_wrote.getReduction("r3") == npi_read.getReduction("r3")).all().all()
    assert (npi_wrote.getReduction("r4") == npi_read.getReduction("r4")).all().all()
    assert (npi_wrote.getReduction("r5") == npi_read.getReduction("r5")).all().all()
