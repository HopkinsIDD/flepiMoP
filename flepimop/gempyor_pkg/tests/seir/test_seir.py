import numpy as np
import os
import pytest
import warnings
import shutil
import pandas as pd
import re

import pathlib
import pyarrow as pa
import pyarrow.parquet as pq

from gempyor import model_info, seir, NPI, file_paths, subpopulation_structure

from gempyor.utils import config

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


ignore_non_csv_mobility_warning = pytest.mark.filterwarnings(
    "ignore:Mobility files as matrices are not recommended. "
    "Please switch to long form csv files.:PendingDeprecationWarning"
)


@ignore_non_csv_mobility_warning
def test_check_values():
    config.set_file(f"{DATA_DIR}/config.yml")

    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
    )

    with warnings.catch_warnings(record=True) as w:
        seeding = np.zeros((modinf.n_days, modinf.nsubpops))

        if np.all(seeding == 0):
            warnings.warn("provided seeding has only value 0", UserWarning)

        seeding[0, 0] = 1

        # if np.all(seeding == 0):
        #    warnings.warn("provided seeding has only value 0", UserWarning)

        if np.all(modinf.mobility.data < 1):
            warnings.warn("highest mobility value is less than 1", UserWarning)

        modinf.mobility.data[0] = 0.8
        modinf.mobility.data[1] = 0.5

        if np.all(modinf.mobility.data < 1):
            warnings.warn("highest mobility value is less than 1", UserWarning)

        assert len(w) == 2
        assert issubclass(w[0].category, UserWarning)
        assert issubclass(w[1].category, UserWarning)
        assert "seeding" in str(w[0].message)
        assert "mobility" in str(w[1].message)


@ignore_non_csv_mobility_warning
def test_constant_population_legacy_integration():
    config.set_file(f"{DATA_DIR}/config.yml")

    first_sim_index = 1
    run_id = "test"
    prefix = ""
    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    assert modinf.get_engine() == "euler"

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )

    states = seir.steps_SEIR(
        modinf,
        parsed_parameters,
        transition_array,
        proportion_array,
        proportion_info,
        initial_conditions,
        seeding_data,
        seeding_amounts,
    )

    completepop = modinf.subpop_pop.sum()
    origpop = modinf.subpop_pop
    for it in range(modinf.n_days):
        totalpop = 0
        for i in range(modinf.nsubpops):
            totalpop += states["prevalence"].sum(axis=1)[it, i]
            assert (
                states["prevalence"].sum(axis=1)[it, i] - 1e-3
                < origpop[i]
                < states["prevalence"].sum(axis=1)[it, i] + 1e-3
            )
        assert completepop - 1e-3 < totalpop < completepop + 1e-3


@ignore_non_csv_mobility_warning
def test_constant_population_rk4jit_integration():
    # config.set_file(f"{DATA_DIR}/config.yml")
    config.set_file(f"{DATA_DIR}/config_seir_integration_method_rk4_2.yml")

    first_sim_index = 1
    run_id = "test"
    prefix = ""
    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )
    # s.integration_method = "rk4.jit"
    assert modinf.seir_config["integration"]["method"].get() == "rk4"

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )
    states = seir.steps_SEIR(
        modinf,
        parsed_parameters,
        transition_array,
        proportion_array,
        proportion_info,
        initial_conditions,
        seeding_data,
        seeding_amounts,
    )
    completepop = modinf.subpop_pop.sum()
    origpop = modinf.subpop_pop
    for it in range(modinf.n_days):
        totalpop = 0
        for i in range(modinf.nsubpops):
            totalpop += states["prevalence"].sum(axis=1)[it, i]
            assert (
                states["prevalence"].sum(axis=1)[it, i] - 1e-3
                < origpop[i]
                < states["prevalence"].sum(axis=1)[it, i] + 1e-3
            )
        assert completepop - 1e-3 < totalpop < completepop + 1e-3


@ignore_non_csv_mobility_warning
def test_steps_SEIR_nb_simple_spread_with_txt_matrices():
    os.chdir(os.path.dirname(__file__))
    config.clear()
    config.read(user=False)
    print("test mobility with txt matrices")
    config.set_file(f"{DATA_DIR}/config.yml")

    first_sim_index = 1
    run_id = "test_SeedOneNode"
    prefix = ""
    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )

    for i in range(5):
        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence") & (df["mc_infection_stage"] == "R")
            ].loc[str(modinf.tf), "10001"]
            > 1
        )
        assert (
            df[
                (df["mc_value_type"] == "prevalence") & (df["mc_infection_stage"] == "R")
            ].loc[str(modinf.tf), "20002"]
            > 1
        )

        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence") & (df["mc_infection_stage"] == "R")
            ].loc[str(modinf.tf), "20002"]
            > 1
        )
        assert (
            df[
                (df["mc_value_type"] == "incidence") & (df["mc_infection_stage"] == "I1")
            ].max()["20002"]
            > 0
        )
        assert (
            df[
                (df["mc_value_type"] == "incidence") & (df["mc_infection_stage"] == "I1")
            ].max()["10001"]
            > 0
        )


@ignore_non_csv_mobility_warning
def test_steps_SEIR_nb_simple_spread_with_csv_matrices():
    os.chdir(os.path.dirname(__file__))
    config.clear()
    config.read(user=False)
    config.set_file(f"{DATA_DIR}/config.yml")
    print("test mobility with csv matrices")

    first_sim_index = 1
    run_id = "test_SeedOneNode"
    prefix = ""

    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )

    for i in range(5):
        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)

        assert (
            df[
                (df["mc_value_type"] == "incidence") & (df["mc_infection_stage"] == "I1")
            ].max()["20002"]
            > 0
        )
        assert (
            df[
                (df["mc_value_type"] == "incidence") & (df["mc_infection_stage"] == "I1")
            ].max()["10001"]
            > 0
        )


@ignore_non_csv_mobility_warning
def test_steps_SEIR_no_spread():
    os.chdir(os.path.dirname(__file__))
    print("test mobility with no spread")
    config.set_file(f"{DATA_DIR}/config.yml")

    first_sim_index = 1
    run_id = "test_SeedOneNode"
    prefix = ""
    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="None",
        write_csv=False,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    modinf.mobility.data = modinf.mobility.data * 0

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )

    for i in range(10):
        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence") & (df["mc_infection_stage"] == "R")
            ].loc[str(modinf.tf), "20002"]
            == 0.0
        )

        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence") & (df["mc_infection_stage"] == "R")
            ].loc[str(modinf.tf), "20002"]
            == 0.0
        )


@ignore_non_csv_mobility_warning
def test_continuation_resume():
    os.chdir(os.path.dirname(__file__))
    config.clear()
    config.read(user=False)
    config.set_file("data/config.yml")
    seir_modifiers_scenario = "Scenario1"
    sim_id2write = 100
    nslots = 1
    write_csv = False
    write_parquet = True
    first_sim_index = 1
    run_id = "test"
    prefix = ""

    modinf = model_info.ModelInfo(
        config=config,
        nslots=nslots,
        seir_modifiers_scenario=seir_modifiers_scenario,
        write_csv=write_csv,
        write_parquet=write_parquet,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )
    seir.onerun_SEIR(sim_id2write=int(sim_id2write), modinf=modinf, config=config)

    states_old = pq.read_table(
        file_paths.create_file_name(
            modinf.in_run_id, modinf.in_prefix, 100, "seir", "parquet"
        ),
    ).to_pandas()
    states_old = states_old[states_old["date"] == "2020-03-15"].reset_index(drop=True)

    config.clear()
    config.read(user=False)
    config.set_file("data/config_continuation_resume.yml")
    seir_modifiers_scenario = "Scenario1"
    sim_id2write = 100
    nslots = 1
    write_csv = False
    write_parquet = True
    first_sim_index = 1
    run_id = "test"
    prefix = ""

    modinf = model_info.ModelInfo(
        config=config,
        nslots=nslots,
        seir_modifiers_scenario=seir_modifiers_scenario,
        write_csv=write_csv,
        write_parquet=write_parquet,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seir.onerun_SEIR(sim_id2write=sim_id2write, modinf=modinf, config=config)

    states_new = pq.read_table(
        file_paths.create_file_name(
            modinf.in_run_id, modinf.in_prefix, sim_id2write, "seir", "parquet"
        ),
    ).to_pandas()
    states_new = states_new[states_new["date"] == "2020-03-15"].reset_index(drop=True)
    assert (
        (
            states_old[states_old["mc_value_type"] == "prevalence"]
            == states_new[states_new["mc_value_type"] == "prevalence"]
        )
        .all()
        .all()
    )

    seir.onerun_SEIR(
        sim_id2write=sim_id2write + 1,
        modinf=modinf,
        sim_id2load=sim_id2write,
        load_ID=True,
        config=config,
    )
    states_new = pq.read_table(
        file_paths.create_file_name(
            modinf.in_run_id, modinf.in_prefix, sim_id2write + 1, "seir", "parquet"
        ),
    ).to_pandas()
    states_new = states_new[states_new["date"] == "2020-03-15"].reset_index(drop=True)
    for path in ["model_output/seir", "model_output/snpi", "model_output/spar"]:
        shutil.rmtree(path)


@ignore_non_csv_mobility_warning
def test_inference_resume():
    os.chdir(os.path.dirname(__file__))
    config.clear()
    config.read(user=False)
    config.set_file("data/config.yml")
    seir_modifiers_scenario = "Scenario1"
    sim_id2write = 100
    nslots = 1
    write_csv = False
    write_parquet = True
    first_sim_index = 1
    run_id = "test"
    prefix = ""

    modinf = model_info.ModelInfo(
        config=config,
        nslots=nslots,
        seir_modifiers_scenario=seir_modifiers_scenario,
        write_csv=write_csv,
        write_parquet=write_parquet,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seir.onerun_SEIR(sim_id2write=int(sim_id2write), modinf=modinf, config=config)
    npis_old = pq.read_table(
        file_paths.create_file_name(
            modinf.in_run_id, modinf.in_prefix, sim_id2write, "snpi", "parquet"
        )
    ).to_pandas()

    config.clear()
    config.read(user=False)
    config.set_file("data/config_inference_resume.yml")
    seir_modifiers_scenario = "Scenario1"
    nslots = 1
    write_csv = False
    write_parquet = True
    first_sim_index = 1
    run_id = "test"
    prefix = ""

    modinf = model_info.ModelInfo(
        config=config,
        nslots=nslots,
        seir_modifiers_scenario=seir_modifiers_scenario,
        write_csv=write_csv,
        write_parquet=write_parquet,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seir.onerun_SEIR(
        sim_id2write=sim_id2write + 1,
        modinf=modinf,
        sim_id2load=sim_id2write,
        load_ID=True,
        config=config,
    )
    npis_new = pq.read_table(
        file_paths.create_file_name(
            modinf.in_run_id, modinf.in_prefix, sim_id2write + 1, "snpi", "parquet"
        )
    ).to_pandas()

    assert npis_old["modifier_name"].isin(["None", "Wuhan", "KansasCity"]).all()
    assert npis_new["modifier_name"].isin(["None", "Wuhan", "KansasCity", "BrandNew"]).all()
    # assert((['None', 'Wuhan', 'KansasCity']).isin(npis_old["modifier_name"]).all())
    # assert((['None', 'Wuhan', 'KansasCity', 'BrandNew']).isin(npis_new["modifier_name"]).all())
    assert (npis_old["start_date"] == "2020-04-01").all()
    assert (npis_old["end_date"] == "2020-05-15").all()
    assert (npis_new["start_date"] == "2020-04-02").all()
    assert (npis_new["end_date"] == "2020-05-16").all()
    for path in ["model_output/seir", "model_output/snpi", "model_output/spar"]:
        shutil.rmtree(path)

    ## Clean up after ourselves


def test_parallel_compartments_with_vacc():
    os.chdir(os.path.dirname(__file__))
    config.clear()
    config.read(user=False)

    config.set_file(f"{DATA_DIR}/config_parallel.yml")

    first_sim_index = 1
    run_id = "test_parallel"
    prefix = ""
    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="Scenario_vacc",
        write_parquet=True,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )

    for i in range(5):
        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence")
                & (df["mc_infection_stage"] == "R")
                & (df["mc_vaccination_stage"] == "first_dose")
            ].max()["10001"]
            > 2
        )

        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence")
                & (df["mc_infection_stage"] == "R")
                & (df["mc_vaccination_stage"] == "first_dose")
            ].max()["10001"]
            > 2
        )


def test_parallel_compartments_no_vacc():
    config.clear()
    config.read(user=False)
    os.chdir(os.path.dirname(__file__))
    config.set_file(f"{DATA_DIR}/config_parallel.yml")

    first_sim_index = 1
    run_id = "test_parallel"
    prefix = ""

    modinf = model_info.ModelInfo(
        config=config,
        nslots=1,
        seir_modifiers_scenario="Scenario_novacc",
        write_parquet=True,
        first_sim_index=first_sim_index,
        in_run_id=run_id,
        in_prefix=prefix,
        out_run_id=run_id,
        out_prefix=prefix,
    )

    seeding_data, seeding_amounts = modinf.get_seeding_data(sim_id=100)
    initial_conditions = modinf.initial_conditions.get_from_config(
        sim_id=100, modinf=modinf
    )

    npi = NPI.NPIBase.execute(
        npi_config=modinf.npi_config_seir,
        modinf_ti=modinf.ti,
        modinf_tf=modinf.tf,
        modifiers_library=modinf.seir_modifiers_library,
        subpops=modinf.subpop_struct.subpop_names,
        pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
        pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
            "reduction_product"
        ],
    )

    params = modinf.parameters.parameters_quick_draw(modinf.n_days, modinf.nsubpops)
    params = modinf.parameters.parameters_reduce(params, npi)

    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()
    parsed_parameters = modinf.compartments.parse_parameters(
        params, modinf.parameters.pnames, unique_strings
    )

    for i in range(5):
        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence")
                & (df["mc_infection_stage"] == "R")
                & (df["mc_vaccination_stage"] == "first_dose")
            ].max()["10001"]
            == 0
        )

        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
        df = seir.states2Df(modinf, states)
        assert (
            df[
                (df["mc_value_type"] == "prevalence")
                & (df["mc_infection_stage"] == "R")
                & (df["mc_vaccination_stage"] == "first_dose")
            ].max()["10001"]
            == 0
        )
