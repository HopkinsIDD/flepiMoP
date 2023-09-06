import datetime
import numpy as np
import os
import pandas as pd
import pytest
import confuse

from gempyor import setup, parameters

from gempyor.utils import config

TEST_SETUP_NAME = "minimal_test"

DATA_DIR = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))


class TestSetup:
    def test_Setup_success(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.csv",
            popnodes_key="population",
            nodenames_key="geoid",
        )
        s = setup.Setup(
            setup_name = TEST_SETUP_NAME,
            spatial_setup =ss,
            nslots = 1,
            ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
            tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
            npi_scenario=None,
            config_version=None,
            npi_config_seir={},
            seeding_config={},
            initial_conditions_config={},
            parameters_config={},
            seir_config=None,
            outcomes_config={},
            outcome_scenario=None,
            interactive=True,
            write_csv=False,
            write_parquet=False,
            dt=None,  # step size, in days
            first_sim_index=1,
            in_run_id=None,
            in_prefix=None,
            out_run_id=None,
            out_prefix=None,
            stoch_traj_flag=False,	
        )

    def test_tf_is_ahead_of_ti_fail(self):
        # time to finish (tf) is ahead of time to start(ti) error
        with pytest.raises(ValueError, match=r".*tf.*less.*"):
           ss = setup.SpatialSetup(
              setup_name=TEST_SETUP_NAME,
              geodata_file=f"{DATA_DIR}/geodata.csv",
              mobility_file=f"{DATA_DIR}/mobility.csv",
              popnodes_key="population",
              nodenames_key="geoid",
           )
           s = setup.Setup(
              setup_name = TEST_SETUP_NAME,
              spatial_setup =ss,
              nslots = 1,
              ti = datetime.datetime.strptime("2020-03-31","%Y-%m-%d"),
              tf = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
              npi_scenario=None,
              config_version=None,
              npi_config_seir={},
              seeding_config={},
              initial_conditions_config={},
              parameters_config={},
              seir_config=None,
              outcomes_config={},
              outcome_scenario=None,
              interactive=True,
              write_csv=False,
              write_parquet=False,
              dt=None,  # step size, in days
              first_sim_index=1,
              in_run_id=None,
              in_prefix=None,
              out_run_id=None,
              out_prefix=None,
              stoch_traj_flag=False,	
          )

    def test_w_config_seir_exists_success(self):
        # if seir_config is None and config["seir"].exists() then update
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_seir.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
          # parameters_config={"alpha":{"value":{"distribution":"fixed","value":.9}}},
           parameters_config={},
           seir_config=None,
           outcomes_config={},
           outcome_scenario=None,
           interactive=True,
           write_csv=False,
           write_parquet=False,
           dt=None,  # step size, in days
           first_sim_index=1,
           in_run_id=None,
           in_prefix=None,
           out_run_id=None,
           out_prefix=None,
           stoch_traj_flag=False,	
        )

        assert s.seir_config != None
        #print(s.seir_config["parameters"])
        assert s.parameters_config != None
        #print(s.integration_method) 
        assert s.integration_method == 'legacy'

    def test_w_config_seir_integration_method_rk4_1_success(self):
        # if seir_config["integration"]["method"] is best.current
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_seir_integration_method_rk4_1.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           outcomes_config={},
           outcome_scenario=None,
           interactive=True,
           write_csv=False,
           write_parquet=False,
           dt=None,  # step size, in days
           first_sim_index=1,
           in_run_id=None,
           in_prefix=None,
           out_run_id=None,
           out_prefix=None,
           stoch_traj_flag=False,	
        )
        assert s.integration_method  == "rk4.jit"

        assert s.dt == float(1/6)

    def test_w_config_seir_integration_method_rk4_2_success(self):
        # if seir_config["integration"]["method"] is rk4
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_seir_integration_method_rk4_2.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           outcomes_config={},
           outcome_scenario=None,
           interactive=True,
           write_csv=False,
           write_parquet=False,
           dt=None,  # step size, in days
           first_sim_index=1,
           in_run_id=None,
           in_prefix=None,
           out_run_id=None,
           out_prefix=None,
           stoch_traj_flag=False,
        )
        assert s.integration_method  == "rk4.jit"

    def test_w_config_seir_no_integration_success(self):
        # if not seir_config["integration"]
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_seir_no_integration.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           outcomes_config={},   
           outcome_scenario=None,
           interactive=True,
           write_csv=False,
           write_parquet=False,
           dt=None,  # step size, in days
           first_sim_index=1,
           in_run_id=None,
           in_prefix=None,
           out_run_id=None,
           out_prefix=None,
           stoch_traj_flag=False,
        )
        assert s.integration_method  == "rk4.jit"

        assert s.dt == 2.0

    def test_w_config_seir_unknown_integration_method_fail(self):
        with pytest.raises(ValueError, match=r".*Unknown.*integration.*"):
        # if in seir unknown integration method
           config.clear()
           config.read(user=False)
           config.set_file(f"{DATA_DIR}/config_seir_unknown_integration.yml")
           ss = setup.SpatialSetup(
              setup_name=TEST_SETUP_NAME,
              geodata_file=f"{DATA_DIR}/geodata.csv",
              mobility_file=f"{DATA_DIR}/mobility.csv",
              popnodes_key="population",
              nodenames_key="geoid",
           )
           s = setup.Setup(
              setup_name = TEST_SETUP_NAME,
              spatial_setup =ss,
              nslots = 1,
              ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
              tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
         #     first_sim_index=1,
           )
         #  print(s.integration_method)

    def test_w_config_seir_integration_but_no_dt_success(self):
        # if not seir_config["integration"]["dt"]
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_seir_no_dt.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           dt=None,  # step size, in days
        )

        assert s.dt == 2.0

    ''' not needed any longer
    def test_w_config_seir_old_integration_method_fail(self):
        with pytest.raises(ValueError, match=r".*Configuration.*no.*longer.*"):
        # if old method in seir
           #config.clear()
           #config.read(user=False)
           #config.set_file(f"{DATA_DIR}/config_seir_unknown_integration.yml")
           ss = setup.SpatialSetup(
              setup_name=TEST_SETUP_NAME,
              geodata_file=f"{DATA_DIR}/geodata.csv",
              mobility_file=f"{DATA_DIR}/mobility.csv",
              popnodes_key="population",
              nodenames_key="geoid",
           )
           s = setup.Setup(
              setup_name = TEST_SETUP_NAME,
              spatial_setup =ss,
              nslots = 1,
              config_version="v2",
              ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
              tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           )
    def test_w_config_seir_config_version_not_provided_fail(self):
        with pytest.raises(ValueError, match=r".*Should.*non-specified.*"):
        # if not seir_config["integration"]["dt"]
       # config.clear()
       # config.read(user=False)
       # config.set_file(f"{DATA_DIR}/config_seir_no_dt.yml")
           ss = setup.SpatialSetup(
              setup_name=TEST_SETUP_NAME,
              geodata_file=f"{DATA_DIR}/geodata.csv",
              mobility_file=f"{DATA_DIR}/mobility.csv",
              popnodes_key="population",
              nodenames_key="geoid",
           )
           s = setup.Setup(
              setup_name = TEST_SETUP_NAME,
              spatial_setup =ss,
              nslots = 1,
              ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
              tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
              npi_scenario=None,
              config_version="v1",
              npi_config_seir={},
              seeding_config={},
              initial_conditions_config={},
              parameters_config={},
              seir_config=None,
              dt=None,  # step size, in days
           )
    '''

    def test_w_config_compartments_and_seir_config_not_None_success(self):
        # if config["compartments"] and iself.seir_config was set
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_compartment.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           dt=None,  # step size, in days
        )

    def test_config_outcome_config_and_scenario_success(self):
        # if outcome_config and outcome_scenario were set
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           dt=None,  # step size, in days
           outcomes_config={"interventions":{"settings":{"None":
             {"template":"Reduce",
              "parameter":"r0",
              "value":
                 {
                   "distribution":"fixed",
                   "value":0
                 }
             }
           }}},
           outcome_scenario="None", # caution! selected the defined "None"
           write_csv=True,
        )
        assert s.npi_config_outcomes ==  s.outcomes_config["interventions"]["settings"]["None"] 
        assert s.extension == "csv"

    def test_config_write_csv_and_write_parquet_success(self):
        # if both write_csv and write_parquet are True
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           dt=None,  # step size, in days
           outcomes_config={"interventions":{"settings":{"None":
             {"template":"Reduce",
              "parameter":"r0",
              "value":
                 {
                   "distribution":"fixed",
                   "value":0
                 }
             }
           }}},
           outcome_scenario="None", # caution! selected the defined "None"
           write_csv=True,
           write_parquet=True,
        )
        assert s.write_parquet

    def test_w_config_seir_exists_and_outcomes_config(self):
        # if seir_config is None and config["seir"].exists() then update
        config.clear()
        config.read(user=False)
        config.set_file(f"{DATA_DIR}/config_seir.yml")
        ss = setup.SpatialSetup(
           setup_name=TEST_SETUP_NAME,
           geodata_file=f"{DATA_DIR}/geodata.csv",
           mobility_file=f"{DATA_DIR}/mobility.csv",
           popnodes_key="population",
           nodenames_key="geoid",
        )
        s = setup.Setup(
           setup_name = TEST_SETUP_NAME,
           spatial_setup =ss,
           nslots = 1,
           ti = datetime.datetime.strptime("2020-01-31","%Y-%m-%d"),
           tf = datetime.datetime.strptime("2020-05-31","%Y-%m-%d"),
           npi_scenario=None,
           config_version=None,
           npi_config_seir={},
           seeding_config={},
           initial_conditions_config={},
           parameters_config={},
           seir_config=None,
           outcomes_config={"interventions":{"settings":{"None":
             {"template":"Reduce",
              "parameter":"r0",
              "value":
                 {
                   "distribution":"fixed",
                   "value":0
                 }
             }
           }}},
           outcome_scenario="None",
           interactive=True,
           write_csv=False,
           write_parquet=False,
           dt=None,  # step size, in days
           first_sim_index=1,
           in_run_id="in_run_id_0",
           in_prefix=None,
           out_run_id="out_run_id_0",
           out_prefix=None,
           stoch_traj_flag=False,	
        )
        #s.get_input_filename(ftype="spar", sim_id=0, extension_override="")
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="seir", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="spar", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="snpi", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="hosp", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="hpar", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="hnpi", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="seir", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="spar", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="snpi", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="hosp", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="hpar", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="hnpi", sim_id=0))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="seir", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="spar", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="snpi", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="hosp", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="hpar", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_input_filename(ftype="hnpi", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="seir", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="spar", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="snpi", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="hosp", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="hpar", sim_id=1, extension_override="csv"))
        os.path.isfile(DATA_DIR+s.get_output_filename(ftype="hnpi", sim_id=1, extension_override="csv"))


    '''
    def test_SpatialSetup_npz_success3(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility.npz",
            popnodes_key="population",
            nodenames_key="geoid",
        )
    def test_SpatialSetup_wihout_mobility_success3(self):
        ss = setup.SpatialSetup(
            setup_name=TEST_SETUP_NAME,
            geodata_file=f"{DATA_DIR}/geodata.csv",
            mobility_file=f"{DATA_DIR}/mobility0.csv",
            popnodes_key="population",
            nodenames_key="geoid",
        )

    def test_bad_popnodes_key_fail(self):
        # Bad popnodes_key error
        with pytest.raises(ValueError, match=r".*popnodes_key.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_small.txt",
                popnodes_key="wrong",
                nodenames_key="geoid",
            )

    def test_population_0_nodes_fail(self):
        with pytest.raises(ValueError, match=r".*population.*zero.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata0.csv",
                mobility_file=f"{DATA_DIR}/mobility.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_fileformat_fail(self):
        with pytest.raises(ValueError, match=r".*Mobility.*longform.*matrix.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_bad_nodenames_key_fail(self):
        with pytest.raises(ValueError, match=r".*nodenames_key.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility.txt",
                popnodes_key="population",
                nodenames_key="wrong",
            )

    def test_duplicate_nodenames_key_fail(self):
        with pytest.raises(ValueError, match=r".*duplicate.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata_dup.csv",
                mobility_file=f"{DATA_DIR}/mobility.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )
    '''
    def test_mobility_shape_in_npz_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*Actual.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_2x3.npz",
                popnodes_key="population",
                nodenames_key="geoid",
            )
    '''
    def test_mobility_dimensions_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*dimensions.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_small.txt",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_same_ori_dest_fail(self):
        with pytest.raises(ValueError, match=r".*Mobility.*same.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_same_ori_dest.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )

    def test_mobility_too_big_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*population.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility_big.txt",
                popnodes_key="population",
                nodenames_key="geoid",
            )
    def test_mobility_data_exceeded_fail(self):
        with pytest.raises(ValueError, match=r".*mobility.*exceed.*"):
            setup.SpatialSetup(
                setup_name=TEST_SETUP_NAME,
                geodata_file=f"{DATA_DIR}/geodata.csv",
                mobility_file=f"{DATA_DIR}/mobility1001.csv",
                popnodes_key="population",
                nodenames_key="geoid",
            )
    '''
