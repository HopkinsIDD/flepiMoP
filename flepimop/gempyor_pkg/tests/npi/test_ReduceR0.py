import pandas as pd
import numpy as np
import os
import pathlib
import confuse

from gempyor import NPI, setup 
from gempyor.utils import config

DATA_DIR  = os.path.dirname(__file__) + "/data"
os.chdir(os.path.dirname(__file__))

class Test_ReduceR0:
    def test_ReduceR0_success(self):
       config.clear()
       config.read(user=False)
       config.set_file(f"{DATA_DIR}/config_minimal.yaml")

       ss = setup.SpatialSetup(
          setup_name="test_seir",
          geodata_file=f"{DATA_DIR}/geodata.csv",
          mobility_file=f"{DATA_DIR}/mobility.csv",
          popnodes_key="population",
          nodenames_key="geoid",
       )

       s = setup.Setup(
          setup_name="test_seir",
          spatial_setup=ss,
          nslots=1,
          npi_scenario="None",
          npi_config_seir=config["interventions"]["settings"]["None"],
          parameters_config=config["seir"]["parameters"],
          seeding_config=config["seeding"],
          ti=config["start_date"].as_date(),
          tf=config["end_date"].as_date(),
          interactive=True,
          write_csv=False,
 #        first_sim_index=first_sim_index,
 #        in_run_id=run_id,
 #        in_prefix=prefix,
 #        out_run_id=run_id,
 #        out_prefix=prefix,
          dt=0.25,
       )
	
       test = NPI.ReduceR0(npi_config=s.npi_config_seir, global_config=config,geoids=s.spatset.nodenames)
      
