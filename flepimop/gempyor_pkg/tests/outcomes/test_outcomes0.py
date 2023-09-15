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
import glob, os, sys
from pathlib import Path

# import seaborn as sns
import pyarrow.parquet as pq
import pyarrow as pa
from gempyor import file_paths, setup, outcomes

config_path_prefix = ""  #'tests/outcomes/'

### To generate files for this test, see notebook Test Outcomes  playbook.ipynb in COVID19_Maryland

geoid = ["15005", "15007", "15009", "15001", "15003"]
diffI = np.arange(5) * 2
date_data = datetime.date(2020, 4, 15)
subclasses = ["_A", "_B"]

os.chdir(os.path.dirname(__file__))


def test_outcome_scenario():
    os.chdir(os.path.dirname(__file__))  ## this is redundant but necessary. Why ?
    inference_simulator = gempyor.GempyorSimulator(
        config_path=f"{config_path_prefix}config.yml",
        run_id=1,
        prefix="",
        first_sim_index=1,
        outcome_scenario="high_death_rate",
        stoch_traj_flag=False,
    )

