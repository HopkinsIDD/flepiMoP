import datetime
import filecmp
import glob
import os
import pathlib
import shutil
import sys
import warnings

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import matplotlib.pyplot as plt

from . import file_paths, outcomes
from .utils import config

config.clear()
config.read(user=False)
config.set_file("config.yml")

run_id = 333
index = 1
outcome_modifiers_scenario = "high_death_rate"
prefix = ""
stoch_traj_flag = True

outcomes.run_delayframe_outcomes(
    config,
    int(index),
    run_id,
    prefix,  # input
    int(index),
    run_id,
    prefix,  # output
    outcome_modifiers_scenario,
    nslots=1,
    n_jobs=1,
    stoch_traj_flag=stoch_traj_flag,
)
