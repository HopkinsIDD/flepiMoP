import pathlib
import numpy as np
import pandas as pd
import scipy.sparse
from .utils import read_df, write_df
import logging

logger = logging.getLogger(__name__)

class SubpopulationStructure:
    """ """
    
    # TODO: allow a configuration without a geodata file.
    def __init__(self, *, setup_name, geodata_file, mobility_file):
        spatial_base_path = config["data_path"].get()
        spatial_base_path = pathlib.Path(spatial_path_prefix + spatial_base_path)
        
        setup_name=self.setup_name,
        geodata_file=spatial_base_path / spatial_config["geodata"].get(),
        mobility_file=spatial_base_path / spatial_config["mobility"].get() if spatial_config["mobility"].exists() else None,
        
        self.setup_name = setup_name
        
        self.data = pd.read_csv(
            geodata_file, converters={"subpop": lambda x: str(x).strip()}, skipinitialspace=True
        )  # subpops and populations, strip whitespaces
        self.nsubpops = len(self.data)  # K = # of locations

        # "population" is the name of the column in geodata_file with populations
        if "population" not in self.data:
            raise ValueError(
                f"There is no column 'population' in geodata, found instead {self.data.columns}"
            )
        self.subpop_pop = self.data["population"].to_numpy()  # population
        if len(np.argwhere(self.subpop_pop == 0)):
            raise ValueError(
                f"There are {len(np.argwhere(self.subpop_pop == 0))} nodes with population zero, this is not supported."
            )

        # "subpop" is the name of the column in geodata_file with the name of the subpopulations
        if "subpop" not in self.data:
            raise ValueError(  f"There is no column 'subpop' in geodata, found instead {self.data.columns}")
        self.subpop_names = self.data["subpop"].tolist()
        if len(self.subpop_names) != len(set(self.subpop_names)):
            raise ValueError(f"There are duplicate subpopulations names in geodata.")

        if mobility_file is not None:
            mobility_file = pathlib.Path(mobility_file)
            if mobility_file.suffix == ".txt":
                print("Mobility files as matrices are not recommended. Please switch soon to long form csv files.")
                self.mobility = scipy.sparse.csr_matrix(
                    np.loadtxt(mobility_file), dtype=int
                )  # K x K matrix of people moving
                # Validate mobility data
                if self.mobility.shape != (self.nsubpops, self.nsubpops):
                    raise ValueError(
                        f"mobility data must have dimensions of length of geodata ({self.nsubpops}, {self.nsubpops}). Actual: {self.mobility.shape}"
                    )

            elif mobility_file.suffix == ".csv":
                mobility_data = pd.read_csv(mobility_file, converters={"ori": str, "dest": str}, skipinitialspace=True)
                nn_dict = {v: k for k, v in enumerate(self.subpop_names)}
                mobility_data["ori_idx"] = mobility_data["ori"].apply(nn_dict.__getitem__)
                mobility_data["dest_idx"] = mobility_data["dest"].apply(nn_dict.__getitem__)
                if any(mobility_data["ori_idx"] == mobility_data["dest_idx"]):
                    raise ValueError(
                        f"Mobility fluxes with same origin and destination in long form matrix. This is not supported"
                    )

                self.mobility = scipy.sparse.coo_matrix(
                    (mobility_data.amount, (mobility_data.ori_idx, mobility_data.dest_idx)),
                    shape=(self.nsubpops, self.nsubpops),
                    dtype=int,
                ).tocsr()

            elif mobility_file.suffix == ".npz":
                self.mobility = scipy.sparse.load_npz(mobility_file).astype(int)
                # Validate mobility data
                if self.mobility.shape != (self.nsubpops, self.nsubpops):
                    raise ValueError(
                        f"mobility data must have dimensions of length of geodata ({self.nsubpops}, {self.nsubpops}). Actual: {self.mobility.shape}"
                    )
            else:
                raise ValueError(
                    f"Mobility data must either be a .csv file in longform (recommended) or a .txt matrix file. Got {mobility_file}"
                )

            # Make sure mobility values <= the population of src node
            tmp = (self.mobility.T - self.subpop_pop).T
            tmp[tmp < 0] = 0
            if tmp.any():
                rows, cols, values = scipy.sparse.find(tmp)
                errmsg = ""
                for r, c, v in zip(rows, cols, values):
                    errmsg += f"\n({r}, {c}) = {self.mobility[r, c]} > population of '{self.subpop_names[r]}' = {self.subpop_pop[r]}"
                raise ValueError(
                    f"The following entries in the mobility data exceed the source node populations in geodata:{errmsg}"
                )

            tmp = self.subpop_pop - np.squeeze(np.asarray(self.mobility.sum(axis=1)))
            tmp[tmp > 0] = 0
            if tmp.any():
                (row,) = np.where(tmp)
                errmsg = ""
                for r in row:
                    errmsg += f"\n sum accross row {r} exceed population of node '{self.subpop_names[r]}' ({self.subpop_pop[r]}), by {-tmp[r]}"
                raise ValueError(
                    f"The following rows in the mobility data exceed the source node populations in geodata:{errmsg}"
                )
        else:
            logging.critical("No mobility matrix specified -- assuming no one moves")
            self.mobility = scipy.sparse.csr_matrix(np.zeros((self.nsubpops, self.nsubpops)), dtype=int)
