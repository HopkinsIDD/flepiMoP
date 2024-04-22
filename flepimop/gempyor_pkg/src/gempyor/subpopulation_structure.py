import pathlib
import numpy as np
import pandas as pd
import scipy.sparse
from .utils import read_df, write_df
import logging


logger = logging.getLogger(__name__)

subpop_pop_key="population"
subpop_names_key="subpop"


class SubpopulationStructure:
    def __init__(self, *, setup_name, subpop_config, path_prefix):
        """ Important attributes:
        - self.setup_name: Name of the setup
        - self.data: DataFrame with subpopulations and populations
        - self.nsubpops: Number of subpopulations
        - self.subpop_pop: Population of each subpopulation
        - self.subpop_names: Names of each subpopulation
        - self.mobility: Mobility matrix
        """

        geodata_file=path_prefix / subpop_config["geodata"].get()

        self.setup_name = setup_name
        self.data = pd.read_csv(
            geodata_file, converters={subpop_names_key: lambda x: str(x).strip()}, skipinitialspace=True
        )  # subpops and populations, strip whitespaces
        self.nsubpops = len(self.data)  # K = # of locations

        # subpop_pop_key is the name of the column in geodata_file with populations
        if subpop_pop_key not in self.data:
            raise ValueError(
                f"subpop_pop_key: {subpop_pop_key} does not correspond to a column in geodata: {self.data.columns}"
            )
        self.subpop_pop = self.data[subpop_pop_key].to_numpy()  # population
        if len(np.argwhere(self.subpop_pop == 0)):
            raise ValueError(
                f"There are {len(np.argwhere(self.subpop_pop == 0))} subpops with population zero, this is not supported."
            )

        # subpop_names_key is the name of the column in geodata_file with subpops
        if subpop_names_key not in self.data:
            raise ValueError(f"subpop_names_key: {subpop_names_key} does not correspond to a column in geodata.")
        self.subpop_names = self.data[subpop_names_key].tolist()
        if len(self.subpop_names) != len(set(self.subpop_names)):
            raise ValueError(f"There are duplicate subpop_names in geodata.")

        if subpop_config["mobility"].exists():
            mobility_file= path_prefix / subpop_config["mobility"].get()
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

            # Make sure mobility values <= the population of src subpop
            tmp = (self.mobility.T - self.subpop_pop).T
            tmp[tmp < 0] = 0
            if tmp.any():
                rows, cols, values = scipy.sparse.find(tmp)
                errmsg = ""
                for r, c, v in zip(rows, cols, values):
                    errmsg += f"\n({r}, {c}) = {self.mobility[r, c]} > population of '{self.subpop_names[r]}' = {self.subpop_pop[r]}"
                raise ValueError(
                    f"The following entries in the mobility data exceed the source subpop populations in geodata:{errmsg}"
                )

            tmp = self.subpop_pop - np.squeeze(np.asarray(self.mobility.sum(axis=1)))
            tmp[tmp > 0] = 0
            if tmp.any():
                (row,) = np.where(tmp)
                errmsg = ""
                for r in row:
                    errmsg += f"\n sum accross row {r} exceed population of subpop '{self.subpop_names[r]}' ({self.subpop_pop[r]}), by {-tmp[r]}"
                raise ValueError(
                    f"The following entries in the mobility data exceed the source subpop populations in geodata:{errmsg}"
                )
        else:
            logging.critical("No mobility matrix specified -- assuming no one moves")
            self.mobility = scipy.sparse.csr_matrix(np.zeros((self.nsubpops, self.nsubpops)), dtype=int)

        if subpop_config["selected"].exists():
            selected = subpop_config["selected"].get()
            if not isinstance(selected, list):
                selected = [selected]
            # find the indices of the selected subpopulations
            selected_subpop_indices = [self.subpop_names.index(s) for s in selected]
            # filter all the lists
            self.data = self.data.iloc[selected_subpop_indices]
            self.subpop_pop = self.subpop_pop[selected_subpop_indices]
            self.subpop_names = selected
            self.nsubpops = len(self.data)
            # TODO: this needs to be tested
            self.mobility = self.mobility[selected_subpop_indices][:, selected_subpop_indices]




        
