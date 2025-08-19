import os
import numpy as np
import pandas as pd
import gempyor.initial_conditions

# path to IC folder (local to this plugin file)
IC_dir = os.path.dirname(__file__)

class InitialConditions(gempyor.initial_conditions.InitialConditions):
    def get_from_config(self, sim_id: int, modinf) -> np.ndarray:
        """
        Assigns initial conditions for the FlepiMoP model.
        Uses inferred initial conditions as estimated by Josh on previous Flu seasons.

        Args
        ----
        sim_id : int
            Simulation ID.
        modinf : ModelInfo
            ModelInfo object containing config and path_prefix.

        Returns
        -------
        y0 : np.ndarray
            Initial condition array, shape = (compartments, subpops).
        """

        # Use path_prefix from ModelInfo instead of CWD
        config_dir = os.path.join(
            modinf.path_prefix,
            f'model_input/{self.initial_conditions_config["config_name"].get()}/'
        )

        # ---- input checks ----
        try:
            seasons = self.initial_conditions_config["seasons"].get()
        except Exception:
            raise ValueError(
                "the plugin initial condition for the influenza model "
                "requires the definition of 'seasons' under 'initial_conditions' in the config."
            )

        if not isinstance(seasons, (list, str)):
            raise TypeError(f"'seasons' must be of type str or list. found: '{type(seasons)}'")
        if isinstance(seasons, str):
            seasons = [seasons]
        assert all(isinstance(item, str) for item in seasons), "'seasons' must be a list of str."

        for s in seasons:
            if ((len(s) != 6) or (not s[:2].isdigit()) or (s[2:4] != 'to') or (not s[4:].isdigit())):
                raise ValueError(f"season '{s}' does not match the required format 'xxtoxx'.")
        seasons.sort()

        # ---- load data ----
        ic_df = pd.read_csv(os.path.join(IC_dir, 'initial_conditions.csv'), header=0)
        ic_df['disease_state'] = ic_df['disease_state'].replace({
            "S": "S_unvaccinated",
            "I": "I_unvaccinated",
            "Iv": "I_vaccinated",
            "V": "S_vaccinated",
            "R": "R_unvaccinated",
            "H": "H_unvaccinated",
            "D": "D_unvaccinated",
        })

        demo_flepi = pd.read_csv(
            os.path.join(config_dir, 'geodata_2019_statelevel.csv'),
            converters={"subpop": lambda x: str(x).strip()}
        )[["subpop", "population"]]

        if len(seasons) == 1:
            if not demo_flepi['subpop'].str.match(r'^\d{2}000').all():
                raise ValueError("demography file entries must match 'xx000' format for single season.")
        else:
            if not demo_flepi['subpop'].str.match(r'^\d{2}000_\d{2}to\d{2}$').all():
                raise ValueError("demography file entries must match 'xx000_xxtoxx' format for multi-season.")

        demo = pd.read_csv(
            os.path.join(IC_dir, 'geodata_2019_agestrat.csv'),
            dtype={'subpop': str}
        )[["subpop", "age_strata", "prop"]]
        demo = demo.rename(columns={'subpop': 'fips'}).set_index('fips').sort_index().reset_index()

        names_df = modinf.compartments.compartments.copy()
        y0 = np.zeros((modinf.compartments.compartments.shape[0], modinf.nsubpops))

        if len(seasons) > 1:
            demo_flepi[['fips', 'season']] = demo_flepi['subpop'].str.split('_', expand=True)
            demo_flepi = demo_flepi.sort_values('season')
        else:
            demo_flepi['fips'] = demo_flepi['subpop']

        demo = demo[demo['fips'].isin(demo_flepi['fips'].unique())]

        for i, subpop in enumerate(modinf.subpop_struct.subpop_names):
            fips = subpop.split('_')[0]
            season = subpop.split('_')[1] if len(seasons) > 1 else seasons[0]
            ic_curr = ic_df[ic_df['season'] == season][['disease_state','mean']].set_index('disease_state')

            names_df['proportion_ic'] = 0.0
            names_df.loc[(names_df['infection_stage'] == 'S') & (names_df['vaccination_stage'] == 'unvaccinated'), 'proportion_ic'] = float(ic_curr.loc['S_unvaccinated'].values)
            names_df.loc[(names_df['infection_stage'] == 'S') & (names_df['vaccination_stage'] == '1dose'), 'proportion_ic'] = float(ic_curr.loc['S_vaccinated'].values)
            names_df.loc[(names_df['infection_stage'] == 'E') & (names_df['vaccination_stage'] == 'unvaccinated'), 'proportion_ic'] = float(ic_curr.loc['I_unvaccinated'].values)
            names_df.loc[(names_df['infection_stage'] == 'E') & (names_df['vaccination_stage'] == '1dose'), 'proportion_ic'] = float(ic_curr.loc['I_vaccinated'].values)
            names_df.loc[(names_df['infection_stage'] == 'R') & (names_df['vaccination_stage'] == 'unvaccinated'), 'proportion_ic'] = float(ic_curr.loc['R_unvaccinated'].values)
            names_df.loc[(names_df['infection_stage'] == 'H') & (names_df['vaccination_stage'] == 'unvaccinated'), 'proportion_ic'] = float(ic_curr.loc['H_unvaccinated'].values)
            names_df.loc[(names_df['infection_stage'] == 'D') & (names_df['vaccination_stage'] == 'unvaccinated'), 'proportion_ic'] = float(ic_curr.loc['D_unvaccinated'].values)

            names_df['total_population'] = float(demo_flepi[demo_flepi['subpop'] == subpop]['population'].values)
            names_df['proportion_age_stratum'] = 0.0
            for age_stratum in names_df['age_strata'].unique():
                names_df.loc[names_df['age_strata'] == age_stratum, 'proportion_age_stratum'] = demo.loc[(demo['fips'] == fips) & (demo['age_strata'] == age_stratum), 'prop'].values[0]
            names_df['population'] = names_df['total_population'] * names_df['proportion_age_stratum'] * names_df['proportion_ic']

            y0[:, i] = names_df['population'].values

        return y0
