from pathlib import Path

from gempyor.utils import read_directory


project_path = Path(__file__).parent

simulation_output = (
    project_path
    / "model_output"
    / "three_state_state_varied_Ro_state_varied_incidH"
    / "sim"
)
if not simulation_output.exists():
    raise FileNotFoundError(
        f"Simulation output directory {simulation_output} does not exist. "
        "Please run the simulation first."
    )

hosp = read_directory(simulation_output, filters="hosp")
hosp = hosp[["date", "subpop", "hospitalizations_curr"]]
hosp = hosp.rename(columns={"hospitalizations_curr": "incidH"})
hosp["incidH"] = hosp["incidH"].astype(int)
hosp.to_csv(project_path / "model_input" / "ground_truth_hospitalizations.csv", index=False)
