# Config writer

The model needs the configurations file to run (described in previous sections). These configs become lengthy and sometimes difficult to type manually. The config writer helps to generate configs provided the relevant files are present.

### Print Functions:

These functions are used to print specific sections of the configuration files.

#### print\_header

Used to generate the global header. For more information on global headers click [HERE](../../gempyor/model-implementation/introduction-to-configuration-files.md#global-header).

<table><thead><tr><th width="172.33333333333331">Variable name</th><th>Required (default value if optional)</th><th>Description</th></tr></thead><tbody><tr><td>sim_name</td><td><strong>Required</strong></td><td>Name of the configuration file to be generated. Generally based on the type of simulation</td></tr><tr><td>setup_name</td><td><strong>Optional</strong> (SMH)</td><td>Type of run - a Scenario Modeling Hub ("SMH") or Forecasting Hub ("FCH") Simulation.</td></tr><tr><td>disease</td><td><strong>Optional</strong> (covid19)</td><td>Pathogen or disease being simulated</td></tr><tr><td>smh_round</td><td><strong>Optional</strong> (NA)</td><td>Round number for Scenario Modeling Hub Submission</td></tr><tr><td>data_path</td><td><strong>Optional</strong> (data)</td><td>Folder path which contains where population data (size, mobility, etc) and ground truth data files are stored</td></tr><tr><td>model_output_dir_name</td><td><strong>Optional</strong> (model_output)</td><td>Folder path where the outputs of the simulated model is stored</td></tr><tr><td>sim_start_date</td><td><strong>Required</strong></td><td>Start date for model simulation</td></tr><tr><td>sim_end_date</td><td><strong>Required</strong></td><td>End date for model simulation</td></tr><tr><td>start_date_groundtruth</td><td><strong>Optional</strong> (NA)</td><td>Start date for fitting data for inference runs</td></tr><tr><td>end_date_groundtruth</td><td><strong>Optional</strong> (NA)</td><td>End date for fitting data for inference runs</td></tr><tr><td>nslots</td><td><strong>Required</strong></td><td>number of independent simulations to run</td></tr></tbody></table>

#### print\_spatial\_setup

Used to generate the spatial setup section of the configuration. For more information on spatial setup click [HERE](../../gempyor/model-implementation/introduction-to-configuration-files.md#spatial\_setup-section).

<table><thead><tr><th width="173.33333333333331">Variable name</th><th width="272">Required (default value if optional)</th><th>Description</th></tr></thead><tbody><tr><td>census_year</td><td><strong>Optional</strong> (2019)</td><td>The year of data uses to generate the geodata files for US simulations ?? [Unsure about this]</td></tr><tr><td>sim_states</td><td><strong>Required</strong></td><td>Vector of locations that will be modeled (US Specific?)</td></tr><tr><td>geodata_file</td><td><strong>Optional</strong> (geodata.csv)</td><td>Name of the geodata file which is imported</td></tr><tr><td>mobility_file</td><td><strong>Optional</strong> (mobility.csv)</td><td>Name of the mobility file which is imported</td></tr><tr><td>popnodes</td><td><strong>Optional</strong> (pop2019est)</td><td>Name of a column in the geodata file that specifies the population of every subpopulation column</td></tr><tr><td>nodenames</td><td><strong>Optional</strong> (subpop)</td><td>Name of a column in the geodata file that specifies the name of the subpopulation</td></tr><tr><td>state_level</td><td><strong>Optional</strong> (TRUE)</td><td>Specifies if the subpopulations are US states</td></tr></tbody></table>

#### print\_compartments

Used to generate the compartment list for each way a population can be divided.

<table><thead><tr><th width="172.33333333333331">Variable Name</th><th>Required (default value if optional)</th><th>Description</th></tr></thead><tbody><tr><td>inf_stages</td><td><strong>Optional</strong> (S,E,I1,I2,I3,R,W)</td><td>Various infection stages an individual can be in</td></tr><tr><td>vaccine_compartments</td><td><strong>Optional</strong> (unvaccinated, 1dose, 2dose, waned)</td><td>Various levels of vaccinations an individual can have</td></tr><tr><td>variant_compartments</td><td><strong>Optional</strong> (WILD, ALPHA, DELTA, OMICRON)</td><td>Variants of the pathogen</td></tr><tr><td>age_strata</td><td><strong>Optional</strong> (age0to17, age18to64, age65to100)</td><td>Different age groups, the population has been stratified in</td></tr></tbody></table>

#### Parts of the configuration files that are printed but not needed for FlepiMop runs (need to be mentioned for US or COVID-19 specific runs??)

### Spatial Setup:

* census year: year of geodata files
* modeled states (sim\_states): This has US state abbreviations. Do we include the names of the sub-populations in the geodata file? Eg: small\_province, large\_province
* state\_level: Specifies if the runs are run for US states
