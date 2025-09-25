# initial_condition

## Rationale

This folder contains the files necessary to extrapolate the initial condition found by calibrating a non-age, non-state stratified SVIRHD model to the US influenza hospitalisation data to function as the initial condition of a FlepiMoP influenza model.

## Files

+ `initial-conditions.csv`: Contains the initial condition of a non-age, non-state stratified SVI2RHD model calibrated to the US data, defined as the fraction of the US population in disease state X in season Y on Aug 1. Columns: 'disease state', 'season', 'mean'.

+ `geodata_2019_agestrat.csv`: Contains the US demography per age group and fips.

+ `sample_initial_conditions.py`: Script containing a "plugin initial condition" for FlepiMoP. Uses the calibrated initial conditions, the 'seasons' keyword in the config under 'initial_condition', and the fips+age demography to compute the appropriate number of inidividuals in each FlepiMoP compartment.