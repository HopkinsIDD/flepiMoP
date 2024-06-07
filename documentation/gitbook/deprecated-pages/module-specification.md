# Module specification

#### THIS IS DEPRECATED. GO TO HopkinsIDD/COVID19\_Minimal

### R interface basics

The python code will call your R scripts, setting some variable in the environment:

* `from_python`: truthy boolean, test for this to know if your code is run automatically.
* `ti_str, tf_str` model start and end as a string
* `foldername` the folder that contains everything related to the setup. You'll have to load `geodata.csv` from there. It include the `/` at the end.

```
if (!from_python) {         # or whatever values you use to test.
    ti_str <- '2020-01-31'
    tf_str <- '2020-08-31'
    foldername <- 'west-coast/'      
}
# write code here that uses what is above and can load more files.
```

**The code is run from the root folder of the repository.**

### Setup

A setup has a `name`, and this `name` is a also folder that contains file `geodata.csv` (see below).

### Modules

(and status if the current `R` implementation respect the specification)

#### Mobility (WIP)

* From R: dataframe named `mobility` with columns: `from, to, amount`. Relationships not specified will be set to zero. You can set different value for A -> B and B -> A (if you only specified A -> B, we'll assume B -> A = 0).
* From file: matrix to be imported with numpy as it is. Dimension: `(nnodes, nnnodes)` (may have a third dimension if time varying). First index is from, second is to, diagonal is zero (`mobility[ori, dest]`)
* From python: numpy matrix as file.

#### Population (DONE)

* From file: geodata.csv : specification of the spatial nodes, with at least column for the zero based index, the geoid or name, the population.

#### Importation (TODO)

* From R: dataframe named `importation` with column `date, to, amount` where **date is a string**, `to` contains a geoid and amount contains an integer.

#### NPI (DONE)

Different R scripts define the Nonpharmaceutical Intervention (NPI) to apply in the simulation. Based on the following system arguments, an R script will be called that generates the appropriate intervention. The start and end dates for each NPI needs to be specified (YYYY-MM-DD).

* None: No intervention, R0 reduction is 0
* SchoolClosure: School closure, counties randomly assigned an R0 reduction ranging from 16-30% (Jackson, M. et al., medRxiv, 2020)
* Influenza1918: Influenza social distancing as observed in 1918 Influenza. Counties are randomly assigned an R reduction value ranging from 44-65% (the most intense social distancing R0 reduction values from Milwaukee) (Bootsma & Ferguson, PNAS, 2007)
* Wuhan: Counties randomly assigned an R0 reduction based on values reported in Wuhan before and after travel ban during COVID-19 outbreak (R0 reduction of 81-88%) (Zhang, B., Zhou, H., & Zhou F. medRxiv, 2020; Mizumoto, R., Kagaya, K., & Chowell, G., medRxiv, 2020)
* TestIsolate: This intervention represents rapid testing and isolation of cases, similar to what was done in Wuhan at the beginning of the outbreak. It reduces R0 by 45-96%.
* Mild: This intervention has two sequential interventions: School closures, followed by a period of Wuhan-style lockdown followed by nothing.
* Mid: This intervention has three sequential interventions: School closures, followed by a period of Wuhan-style lockdown, followed by social distancing practices used during the 1918 Influenza pandemic
* Severe: This intervention has three sequential interventions: School closures, followed by a Wuhan-style lockdown, followed by rapid testing and isolation.

#### Transmission parameters (TODO)
