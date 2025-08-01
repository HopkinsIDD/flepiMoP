---
description: |
    Both initial conditions and seeding can be used for similar purposes, but have
    a few differences that may affect which feature best suites your use case.
---

# Initial Conditions vs Seeding

Infectious disease models require some infections to kick off transmission dynamics. These can be present in the starting state for the simulation (as `initial_conditions`) or introduced over time (via `seeding`). The [Specifying Initial Conditions](./specifying-initial-conditions.md) and [Specifying Seeding](./specifying-seeding.md) guides provide a detailed breakdown, but this table highlights the major differences between these approaches to help you decide what to use for your situation.

|                         | Initial Conditions                                                                      | Seeding                                                                            |
|-------------------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Main purpose            | Specifying compartments sizes at the initial time.                                      | Instantaneously transferring between compartments at arbitrary times.              |
| Default functionality   | Each subpopulations entire population is placed in the first compartment.               | No seeding events occur.                                                           |
| Config section needed   | Optional, results in warning that default functionality used.                           | Optional.                                                                          |
| Required input files    | Depending on method could be a parquet or CSV file(s).                                  | Yes, a CSV describing seeding events with a format dependent on method used.       |
| Incidence or prevalence | Amounts specified are prevalence values, either as amounts or proportion of population. | Amounts specified are instantaneous incidence values.                              |
| Inference integration   | Yes, initial condition plugins can request model parameters.                            | None, seeding does not integrate well with inference yet.                          |
| Use cases               | Specifying the model's starting point.                                                  | Modeling importations, evolution of new strains, and modifying initial conditions. |
