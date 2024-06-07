---
description: >-
  (This section describes the location and contents of the additional output
  files produced during an inference model run)
---

# Inference Model Output

### Updates to other files

### \LLIK (inference runs only)

During inference runs, an additional file type, `llik`, is created, which is described in the [Inference Model Output](inference-model-output.md) section.&#x20;

These files contain the log-likelihoods of the model simulation for each subpopulation, as well as some diagnostics on acceptance.

The meanings of the columns are:

`ll` - These values are the log-likelihoods of data given the model and parameter values for a single subpopulation (in `subpop` column).&#x20;

`filename` - ...

`subpop` - The values of this column are the names of the nodes from the `geodata` file.

`accept` - Either 0 or 1, depending on whether the parameters during this iteration of the simulation were accepted (1) or rejected (0) in that subpopulation.&#x20;

`accept_avg` -&#x20;

`accept_prob` -&#x20;





For inference runs, `...` _flepiMoP_ produces one file per parallel slot, for both global and chimeric outputs...

```
flepimop_sample
├── model_output
│   ├── seir
│   ├── spar
│   ├── snpi
│   └── llik
│       └── sample_2pop
│           └── None
│               └── 2023.05.24.02/12/48.
│                   ├── chimeric
│                   └── global
│                       ├── final
│                       │   └── 000000001.2023.05.24.02/12/48..llik.parquet
│                       └── intermediate
│                           └── 000000001.000000001.2023.05.24.02/12/48..llik.parquet
```
