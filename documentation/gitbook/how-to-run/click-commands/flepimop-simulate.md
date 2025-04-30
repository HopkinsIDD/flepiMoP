# NAME

flepimop-simulate - Forward simulate a model using gempyor.

# SYNOPSIS

**flepimop simulate** \[OPTIONS\] \[CONFIG_FILES\]\...

# DESCRIPTION

Forward simulate a model using gempyor.

# OPTIONS

**-c,** \--config PATH

:   Deprecated: configuration file(s) for this simulation

**-p,** \--populations TEXT

:   Population(s) to run use in simulation.

**-s,** \--seir_modifiers_scenarios TEXT

:   override/select the transmission scenario(s) to run

**-d,** \--outcome_modifiers_scenarios TEXT

:   override/select the outcome scenario(s) to run

**-j,** \--jobs INTEGER RANGE

:   the parallelization factor \[default: 4; x\>=1\]

**-n,** \--nslots INTEGER RANGE

:   override the \# of simulation runs in the config file \[x\>=1\]

**\--in-id** TEXT

:   Unique identifier for the run

**\--out-id** TEXT

:   Unique identifier for the run

**\--in-prefix** TEXT

:   unique identifier for the run

**-i,** \--first_sim_index INTEGER RANGE

:   The index of the first simulation \[default: 1; x\>=1\]

**-m,** \--method TEXT

:   If provided, overrides seir::integration::method

**\--write-csv** / \--no-write-csv

:   write csv output? \[default: no-write-csv\]

**\--write-parquet** / \--no-write-parquet

:   write parquet output? \[default: write-parquet\]
