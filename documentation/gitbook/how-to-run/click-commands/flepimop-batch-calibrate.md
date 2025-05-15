# NAME

flepimop-batch-calibrate - Submit a calibration job to a batch system.

# SYNOPSIS

**flepimop batch-calibrate** \[OPTIONS\] \[CONFIG_FILES\]\...

# DESCRIPTION

Submit a calibration job to a batch system. This job makes it
straightforward to submit a calibration job to a batch system. The job
will be submitted with the given configuration file and additional
options. The general steps this tool follows are:  1) Generate a unique
job name from the configuration and timestamp, 2) Determine the
outcome/SEIR modifier scenarios to use, 3) Determine the batch system to
use and required job size/resources/time limit, 4) Write a
\'manifest.json\' with job metadata, write the config used to a file,
and checkout a new branch in the project git repository, 5) Loop over
the outcome/SEIR modifier scenarios and submit a job for each scenario.
To get a better understanding of this tool you can use the
\`\--dry-run\` flag which will complete all of steps described above
except for submitting the jobs. Or if you would like to test run the
batch scripts without submitting to slurm or other batch systems you can
use the \`\--local\` flag which will run the \"batch\" job locally (only
use for small test jobs). Here is an example of how to use this tool
with the \`examples/tutorials/\` directory:  \`\`\`bash \$ flepimop
batch-calibrate \# The paths and conda environment to use \--flepi-path
\$FLEPI_PATH \--project-path \$FLEPI_PATH/examples/tutorials
\--conda-env flepimop-env \# The size of the job to run \--blocks 1
\--chains 50 \--samples 100 \--simulations 500 \# The time limit for the
job \--time-limit 8hr \# The batch system to use, equivalent to
\`\--batch-system slurm\` \--slurm \# Resource options \--nodes 50
\--cpus 2 \--memory 4GB \# Batch system specific options can be provided
via \`\--extra\` \--extra partition=normal \--extra
email=bob@example.edu \# Only run a dry run to see what would be
submitted for the config \--dry-run -vvv
config_sample_2pop_inference.yml \`\`\`

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

**\--flepi-path** PATH

:   Path to the flepiMoP directory being used. \[required\]

**\--project-path** PATH

:   Path to the project directory being used. \[required\]

**\--conda-env** TEXT

:   The conda environment to use for the job.

**\--blocks** INTEGER RANGE

:   The number of sequential blocks to run per a chain. \[x\>=1;
    required\]

**\--chains** INTEGER RANGE

:   The number of chains or walkers, depending on inference method, to
    run. \[x\>=1; required\]

**\--samples** INTEGER RANGE

:   The number of samples per a block. \[x\>=1; required\]

**\--simulations** INTEGER RANGE

:   The number of simulations per a block. \[x\>=1; required\]

**\--time-limit** DURATION

:   The time limit for the job. If units are not specified, minutes are
    assumed.

**\--batch-system** TEXT

:   The name of the batch system being used.

**\--local**

:   Flag to use the local batch system. Equivalent to \`\--batch-system
    local\`.

**\--slurm**

:   Flag to use the slurm batch system. Equivalent to \`\--batch-system
    slurm\`.

**\--cluster** TEXT

:   The name of the cluster being used, only needed if cluster info is
    required.

**\--nodes** INTEGER RANGE

:   Override for the number of nodes to use. \[x\>=1\]

**\--cpus** INTEGER RANGE

:   Override for the number of CPUs per node to use. \[x\>=1\]

**\--memory** MEMORY

:   Override for the amount of memory per node to use in MB.

**\--from-estimate** PATH

:   The path to a previous estimation file to use for the job. This will
    override the job resources and time limit given.

**\--estimate**

:   Should this be submitted as an estimation job? If this flag is given
    then several jobs will be submitted with smaller sizes to estimate
    the time and resources needed for the full job. A time limit and
    memory requirement must still be given, but act as upper bounds on
    estimation jobs.

**\--estimate-runs** INTEGER RANGE

:   The number of estimation runs to perform. Must be at least 6 due to
    the estimation method, but more runs will provide a better estimate.
    \[x\>=6\]

**\--estimate-interval** FLOAT RANGE

:   The size of the prediction interval to use for estimating the
    required resources. Must be between 0 and 1. \[0.0\<=x\<=1.0\]

**\--estimate-vary** \[blocks\|chains\|simulations\]

:   The job size fields to vary for estimating the resources needed for
    the job. This should be a subset of the job size fields.

**\--estimate-factors** \[blocks\|total_samples\|simulations_per_chain\|samples\|total_simulations\|simulations\|samples_per_chain\|chains\]

:   The factors to use for estimating the resources needed for the job.
    This should be a subset of the job size fields. Also keep in mind to
    avoid using colinear factors, i.e. blocks and simulations per a
    block. Doing so can lead to unstable estimates.

**\--estimate-measurements** \[cpu\|memory\|time\]

:   The measurements to use for estimating the resources needed for the
    job.

**\--estimate-scale-upper** FLOAT

:   The upper scale to use for estimating the resources needed for the
    job. This is the factor to scale the job size by to get the upper
    bound for the estimation job sizes.

**\--estimate-scale-lower** FLOAT

:   The lower scale to use for estimating the resources needed for the
    job. This is the factor to scale the job size by to get the lower
    bound for the estimation job sizes.

**\--skip-manifest**

:   Flag to skip writing a manifest file, useful in dry runs.

**\--skip-checkout**

:   Flag to skip checking out a new branch in the git repository, useful
    in dry runs.

**\--debug**

:   Flag to enable debugging in batch submission scripts.

**\--extra** TEXT

:   Extra options to pass to the batch system. Please consult the batch
    system documentation for valid options.

**-v,** \--verbose

:   The verbosity level to use for this command.

**\--dry-run**

:   Should this command be run using dry run?
