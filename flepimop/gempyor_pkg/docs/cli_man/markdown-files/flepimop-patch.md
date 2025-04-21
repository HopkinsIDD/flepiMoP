# NAME

flepimop-patch - Merge configuration files.

# SYNOPSIS

**flepimop patch** \[OPTIONS\] \[CONFIG_FILES\]\...

# DESCRIPTION

Merge configuration files.

This command will merge multiple config files together by overriding the
top level keys in config files. The order of the config files is
important, as the last file has the highest priority and the first has
the lowest.

A brief example of the command is shown below using the sample config
files from the \`examples/tutorials\` directory. The command will merge
the two files together and print the resulting configuration to the
console.

\`\`\`bash \$ flepimop patch config_sample_2pop_modifiers_part.yml
config_sample_2pop_outcomes_part.yml \> config_sample_2pop_patched.yml
\$ cat config_sample_2pop_patched.yml write_csv: false stoch_traj_flag:
false jobs: 14 write_parquet: true first_sim_index: 1 config_src:
\[config_sample_2pop_modifiers_part.yml,
config_sample_2pop_outcomes_part.yml\] seir_modifiers: scenarios:
\[Ro_lockdown, Ro_all\] modifiers: Ro_lockdown: method:
SinglePeriodModifier parameter: Ro period_start_date: 2020-03-15
period_end_date: 2020-05-01 subpop: all value: 0.4 Ro_relax: method:
SinglePeriodModifier parameter: Ro period_start_date: 2020-05-01
period_end_date: 2020-08-31 subpop: all value: 0.8 Ro_all: method:
StackedModifier modifiers: \[Ro_lockdown, Ro_relax\] outcome_modifiers:
scenarios: \[test_limits\] modifiers: test_limits: method:
SinglePeriodModifier parameter: incidCase::probability subpop: all
period_start_date: 2020-02-01 period_end_date: 2020-06-01 value: 0.5
outcomes: method: delayframe outcomes: incidCase: source: incidence:
infection_stage: I probability: value: 0.5 delay: value: 5 incidHosp:
source: incidence: infection_stage: I probability: value: 0.05 delay:
value: 7 duration: value: 10 name: currHosp incidDeath: source:
incidHosp probability: value: 0.2 delay: value: 14 \`\`\`

# OPTIONS

**-c,** \--config PATH

:   Deprecated: configuration file(s) for this simulation

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

**\--stochastic** / \--non-stochastic

:   Run stochastic simulations?

**\--write-csv** / \--no-write-csv

:   write csv output? \[default: no-write-csv\]

**\--write-parquet** / \--no-write-parquet

:   write parquet output? \[default: write-parquet\]
