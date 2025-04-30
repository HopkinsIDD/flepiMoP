NAME
====

flepimop-modifiers-config-plot - Plot the seir/outcome modifiers affects
on\...

SYNOPSIS
========

**flepimop modifiers config-plot** \[OPTIONS\]

DESCRIPTION
===========

Plot the seir/outcome modifiers affects on the parameters.

This command will plot the activation of the modifiers and the parsed
parameters from the config file. This command will produce several PDF
files in the current directory contain the plots:

1\. {outocmes/seir}\_modifiers_activation\_{subpop}.pdf: Contains plots
of the activation of the modifiers for the given subpopulation vs time.
2. unique_parsed_parameters\_{run_id}.pdf: Contains plots of the parsed
transition rates with modifiers applied vs time. 3.
outcomesNPIcaveat.pdf: Contains plots of the outcomes affected by
modifiers vs time.

OPTIONS
=======

**-c,** \--config PATH

:   configuration file for this simulation

**-p,** \--project_path PATH

:   path to the flepiMoP directory \[required\]

**\--id,** \--id TEXT

:   Unique identifier for this run

**\--nsamples** INTEGER

:   Number of samples to draw

**\--subpop** TEXT

:   Subpopulation to plot, if not set then all are plotted
