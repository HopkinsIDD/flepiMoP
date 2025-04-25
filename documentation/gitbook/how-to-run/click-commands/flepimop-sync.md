# NAME

flepimop-sync - Sync flepimop files between local and\...

# SYNOPSIS

**flepimop sync** \[OPTIONS\] \[CONFIG_FILES\]\...

# DESCRIPTION

Sync flepimop files between local and remote locations. For the filter
options, see \`man rsync\` for more information - sync supports basic
include / exclude filters, and follows the rsync precendence rules:
earlier filters have higher precedence.

All of the filter options (-a, -e, -f) can be specified multiple times
to add multiple filters. For the prefix and suffix filters, they are
first assembled into a list in the order specified and then added to the
beginning or end of the filter list in the config file. So e.g. \`-a \"+
foo\" -a \"- bar\"\` adds \[\`+ foo\`, \`- bar\`\] to the beginning of
the filter list, meaning the include filter \`+ a\` has higher
precedence than the exclude filter \`- bar\`.

# OPTIONS

**-p,** \--protocol TEXT

:   sync protocol to use from CONFIG_FILES

**-s,** \--source PATH

:   source directory to \'push\' changes from

**-t,** \--target PATH

:   target directory to \'push\' changes to

**-e,** \--fsuffix TEXT

:   Add a filter to the end of the filter list in CONFIG_FILES (same as
    \`-f\` if that list is empty)

**-a,** \--fprefix TEXT

:   Add a filter to the beginning of the filter list in CONFIG_FILES
    (same as \`-f\` if that list is empty)

**-f,** \--filter TEXT

:   replace the filter list in CONFIG_FILES; can be specified multiple
    times

**\--no-filter**

:   ignore all filters in config file

**\--reverse**

:   reverse the source and target directories

**\--mkpath**

:   Before syncing, manually ensure destination directory exists.

**-n,** \--dry-run

:   perform a dry run of the sync operation

**-v,** \--verbose

:   The verbosity level to use for this command.
