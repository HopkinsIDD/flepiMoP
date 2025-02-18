# Synchronizing files: Syntax and Applications

The `flepimop` pipeline provides a uniform interface to synchronization tools: `sync`. In the standard library, we support three underlying tools:

 - `rsync`: generally for use on inputs and outputs between a local machine and an HPC
 - `aws s3 sync`: generally for long term record of outputs or external sharing
 - `git`: for version controlled elements, like pre-/post-processing scripts, configuration files, etc

Used directly, these underlying tools are flexible-but-complex. Via `flepimop sync`, we provide a simplified-but-limited interface, which makes it easy to accomplish typical tasks, but won't solve every last data transfer problem. For a particular project, you can specify multiple `sync` "protocols", associated with different tasks. By default, the first protocol will be used, but you can specify a specific one to execute from a collection. A `sync` section is defined by a top-level `sync` key, mapping to any number of keys (which name the protocols), each of which has a `type` key (indicating the underlying tool) and any necessary configuration options (see following sections for the necessary fields by type). So a typical `sync` section might look like:

```yaml
sync:
  protocolA: # the default protocol: fetch hpc results to local directory
    type: rsync
    source: host:someproj/model_output
    target: some/local/dir
  protocolB: # a protocol to store results to S3
    type: s3sync
    source: some/hpc/model_output
    target: //s3bucket/someproj/model_output
```

All of these tools have push vs pull modes, so protocols need to define that direction. We distinguish these by setting source and target locations (`rsync` and `s3sync`) or by setting mode (`git`). When invoking `sync`, however, you can supply a `--reverse` flag, which will swap the synchronization direction.

Generally `sync`-style operations support filters to include or exclude files. By default, all `sync` actions will include everything within the source definition (in the case of `git`, everything that is tracked), and if the source is a directory, it will be recursively crawled. To modify this behavior, you can use filters either as part of the protocol definition OR with options provided when invoking sync. See the (Filters)[#filters] section below for details about filtering, but in general `sync` uses the `rsync` conventions for including / excluding files.

## `rsync` mode

The `rsync` mode is intended to be used on a local machine to sync with another machine running an rsync server, for example an HPC system. Without special setup, you won't be able to initiate sync *from* that "other" system back to your local machine: you're local machine probably isn't running an rsync server. You can still get files from the HPC to that local machine, you just have to run the `flepimop sync` command on your machine.

### Example: Pushing Inputs

Let's say your have some inputs that you generate by hand on your personal machine, that you need to push to an HPC ahead of running work on it. You might define a protocol as:

```yaml
sync:
  pushlongleaf: # defines the protocol name
    type: rsync # defines this is an rsync protocol
    source: model_input # what *local* folder to sync from
    target: longleaf:~/flepiproject/model_input # what *remote* project folder to sync to
```

Note: we are assuming you have setup your `.ssh/config` file to define you username, credentials location, host details, etc - so here `longleaf` corresponds to the host name of an HPC system. If you haven't done that, `flepimop` makes no guarantees about handling prompts for username, password, etc when using `sync`.

When the files were ready, you could then run from your local project folder:

```bash
$ flepimop sync myconfig.yml
```

Or if necessary using the `-(-p)rotocol=pushlongleaf` option to identify `pushlongleaf` is the `sync` item to execute (when you have multiple protocols specified and `pushlongleaf` wasn't the first / default protocol).

### Example: Pulling Outputs

Now imagine you have run your flepimop pipeline on the HPC, and you want to pull the results back to your local machine to do some plotting or analysis. You might define a protocol as:

```yaml
sync:
  pulllongleaf: # defines the protocol name
    type: rsync # defines this is an rsync protocol
    source: "longleaf:~/flepiproject/model_output" # what *remote* project folder to sync from
    target: model_output # what *local* folder to sync to
    filters: '+ *.SOME_RUN_INDEX.*' # an optional match-only SOME_RUN_INDEX filter
```

then

```bash
$ flepimop sync myconfig.yml
```

would pull the results from "model_output" matching `SOME_RUN_INDEX`. If you're iterating on some model specification, you might be working a series of run indicies. Rather than revising the configuration file repeatedly, you could instead call:

```bash
$ flepimop sync -f'+ $FLEPI_RUN_INDEX' --target=model_output/$FLEPI_RUN_INDEX myconfig.yml
```

This would fetch only the results associated with the defined `$FLEPI_RUN_INDEX` and group them together in a corresponding subfolder of `model_output`.

Of course, if your local machine still had earlier results, `sync` will automatically understand that those files haven't changed and that it only needs to fetch new run results.

### Troubleshooting

...

## `aws s3 sync` mode

The `aws s3 sync` mode is intended to be used to get results to and from longterm storage on AWS S3. That should generally be snapshotting a "final" analysis run, rather troubleshooting results during development towards such a run. Use of this tool assume that you have already taken two steps. First, that `aws s3 sync` is available on the command line, which might require e.g. `module load s3` or adjusting your `$PATH` such that aws command line interface is available with having to provide a fully qualified location. Second, your credentials are setup such that you can directly invoked `aws s3 sync` without having to provide username, etc.

### Example: Pushing Results

Imagining that you've got some final results and its time to send them to S3 (e.g. for a dashboard to pull from). You could define a protocol as:

```yaml
sync:
  snapshots3: # defines the protocol name
    type: s3sync # defines this is an aws s3 sync protocol
    source: model_output # what folder to sync from
    target: //idd-inference-runs/myproject # what *remote* project folder to sync to
```

Note the distinction here where target starts with `//` - that defines that this end is the s3 bucket. A valid `s3sync` protocol requires at least one end to be an s3 bucket, and thus to start with `//`.

You could then

```bash
$ flepimop sync myconfig.yml
```

to send your outputs to the s3 bucket.

## `git` mode

Though `git` is fairly straightforward to use directly, we also provide an simplified `sync` mode associated with `git` to ensure a model has the latest code elements associated with it. Practically, this can be used on an either a local machine or HPC setup to ensure that you have the latest version, or that if you have made changes, those have been pushed to authorative reference.

In general, `git` mode is much simpler to specify and use than the other two options, since its for different concerns. An example configuration protocol looks like:

```yaml
sync:
  checkcode:
    type: git
```

The `git` mode is simply a wrapper around normal git operations and expects that you are dealing with a normal git flow for staging files, making commits, marking files to be ignored, etc. It will issue a warning and take the following actions when various conditions are met:

 - halt: there are staged-but-not-committed files (e.g. `git add/rm ...` operations, but not yet committed)
 - halt: there unstaged changes to tracked files
 - warn: changes to files which are untracked, but are also unignored

If there are no issues with the repository, `sync` will fetch the authoritative repository version, attempting to update the local repository. If there are any merge conflicts, the `sync` operation will fail and refer you to the normal process for resolving such conflicts.

## Filters

Filtering happens by applying include or exclude filters in sequence. A filter is a string that starts either with a "- " (an exclude filter), "+ " (an include filter), or neither of those (defaulting to an include filter). Filters can include `*` or `**` for file or directory globs - see the particular tool documentation for supported patterns.  We adopt the `rsync` convention where earlier filters in the sequence have precedence over filters applied later, which we adapt to other tools conventions as necessary. So an "+ *" as the first filter means "include everything" and has precedence over subsequent filters; similarly, a "- *" filter (meaning exclude everything) and would block all inclusions specified.

For convenience, when users provide **single include** filter (after resolving the configuration file(s) and any command line options), this is interpretted as "include whatever matches this filter, and then exclude everything else".

In configuration files, the filter key is `filters` within a supporting protocol type. The value of that key can be a single string or a list of strings (in either square-bracket or bullet form). The left-to-right (or top-to-bottom) order determines which filter is first vs last.

When invoked on the command line, you can also specify changes to the filters in a few ways:
 - `-(-f)ilter` option(s) to **override** any configuration file filters. To provide multiple stages of filters, simply provide the option multiple times: `-f'+ include.me' -f'- *.me'` would include a `include.me` and exclude all other `.me` files.
 - `-e|--fsuffix` and `-a|--fprefix` option(s) to prefix and/or suffix filter(s) to the core filter (which can be from the configuration file, or an via override `-f`s). If there are no configuration-based filters, these are equivalent to just using `-f` filters.
 - `--no-filter` overrides specified configuration filter(s) to be an empty list; cannot be combined with `-f|a|e` options.

## Applications: `gempyor` resume and continue operations

The `gempyor` approaches to projection and inference support resuming from previously completed work.

### Resuming Inference

### Communication Between Iterations

The pipeline uses files to communicate between different iterations. Currently, the following file types exist:

* seed
* init
* snpi
* spar
* seir
* hpar
* hnpi
* hosp
* llik

During each iteration, inference uses these files to communicate with the compartmental model and outcomes. The intent is that inference should only need to read and write these files, and that the compartmental model can handle everything else. In addition to the `global` versions of these files actually passed to the compartmental/reporting model, there exist `chimeric` versions used internally by inference and stored in memory. These copies are what inference interacts with when it needs to perturb values. While this design was chosen primarily to support modularity (a fixed communication boundary makes it easy to swap out the compartmental model), it has had a number of additional benefits.

### Bootstrapping

The first iteration of an MCMC algorithm is a special case, because we need to pull initial conditions for our parameters. We originally developed the model without inference in mind, so the compartmental model is already set up to read parameter distributions from the configuration file, and to draw values from those distributions, and record those parameters to file. We take advantage of this to bootstrap our MCMC parameters by running the model one time, and reading the parameters it generated from file.

### Resume from previous run

We can, instead of bootstrapping our first iteration, read in final values of a previous iteration. This allows us to resume from runs to save computational time and effectively continue iterating on the same chain. We call these **resumes**: inferred parameters are taken from a previous run and allowed to continue being inferred ;

Resumes take the following files (if they exist) from previous runs and uses them as the starting point of a new run:

* hnpi
* snpi
* seed

So a resume protocol for `sync` (to fetch previously computed results), might look something like:

```yaml
sync:
  resumerun:
    type: s3sync
    source: //mybucket/myproject/
    target: model_output
    filters: ["*hnpi*", "*snpi*", "*seed*", "- *"]
```

### Continuing projection

In addition to resuming parameters (above), we can also perform a **continuation resume**. In addition to resuming parameters and seeding, continuations also use the compartmental fits from previous runs. For a config starting at time $$t_s$$ continuing and resuming from a previous run, the compartmental states of the previous run at time $$t_s$$are used as the initial conditions of the continuation resume ;
