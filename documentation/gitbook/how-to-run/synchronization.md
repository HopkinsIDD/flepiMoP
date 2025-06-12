# Synchronizing Files

The `flepimop` pipeline typically requires a large set of input files and can produce a large set of output files, particularly for calibration runs, which can be challenging to move around to different machines/storage systems for analysis or backup. To address this need the `flepimop sync` tool can move files to and from the working location. Currently the `flepimop sync` command supports three underlying tools:

 - `rsync`: Generally for use on inputs and outputs between a local machine and an HPC,
 - `aws s3 sync`: Generally for long term record of outputs or external sharing, and
 - `git`: For version controlled elements, like pre-/post-processing scripts, configuration files, model inputs, etc.

Used directly, these underlying tools are flexible, but complex. When abstracted by `flepimop sync` these tools can be used indirectly with a simplified, but limited interface. This trade off makes it easy to have reproducible and cross environment tooling support but fails to address more complicated use cases.

For a particular project multiple `sync` "protocols" can be defined, associated with different tasks. By default the first protocol will be used, but users can also specify a sync "protocol" to use explicitly. A `sync` section is defined by a top-level `sync` key, mapping to any number of keys (which name the protocols). Each "protocol" has a `type` key, indicating the underlying tool, and any necessary configuration options (see following sections for the necessary fields by type). An example of what a sync section in a configuration file might look like is:

```yaml
sync:
  protocolA: # the default protocol: fetch hpc results to local directory
    type: rsync
    source: host:someproj/model_output
    target: some/local/dir
  protocolB: # a protocol to store results to S3
    type: s3sync
    source: some/hpc/model_output
    target: s3://s3bucket/someproj/model_output
```

All of these tools have push vs pull modes, so protocols need to define that direction. We distinguish these by setting source and target locations for `rsync` and `s3sync` or by setting mode for `git`. The directionality can be flipped with the `--reverse` flag.

Both `rsync` and `s3sync` protocols support filters to include or exclude files. By default, all `sync` actions will include everything within the source definition, and if the source is a directory, it will be recursively crawled. For the `git` protocol which does not support filters users can take advantage of [`.gitignore` files provided by `git`](https://git-scm.com/docs/gitignore). To modify this behavior for other protocols, you can use filters either as part of the protocol definition OR with options provided when invoking sync. See the (Filters)[#filters] section below for details about filtering, but in general `sync` uses the `rsync` conventions for including/excluding files.

## Protocols

### `rsync` Protocol

The `rsync` mode is intended to be used on a local machine to sync with another machine running an rsync server, for example an HPC system. Without special setup, you won't be able to initiate sync *from* that "other" system back to your local machine. Typically personal laptops will not be running an rsync server, but shared resources like HPCs will. You can still get files from the HPC to that local machine, you just have to run the `flepimop sync` command on your machine possibly with the `--reverse` flag depending on how the protocol is configured.

A template for configuring an `rsync` protocol is:

```yaml
sync:
  <protocol name>:              # User supplied name for referencing protocol explicitly
    type: rsync
    source: <path to source>   # A path to a source, can be a local directory like `/abc/def` or `~/ghi` or a remote directory like `user@machine:~/xyz
    target: <path to target>   # A path to a target with the same format as source
    filters:                   # An optional set of filters to apply in order, if not provided then no filters are used.
      - <optional filter one>
      - <optional filter two>
      ...
```

While the `source` and `target` values should have the same format, they should not be the same value as this would result in a no-op. Typically one of `source` or `target` will be a remote directory on an HPC environment and the other will be a local directory so the sync protocol can be used to move model inputs/outputs from compute environment to local environments, or vise versa. The examples below will help guide you through the details of how to set this up with some concrete applications.

#### Example: Pushing Inputs

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

#### Example: Pulling Outputs

Now imagine you have run your flepimop pipeline on the HPC, and you want to pull the results back to your local machine to do some plotting or analysis. You might define a protocol as:

```yaml
sync:
  pulllongleaf: # defines the protocol name
    type: rsync # defines this is an rsync protocol
    source: "longleaf:~/flepiproject/model_output" # what *remote* project folder to sync from
    target: model_output # what *local* folder to sync to
    filters: '+ *.SOME_RUN_INDEX.*' # an optional match-only SOME_RUN_INDEX filter
```

Then

```bash
$ flepimop sync myconfig.yml
```

would pull the results from "model_output" matching `SOME_RUN_INDEX`. If you're iterating on some model specification, you might be working a series of run indices. Rather than revising the configuration file repeatedly, you could instead call:

```bash
$ flepimop sync -f'+ $FLEPI_RUN_INDEX' --target=model_output/$FLEPI_RUN_INDEX myconfig.yml
```

This would fetch only the results associated with the defined `$FLEPI_RUN_INDEX` and group them together in a corresponding subfolder of `model_output`.

Of course, if your local machine still had earlier results, `sync` will automatically understand that those files haven't changed and that it only needs to fetch new run results.

### `s3sync` Protocol

The `aws s3 sync` mode is intended to be used to get results to and from long term storage on AWS S3. That should generally be snapshotting a "final" analysis run, rather troubleshooting results during development towards such a run. Use of this tool assume that you have already taken two steps. First, that `aws s3 sync` is available on the command line, which might require e.g. `module load s3` or adjusting your `$PATH` such that aws command line interface is available with having to provide a fully qualified location. This should be handled for you automatically by `batch/hpc_init` on either Longleaf or Rockfish HPCs. Second, your credentials are setup such that you can directly invoked `aws s3 sync` without having to provide username, etc.

#### Example: Pushing Results

Imagining that you've got some final results and its time to send them to S3 (e.g. for a dashboard to pull from). You could define a protocol as:

```yaml
sync:
  snapshots3: # defines the protocol name
    type: s3sync # defines this is an aws s3 sync protocol
    source: model_output # what folder to sync from
    target: s3://idd-inference-runs/myproject # what *remote* project folder to sync to
```

Note the distinction here where target starts with `s3://` - that defines that this end is the s3 bucket. A valid `s3sync` protocol requires at least one end to be an s3 bucket, and thus to start with `s3://`. Furthermore note that there is no trailing slash for the `model_output` directory, similarly to the `rsync` protocol, this tells the `flepimop rsync` command to sync the whole `model_output` directory to `s3://idd-inference-runs/myproject`. If there had been a trailing slash (i.e. `model_output/`) then the contents of that directory would have been synced to `s3://idd-inference-runs/myproject` instead.

You could then

```bash
$ flepimop sync myconfig.yml
```

To send your outputs to the s3 bucket.

### `git` Protocol

Though `git` is fairly straightforward to use directly, we also provide an simplified `sync` mode associated with `git` to ensure a model has the latest code elements associated with it. Practically, this can be used on an either a local machine or HPC setup to ensure that you have the latest version, or that if you have made changes, those have been pushed to authoritative reference.

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

Filtering happens by applying include or exclude filters in sequence. A filter is a string that starts either with a "- " for an exclude filter, "+ " for an include filter, "s " for a substring filter, or none of those which defaults to an include filter. Filters can include `*` or `**` for file or file/directory globs - see the particular tool documentation for more supported patterns. We adopt the `rsync` convention where earlier filters in the sequence have precedence over filters applied later, which flepiMoP translates to other tools conventions as necessary. So an "+ *" as the first filter means "include everything" and has precedence over subsequent filters. Similarly, an initial "- *" filter, meaning exclude everything, would block all subsequent inclusions specified. Substring filters are resolved by the protocol to only include paths with a user specified substring in them.

For convenience, when users provide **only include** filters (after resolving all configuration file(s) and any command line options), this is interpreted as "include whatever matches this filter, and then exclude everything else". This happens by automatically adding a `- **` as the final (lowest precedence) filter.

In configuration files, the filter key is `filters` within a supporting protocol type. The value of that key can be a single string or a list of strings (in either square-bracket or bullet form). The left-to-right (or top-to-bottom) order determines which filter is first vs last.

When invoked on the command line, you can also specify changes to the filters in a few ways:
 - `-(-f)ilter` option(s) to **override** any configuration file filters. To provide multiple stages of filters, simply provide the option multiple times: `-f'+ include.me' -f'- *.me'` would include a `include.me` and exclude all other `.me` files.
 - `-e|--fsuffix` and `-a|--fprefix` option(s) to prefix and/or suffix filter(s) to the core filter (which can be from the configuration file, or an via override `-f`s). If there are no configuration-based filters, these are equivalent to just using `-f` filters.
 - `--no-filter` overrides specified configuration filter(s) to be an empty list; cannot be combined with `-f|a|e` options.

## Troubleshooting

Before running a `flepimop sync` command for the first time it is helpful to take advantage of the `--dry-run` flag to see what the command would do without actually running the command. The output of this can be quite verbose, especially when using `-vvv` for full verbosity, so it can be helpful to pipe the output of the dry run to a text file for inspection.

## Applications: `gempyor` resume and continue operations

The `gempyor` approaches to projection and inference support resuming from previously completed work.

### Resuming Inference

#### Communication Between Iterations

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

#### Bootstrapping

The first iteration of an MCMC algorithm is a special case, because we need to pull initial conditions for our parameters. We originally developed the model without inference in mind, so the compartmental model is already set up to read parameter distributions from the configuration file, and to draw values from those distributions, and record those parameters to file. We take advantage of this to bootstrap our MCMC parameters by running the model one time, and reading the parameters it generated from file.

#### Resume from previous run

Instead of bootstrapping our first iteration, flepiMoP supports reading in final values of a previous iteration. This allows us to resume from runs to save computational time and effectively continue iterating on the same chain. We call these **resumes**, in which inferred parameters are taken from a previous run and allowed to continue being inferred.

Resumes take the following files, if they exist, from previous runs and uses them as the starting point of a new run:

* hnpi
* snpi
* seed

So a resume protocol for `sync`, to fetch previously computed results, might look something like:

```yaml
sync:
  resumerun:
    type: s3sync
    source: s3://mybucket/myproject/
    target: model_output
    filters: ["*hnpi*", "*snpi*", "*seed*", "- *"]
```

#### Continuing projection

In addition to resuming parameters, we can also perform a **continuation resume**. In addition to resuming parameters and seeding, continuations also use the compartmental fits from previous runs. For a config starting at time $$t_s$$ continuing and resuming from a previous run, the compartmental states of the previous run at time $$t_s$$are used as the initial conditions of the continuation resume.

### Saving Model Outputs To AWS S3 With `flepimop batch-calibrate`

For details on how to do this please refer to the [Saving Model Outputs On Batch Inference Job Finish](./advanced-run-guides/running-on-a-hpc-with-slurm.md#saving-model-outputs-on-batch-inference-job-finish) guide for the latest information.
