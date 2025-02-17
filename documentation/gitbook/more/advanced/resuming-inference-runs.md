# Synchronizing files: Syntax and Applications

The `flepimop` pipeline provides a uniform interface to synchronization tools: `sync`. In the standard library, we support three underlying tools:

 - `rsync`: generally for use on inputs and outputs between a local machine and an HPC
 - `aws s3 sync`: generally for long term record of outputs or external sharing
 - `git`: for version controlled elements, like pre-/post-processing scripts, configuration files, etc

Used directly, these underlying tools are quite flexible-but-complex. Via `flepimop sync`, we provide a simplified-but-limited interface, which makes it easy to accomplish typical tasks, but won't solve every last data transfer problem. For a particular project, you can specify multiple `sync` "protocols", associated with different tasks. By default, the first protocol will be used, but you can specify a specific one to execute from a collection.

All of these tools have push vs pull modes, so protocols need to define that direction. We distinguish these by setting source and target locations (rsync and aws s3 sync) or by setting mode (git). When invoking sync, however, you can supply a `--reverse` flag, which will swap the synchronization direction.

Lastly, generally `sync`-style operations support filters to include or exclude files. By default, all `sync` actions will include everything within the source definition (in the case of `git`, everything that is tracked). If the source is a directory, it will be recursively crawled. To modify this behavior, you can use filters either as part of the protocol definition OR with options provided when invoking sync (which override any in the protocol definition). See the (Filters)[#filters] section below for details about filtering, but in general `sync` uses the `rsync` conventions for including / excluding files.

## `rsync` mode

The `rsync` mode is intended to be used on a local machine to sync with another machine running an rsync server, for example an HPC system. Without special setup, you won't be able to initiate sync *from* that "other" system back to your local machine: you're local machine probably isn't running an rsync server. You can still get files from the HPC to that local machine, you just have to run the `flepimop sync` command on your machine.

### Example: Pushing Inputs

Let's say your have some inputs that you generate by hand on your personal machine, that you need to push to an HPC ahead of running work on it. You might define a protocol as:

```yaml
sync:
  pushlongleaf: # defines the protocol name
    type: rsync # defines this is an rsync protocol
    source: model_input # what *local* folder to sync from
    target: "longleaf:~/flepiproject/model_input" # what *remote* project folder to sync to
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
    filter: SOME_RUN_INDEX # an optional matching filter
```

then

```bash
$ flepimop sync myconfig.yml
```

would pull the results from "model_output" matching `SOME_RUN_INDEX`. If you're iterating on some model specification, you might be working a series of run indicies. Rather than revising the configuration file repeatedly, you could instead call:

```bash
$ flepimop sync --filter=$FLEPI_RUN_INDEX --target=model_output/$FLEPI_RUN_INDEX myconfig.yml
```

This would fetch only the results associated with the defined `$FLEPI_RUN_INDEX` and group them together in a corresponding subfolder of `model_output`.

Of course, if your local machine still had earlier results, `sync` will automatically understand that those files haven't changed and that it only needs to fetch new run results.

### Troubleshooting

...

## `aws s3 sync` mode

The `aws s3 sync` mode is intended to be used to get results to and from longterm storage on AWS S3. That should generally be snapshotting a "final" analysis run, rather troubleshooting results during development towards such a run.

### Example: Pushing Results

Imagining that you've got some final results and its time to send them to S3 (e.g. for a dashboard to pull from). You could define a protocol as:

```yaml
sync:
  snapshots3: # defines the protocol name
    type: s3sync # defines this is an aws s3 sync protocol
    source: model_output # what folder to sync from
    target: idd-inference-runs/myproject # what *remote* project folder to sync to
```

Note: we are assuming you have setup your WHATEVER FILES ARE NECESSARY FOR S3 ACCESS.

You could then

```bash
$ flepimop sync myconfig.yml
```

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

## Filtering

TODO

## Applications

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

#### Resume from previous run

We can, instead of bootstrapping our first iteration, read in final values of a previous iteration. This allows us to resume from runs to save computational time and effectively continue iterating on the same chain. We call these **resumes**: inferred parameters are taken from a previous run and allowed to continue being inferred ;

Resumes take the following files (if they exist) from previous runs and uses them as the starting point of a new run:

* hnpi
* snpi
* seed

### Continuing projection

In addition to resuming parameters (above), we can also perform a **continuation resume**. In addition to resuming parameters and seeding, continuations also use the compartmental fits from previous runs. For a config starting at time $$t_s$$ continuing and resuming from a previous run, the compartmental states of the previous run at time $$t_s$$are used as the initial conditions of the continuation resume ;
