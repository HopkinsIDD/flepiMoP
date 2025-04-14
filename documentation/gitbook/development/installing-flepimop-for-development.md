---
description: |
    Instructions on how to install `flepiMoP` for development purposes, which uses a
    specific utility script that installs extras and force reinstall.
---

# Installing `flepiMoP` For Development

When developing `flepiMoP` it is helpful to install it in a way that gives developers more control over the development environment. To assist developers with this there is the `bin/flepimop-install-dev` helper script which wraps the `bin/flepimop-install` script with some defaults. 

1. First obtain a clone of `flepiMoP` where you plan on doing development work:

```shell
$ git clone git@github.com:HopkinsIDD/flepiMoP.git
Cloning into 'flepiMoP'...
remote: Enumerating objects: 28538, done.
remote: Counting objects: 100% (3449/3449), done.
remote: Compressing objects: 100% (857/857), done.
Receiving objects: 100% (28538/28538), 146.00 MiB | 37.49 MiB/s, done.
remote: Total 28538 (delta 2915), reused 2808 (delta 2589), pack-reused 25089 (from 2)
Resolving deltas: 100% (14847/14847), done.
```

Or replace `git@github.com:HopkinsIDD/flepiMoP.git` with the appropriate URI for a fork if applicable.

2. Then change directory into this clone, checkout a branch to do development on, and then run the `bin/flepimop-install-dev` script. For more details on branch naming and GitHub usage please refer to the [Git and GitHub Usage](./git-and-github-usage.md) documentation.

```shell
$ cd flepiMoP
$ git checkout -b feature/XYZ/my-cool-new-thing
$ ./bin/flepimop-install-dev
```

The `bin/flepimop-install-dev` script is a thin wrapper around the standard `bin/flepimop-install` script that:

* Installs `flepiMoP` to a conda environment located inside of this clone in a directory named `venv/`,
* Force reinstalls `flepiMoP` which will wipe out an existing conda environment with a new one, and
* Install all extra/optional dependencies for `gempyor` (like `pytest`, `black`, `pylint`, etc.) that are particularly useful for python development.

3. Then you can activate this conda environment with:

```shell
conda activate venv/
```

You can verify that the installation was made to this local conda environment with:

```shell
$ which flepimop
/path/to/your/dev/flepiMoP/venv/bin/flepimop
```
 
4. If you need to refresh your development install, for example if you edit the R packages or add a new dependency, you can simply rerun the `bin/flepimop-install-dev` script which will wipe the previous conda environment and create a fresh new one.
