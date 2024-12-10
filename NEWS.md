# December 11th, 2024

**Bug Fixes:**

- The HPC init script no longer fails if a conda environment is already active, GH-388.
- Stdout/stderr from `flepimop-inference-slot` called by `flepimop-inference-main` are piped to a log file via `system2` instead of pipes to support MINGW64, GH-289.

**Dependencies:**

- `click` minimum is now 8.1.7 (latest as of Aug 17, 2023).
- Added missing `h5py` dependency to `gempyor` requirements and specified `dask` dependency to include `dataframe` optional dependencies, GH-391.

**Deprecates:**

- `gempyor-simulate ...` in favor of `flepimop simulate ...`.
- Soft deprecated the `-c/--config_files` option (config file(s) are now *arguments* not options).

**New Features:**

- Basic support for multiple config files
- A `patch` command that takes multiple config files and yields the merged result
- Converted `gempyor`'s `setup.cfg` to the more modern `pyproject.toml`, GH-391. No user facing changes.
- Added `flepimop modifiers` subcommand with one action, `config-plot`, for plotting the effects of modifiers on a config, GH-404.

**Removes/Modifies:**

- `gempyor-(seir|outcomes) ...` - these were already no longer supported, just pruning entry points
