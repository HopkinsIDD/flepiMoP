# Diagnostic plotting scripts

We provide helper scripts to aid users in understanding model outputs and diagnosing simulations and iterations. These scripts may be set to run automatically after a model run, and are dependent on the model defined in the user's defined config file.&#x20;

The script `postprocess_snapshot.R` requires the following command line inputs:

* a user-defined config, `$CONFIG_PATH`
* a run index, `$FLEPI_RUN_INDEX`
* a path to the model output results, `$FS_RESULTS_PATH`
* a path to the flepiMoP repository, `$FLEPI_PATH`; and&#x20;
*   a list of outputs to plot, `$OUTPUTS`, by default the script provides diagnostics for the following model output files&#x20;

    ```r
    "hosp, hpar, snpi, hnpi, llik"
    ```

Plots of `hosp` output files show confidence intervals of model runs, against the provided ground truth data for inference runs, for each metapopulation node. `hnpi` and `snpi` plots provide violin plots of parameter values for each slot.&#x20;

Other scripts are included as more specific examples of post-processing, used for diagnostic tools. `processing_diagnostics.R` scripts provides a detailed diagnosis of inference model runs and fits.&#x20;

