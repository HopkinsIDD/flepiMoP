
# START: your postprocessing scripts goes here.

Rscript $FLEPI_PATH/postprocessing/postprocess_snapshot.R -c $CONFIG_PATH --run-id $FLEPI_RUN_INDEX --results-path $FS_RESULTS_PATH --flepimop-repo $FLEPI_PATH

# END: your postprocessing scripts goes here.


# From chadi: this makes a plot of the llik files and send everything in pplot/ to slack with a little message from the bot.

# --fs-results-path . instead of --fs-results-path $FS_RESULTS_PATH so it can takes advantage of all simulations and not just the copied one.
python $FLEPI_PATH/postprocessing/postprocess_auto.py -c $CONFIG_PATH --run-id $FLEPI_RUN_INDEX --job-name $JOB_NAME --fs-results-path . --slack-token $SLACK_TOKEN --slack-channel $SLACK_CHANNEL

