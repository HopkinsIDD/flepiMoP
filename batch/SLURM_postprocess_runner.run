#!/bin/bash
#SBATCH --nodes 1                      # Node PER JOB. Apparently this works on MARC
#SBATCH --ntasks-per-node 1            # MPI sense -> just 1.
#SBATCH --cpus-per-task 48             # No more than the number of CPU in a node. CPU <- coeur dans slurm
# #SBATCH --qos=week                     # Echo is authorized to launch week-long job
# #SBATCH --reservationname=COVID19

set -x

module purge
# on marcc this is anaconda3/2022.05 to circumvent anaconda python bug. Otherwise that is just anaconda 
module load anaconda
module load anaconda3/2022.05
conda activate flepimop-env
# in case conda not found
#source /home/jcblemai/.bashrc
#source ~/miniconda3/etc/profile.d/conda.sh # loading conda in case
#eval "$(conda shell.bash hook)"
which python
which Rscript

# aws cli to export plots (location according to instruction)
export PATH=~/aws-cli/bin:$PATH

# move all the slurm logs into the right folder:
mv slurm-$SLURM_ARRAY_JOB_ID_${SLURM_ARRAY_TASK_ID}.out $FS_RESULTS_PATH/slurm-$SLURM_ARRAY_JOB_ID_${SLURM_ARRAY_TASK_ID}.out

curl \
  -H "Title: $FLEPI_RUN_INDEX Done ✅" \
  -H "Priority: urgent" \
  -H "Tags: warning,snail" \
  -d "Hopefully the results look alright" \
  ntfy.sh/flepimop_alerts

# get the slack credentials
source /data/struelo1/flepimop-code/slack_credentials.sh # populate $SLACK_TOKEN

rm -r pplot
mkdir pplot

source $FLEPI_PATH/batch/postprocessing-scripts.sh

cp -R pplot $FS_RESULTS_PATH
if [[ $S3_UPLOAD == "true" ]]; then
    aws s3 cp --quiet pplot $S3_RESULTS_PATH/pplot
fi

