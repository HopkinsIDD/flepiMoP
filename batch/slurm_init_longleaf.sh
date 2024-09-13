USERO=$( echo $USER | awk '{ print substr($0, 1, 1) }' )
USERN=$( echo $USER | awk '{ print substr($0, 2, 1) }' )
USERDIR="/users/$USERO/$USERN/$USER"

echo ">>> Loading required modules"

module purge
module load gcc/9.1.0
module load anaconda/2023.03
module load r


echo ">>> Creating conda environement at $USERDIR/flepimop-env"

conda create --channel conda-forge --prefix $USERDIR/flepimop-env python=3.10
conda activate $USERDIR/flepimop-env
pip install --editable $USERDIR/flepiMoP/flepimop/gempyor_pkg


echo ">>> Installing required R packages"

R -e "install.packages(c('readr', 'sf', 'lubridate', 'tidyverse', 'gridExtra', 'reticulate', 'truncnorm', 'xts', 'ggfortify', 'flextable', 'doParallel', 'foreach', 'optparse', 'arrow', 'devtools', 'cowplot', 'ggraph'), repos = 'https://cloud.r-project.org')"
Rscript $USERDIR/flepiMoP/build/local_install.R


echo ">>> Loading/setting environment variables"

source $USERDIR/flepiMoP/batch/slurm_prerun_longleaf.sh

echo ">>> Done"
