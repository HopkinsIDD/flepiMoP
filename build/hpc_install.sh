# Generic setup
set -e

# Cluster specific setup
if [[ $1 == "longleaf" ]]; then
    # Setup general purpose user variables needed for Longleaf
    USERO=$( echo $USER | awk '{ print substr($0, 1, 1) }' )
    USERN=$( echo $USER | awk '{ print substr($0, 2, 1) }' )
    USERDIR="/users/$USERO/$USERN/$USER"

    # Load required modules
    module purge
    module load gcc/9.1.0
    module load anaconda/2023.03
    module load git
elif [[ $1 == "rockfish" ]]; then
    # Setup general purspose user variables needed for RockFish
    USERDIR="/scratch4/struelo1/flepimop-code/$USER/"

    # Load required modules
    module purge
    module load gcc/9.3.0
    module load anaconda/2020.07
    module load git/2.42.0
else
    echo "The cluster name '$1' is not recognized, must be one of: 'longleaf', 'rockfish'."
    exit 1
fi

# Make sure the credentials is is where we expect and have the right perms
if [ ! -f "$USERDIR/slack_credentials.sh" ]; then
    echo "You need to place sensitive credentials in '$USERDIR/slack_credentials.sh'."
    exit 1
fi
chmod 600 $USERDIR/slack_credentials.sh
source $USERDIR/slack_credentials.sh

# Ensure we have a $FLEPI_PATH
if [ -z "${FLEPI_PATH}" ]; then
    echo "An explicit \$FLEPI_PATH was not provided, setting to '$USERDIR/flepiMoP'."
    export FLEPI_PATH="$USERDIR/flepiMoP"
fi

# Test that flepiMoP is located there
if [ ! -d "$FLEPI_PATH" ]; then
    echo "Did not find flepiMoP at '$FLEPI_PATH', cloning on your behalf."
    git clone git@github.com:HopkinsIDD/flepiMoP.git $FLEPI_PATH
elif [ ! -d "$FLEPI_PATH/.git" ]; then
    echo "The flepiMoP found at '$FLEPI_PATH' is not a git clone, unsure of how to proceed."
    exit 1
fi

# Setup the conda environment
if [ ! -d "$USERDIR/flepimop-env" ]; then
cat << EOF > $USERDIR/environment.yml
channels:
- conda-forge
- defaults
dependencies:
- python=3.10
- pip
- r-base>=4.4
- r-essentials
- r-devtools
- pyarrow=17.0.0
- r-arrow=17.0.0
# Manually specify this one because of the paths for libudunits2 on longleaf
- r-sf 
# This packages are probably missing from the DESCRIPTION of the R packages
- r-optparse
EOF
conda env create --prefix $USERDIR/flepimop-env --file $USERDIR/environment.yml
fi
conda activate $USERDIR/flepimop-env

# Check the conda environment is valid
WHICH_PYTHON=$( which python )
WHICH_R=$( which R )
WHICH_PYTHON_OKAY=$( echo "$WHICH_PYTHON" | grep "flepimop-env" | wc -l )
WHICH_R_OKAY=$( echo "$WHICH_R" | grep "flepimop-env" | wc -l )
if [[ "$WHICH_PYTHON_OKAY" -ne 1 ]]; then
    echo "The python found is '$WHICH_PYTHON', which does not contain the expected 'flepimop-env'."
    exit 1
fi
if [[ "$WHICH_R_OKAY" -ne 1 ]]; then
    echo "The R found is '$WHICH_R', which does not contain the expected 'flepimop-env'."
    exit 1
fi
PYTHON_ARROW_VERSION=$( python -c "import pyarrow; print(pyarrow.__version__)" )
R_ARROW_VERSION=$( Rscript -e "cat(as.character(packageVersion('arrow')))" )
COMPATIBLE_ARROW_VERSION=$( echo "$R_ARROW_VERSION" | grep "$PYTHON_ARROW_VERSION" | wc -l )
if [[ "$COMPATIBLE_ARROW_VERSION" -ne 1 ]]; then
    echo "The R version of arrow is '$R_ARROW_VERSION' and the python version is '$PYTHON_ARROW_VERSION'. These may not be compatible versions."
fi

# Install the gempyor package from local
pip install --force-reinstall $FLEPI_PATH/flepimop/gempyor_pkg

# Install the local R packages
INSTALL_R=$( mktemp )
cat << EOF > $INSTALL_R
library(devtools)
devtools::install_github('HopkinsIDD/flepiMoP', subdir='flepimop/R_packages/flepicommon')
devtools::install_github('HopkinsIDD/flepiMoP', subdir='flepimop/R_packages/flepiconfig')
devtools::install_github('HopkinsIDD/flepiMoP', subdir='flepimop/R_packages/inference')
EOF
Rscript $INSTALL_R
rm $INSTALL_R

# Set correct env vars
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
export TODAY=`date --rfc-3339='date'`

echo -n "Please set a project path (relative to '$USERDIR'): "
read PROJECT_PATH
export PROJECT_PATH="$USERDIR/$PROJECT_PATH"

echo -n "Please set a config path (relative to '$PROJECT_PATH'): "
read CONFIG_PATH
export CONFIG_PATH="$PROJECT_PATH/$CONFIG_PATH"

echo -n "Please set a validation date (today is $TODAY): "
read VALIDATION_DATE

echo -n "Please set a resume location: "
read RESUME_LOCATION

echo -n "Please set a flepi run index: "
read FLEPI_RUN_INDEX

# Done
cat << EOM
> The HPC install script has successfully finished.

If you are testing if this worked, say installing for the first time, you can use the inference example from the \`flepimop_sample\` repository:
\`\`\`bash
cd \$PROJECT_PATH
Rscript \$FLEPI_PATH/flepimop/main_scripts/inference_main.R -c \$CONFIG_PATH -j 1 -n 1 -k 1
\`\`\`
Just make sure to \`rm -r model_output\` after running.

Otherwise make sure this diagnostic info looks correct before continuing:
* Cluster:         $1
* User directory:  $USERDIR
* Flepi path:      $FLEPI_PATH
* Project path:    $PROJECT_PATH
* Python:          $WHICH_PYTHON
* R:               $WHICH_R
* Python arrow:    $PYTHON_ARROW_VERSION
* R arrow:         $R_ARROW_VERSION
* Stochastic run:  $FLEPI_STOCHASTIC_RUN
* Reset chimerics: $FLEPI_RESET_CHIMERICS
* Today:           $TODAY
* Config path:     $CONFIG_PATH
* Validation date: $VALIDATION_DATE
* Resume location: $RESUME_LOCATION
* Flepi run index: $FLEPI_RUN_INDEX
EOM
