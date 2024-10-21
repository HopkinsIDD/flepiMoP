# Generic setup
set -e

# Cluster specific setup
if [[ $1 == "longleaf" ]]; then
    # Setup general purpose user variables needed for Longleaf
    USERO=$( echo $USER | awk '{ print substr($0, 1, 1) }' )
    USERN=$( echo $USER | awk '{ print substr($0, 2, 1) }' )
    WORKDIR=$( realpath "/work/users/$USERO/$USERN/$USER/" )
    USERDIR=$WORKDIR

    # Load required modules
    module purge
    module load gcc/9.1.0
    module load anaconda/2023.03
    module load git
elif [[ $1 == "rockfish" ]]; then
    # Setup general purspose user variables needed for RockFish
    WORKDIR=$( realpath "/scratch4/struelo1/flepimop-code/$USER/"  )
    USERDIR=$WORKDIR
    mkdir -vp $WORKDIR

    # Load required modules
    module purge
    module load slurm
    module load gcc/9.3.0
    module load anaconda/2020.07
    module load git/2.42.0
else
    echo "The cluster name '$1' is not recognized, must be one of: 'longleaf', 'rockfish'."
    exit 1
fi

# Ensure we have a $FLEPI_PATH
if [ -z "${FLEPI_PATH}" ]; then
    echo -n "An explicit \$FLEPI_PATH was not provided, please set one (or press enter to use '$USERDIR/flepiMoP'): "
    read FLEPI_PATH
    if [ -z "${FLEPI_PATH}" ]; then
        export FLEPI_PATH="$USERDIR/flepiMoP"
    fi
    export FLEPI_PATH=$( realpath "$FLEPI_PATH" )
    echo "Using '$FLEPI_PATH' for \$FLEPI_PATH."
fi

# Conda init
if [ -z "${FLEPI_CONDA}" ]; then
    echo -n "An explicit \$FLEPI_CONDA was not provided, please set one (or press enter to use '$USERDIR/flepimop-env'): "
    read FLEPI_CONDA
    if [ -z "${FLEPI_CONDA}" ]; then
        export FLEPI_CONDA="$USERDIR/flepimop-env"
    fi
    export FLEPI_CONDA=$( realpath "$FLEPI_CONDA" )
    echo "Using '$FLEPI_CONDA' for \$FLEPI_CONDA."
fi
conda activate $FLEPI_CONDA

# Check the conda environment is valid
WHICH_PYTHON=$( which python )
WHICH_R=$( which R )
PYTHON_ARROW_VERSION=$( python -c "import pyarrow; print(pyarrow.__version__)" )
R_ARROW_VERSION=$( Rscript -e "cat(as.character(packageVersion('arrow')))" )
COMPATIBLE_ARROW_VERSION=$( echo "$R_ARROW_VERSION" | grep "$PYTHON_ARROW_VERSION" | wc -l )
if [[ "$COMPATIBLE_ARROW_VERSION" -ne 1 ]]; then
    echo "The R version of arrow is '$R_ARROW_VERSION' and the python version is '$PYTHON_ARROW_VERSION'. These may not be compatible versions."
fi

# Make sure the credentials is is where we expect and have the right perms
if [ ! -f "$USERDIR/slack_credentials.sh" ]; then
    echo "You should place sensitive credentials in '$USERDIR/slack_credentials.sh'."
else
    chmod 600 $USERDIR/slack_credentials.sh
    source $USERDIR/slack_credentials.sh
fi

# Set correct env vars
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
export TODAY=`date --rfc-3339='date'`

echo -n "Please set a project path (relative to '$WORKDIR'): "
read PROJECT_PATH
export PROJECT_PATH="$WORKDIR/$PROJECT_PATH"
if [ ! -d $PROJECT_PATH ]; then
    echo "> The project path provided, $PROJECT_PATH, is not a directory. Please ensure this is correct."
fi

echo -n "Please set a config path (relative to '$PROJECT_PATH'): "
read CONFIG_PATH
export CONFIG_PATH="$PROJECT_PATH/$CONFIG_PATH"
if [ ! -f $CONFIG_PATH ]; then
    echo "> The config path provided, $CONFIG_PATH, is not a file. Please ensure this is correct."
fi

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
flepimop-inference-main -c \$CONFIG_PATH -j 1 -n 1 -k 1
\`\`\`
Just make sure to \`rm -r model_output\` after running.

Otherwise make sure this diagnostic info looks correct before continuing:
* Cluster:         $1
* User directory:  $USERDIR
* Work directory:  $WORKDIR
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

set +e
