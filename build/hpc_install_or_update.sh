#!/usr/bin/env bash

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
    module load gcc/9.3.0
    module load anaconda/2020.07
    module load git/2.42.0
else
    echo "The cluster name '$1' is not recognized, must be one of: 'longleaf', 'rockfish'."
    set +e
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

# Test that flepiMoP is located there
if [ ! -d "$FLEPI_PATH" ]; then
    while true; do
        read -p "Did not find flepiMoP at $FLEPI_PATH, do you want to clone the repo? (y/n) " resp
        case "$resp" in
            [yY])
                echo "Cloning on your behalf."
                git clone git@github.com:HopkinsIDD/flepiMoP.git $FLEPI_PATH
                break
                ;;
            [nN])
                echo "Then you need to set a \$FLEPI_PATH before running, cannot proceed with install."
                set +e
                exit 1
                ;;
            *)
                echo "Invalid input. Please enter 'y' or 'n'. "
                ;;
        esac
    done
fi

# Setup the conda environment
if [ -z "${FLEPI_CONDA}" ]; then
    echo -n "An explicit \$FLEPI_CONDA was not provided, please set one (or press enter to use 'flepimop-env'): "
    read FLEPI_CONDA
    if [ -z "${FLEPI_CONDA}" ]; then
        export FLEPI_CONDA="flepimop-env"
    fi
    echo "Using '$FLEPI_CONDA' for \$FLEPI_CONDA."
fi
FLEPI_CONDA_ENV_MATCHES=$( conda info --envs | awk '{print $1}' | grep -x "$FLEPI_CONDA" | wc -l )
if [ "$FLEPI_CONDA_ENV_MATCHES" -eq 0 ]; then
conda env create --name $FLEPI_CONDA --file $FLEPI_PATH/environment.yml
fi

# Load the conda environment
conda activate $FLEPI_CONDA
[ -e "$CONDA_PREFIX/conda-meta/pinned" ] && rm $CONDA_PREFIX/conda-meta/pinned
cat << EOF > $CONDA_PREFIX/conda-meta/pinned
r-arrow==17.0.0
arrow==17.0.0
EOF

# Install the gempyor package from local
pip install --editable --force-reinstall $FLEPI_PATH/flepimop/gempyor_pkg

# Install the local R packages
R -e "install.packages('covidcast', repos='https://cloud.r-project.org')"
RETURNTO=$( pwd )
cd $FLEPI_PATH/flepimop/R_packages/
for d in $( ls ); do
    R CMD INSTALL $d
done
cd $RETURNTO
R -e "library(inference); inference::install_cli()"

# Done
echo "> Done installing/updating flepiMoP."
set +e
