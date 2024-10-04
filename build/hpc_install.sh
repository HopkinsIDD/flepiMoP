# Generic setup
set -e

# Cluster specific setup
if [[ $1 == "longleaf" ]]; then
    # Setup general purpose user variables needed for Longleaf
    USERO=$( echo $USER | awk '{ print substr($0, 1, 1) }' )
    USERN=$( echo $USER | awk '{ print substr($0, 2, 1) }' )
    USERDIR="/users/$USERO/$USERN/$USER/"
    WORKDIR="/work/users/$USERO/$USERN/$USER/"

    # Load required modules
    module purge
    module load gcc/9.1.0
    module load anaconda/2023.03
    module load git
elif [[ $1 == "rockfish" ]]; then
    # Setup general purspose user variables needed for RockFish
    USERDIR=$HOME
    WORKDIR="/scratch4/struelo1/flepimop-code/$USER/"
    mkdir -vp $WORKDIR

    # Load required modules
    module purge
    module load gcc/9.3.0
    module load anaconda/2020.07
    module load git/2.42.0
else
    echo "The cluster name '$1' is not recognized, must be one of: 'longleaf', 'rockfish'."
    exit 1
fi

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
cat << EOF > $USERDIR/flepimop-env/conda-meta/pinned
r-arrow==17.0.0
arrow==17.0.0
EOF
fi
conda activate $USERDIR/flepimop-env

# Install the gempyor package from local
pip install --force-reinstall $FLEPI_PATH/flepimop/gempyor_pkg

# Install the local R packages
$FLEPI_PATH/build/setup.R $FLEPI_PATH

# Out source to flepi init for per run setup
source $FLEPI_PATH/build/flepi_init.sh $1
