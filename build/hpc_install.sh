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
    module load r/4.4.0
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
    module load r/4.3.0
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
R_DEPENDS_SCRIPT=$( mktemp )
cat << EOF > $R_DEPENDS_SCRIPT
# Helper
split_pkgs <- \\(x) unique(unlist(strsplit(gsub("\\\\s+", "", x), ",")))
# Command line args
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: r_dependencies.R <flepi-path>")
}
flepi_path <- args[1L]
rpkgs <- list.files(file.path(flepi_path, "flepimop", "R_packages"), full.names = TRUE)
# Loop through and parse the dependencies
dependencies <- sapply(rpkgs, function(rpkg) {
    description <- read.dcf(file.path(rpkg, "DESCRIPTION"))
    sections <- c("Depends", "Imports")
    contained_sections <- sections %in% colnames(description)
    if (sum(contained_sections) >= 1L) {
        return(split_pkgs(description[, sections[contained_sections]]))
    }
    character()
}, USE.NAMES = FALSE)
dependencies <- sort(unique(unlist(dependencies)))
# Exclude arrow, methods, covidcast, self packages, and R
dependencies <- setdiff(dependencies, c("arrow", "covidcast", "methods", basename(rpkgs)))
dependencies <- dependencies[!grepl("^R(\\\\(.*\\\\))?$", dependencies)]
# Print the dependencies
cat(paste(dependencies, collapse = ","))
EOF
R_DEPENDS=$( Rscript $R_DEPENDS_SCRIPT $FLEPI_PATH )
rm $R_DEPENDS_SCRIPT
R_DEPENDS=$( echo ",$R_DEPENDS" | sed 's/,/\n- r-/g' | sed '/^[[:space:]]*$/d' )
cat << EOF > $USERDIR/environment.yml
channels:
- conda-forge
- defaults
dependencies:
- python=3.10
- pip
- r-base>=4.3
- r-essentials
- pyarrow=17.0.0
- r-arrow=17.0.0
# Manually specify this one because of the paths for libudunits2 on longleaf
- r-sf 
# Extracted dependencies
$R_DEPENDS
EOF
conda env create --prefix $USERDIR/flepimop-env --file $USERDIR/environment.yml
cat << EOF > $USERDIR/flepimop-env/conda-meta/pinned
r-arrow==17.0.0
arrow==17.0.0
EOF
fi

# Load the conda environment
module unload r
conda activate $USERDIR/flepimop-env

# Install the gempyor package from local
pip install --force-reinstall $FLEPI_PATH/flepimop/gempyor_pkg

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
