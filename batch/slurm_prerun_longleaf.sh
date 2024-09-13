USERO=$( echo $USER | awk '{ print substr($0, 1, 1) }' )
USERN=$( echo $USER | awk '{ print substr($0, 2, 1) }' )
USERDIR="/users/$USERO/$USERN/$USER"


source $USERDIR/slack_credentials.sh
export FLEPI_STOCHASTIC_RUN=false
export FLEPI_RESET_CHIMERICS=TRUE
export FLEPI_PATH=$USERDIR/flepiMoP

echo "Doing some inference_run setup, hit enter to skip a question if not relevant or you're not planning on doing a run"

export TODAY=`date --rfc-3339='date'`
echo "Let's set up a flepiMoP run. Today is the $TODAY"

echo "(1/3) Please input the validation date:"
read input
export VALIDATION_DATE="$input"
echo -e ">>> set VALIDATION DATE to $VALIDATION_DATE \n"

echo "(2/3) Please input the resume location (empty if no resume, or a s3:// url or a local folder):"
read input
export RESUME_LOCATION="$input"
echo -e ">>> set RESUME_LOCATION to $RESUME_LOCATION \n"

echo "(3/3) Please provide the Run Index for the current run:"
read input
export FLEPI_RUN_INDEX="$input"
echo -e ">>> set  FLEPI_RUN_INDEX to $FLEPI_RUN_INDEX \n"

echo "DONE. if no error please manually set export CONFIG_PATH=YOURCONFIGPATH.yml"
echo "(in case of error, override manually some variables or rerun this script)"
