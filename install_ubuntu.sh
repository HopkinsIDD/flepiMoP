#! /bin/bash

script_path=$(dirname $(realpath $0))

# ensure FLEPI_PATH available as environment variable

# if FLEPI_PATH is not equal to script path, set or update it
if [ "$FLEPI_PATH" != $script_path ]; then
    if [ -z "$FLEPI_PATH" ]; then
        echo "FLEPI_PATH not set; updating ~/.bashrc ..."
        export FLEPI_PATH=$script_path
        sudo echo "export FLEPI_PATH=$FLEPI_PATH" >> ~/.bashrc
        sudo echo "export ''" >> ~/.bashrc
        echo "appended to ./bashrc to set FLEPI_PATH=$FLEPI_PATH"
    else
        echo "current FLEPI_PATH=$FLEPI_PATH - updating ..."
        export FLEPI_PATH=$script_path
        sudo sed 's,FLEPI_PATH=[^;]*,"FLEPI_PATH=$script_path",' -i ~/.bashrc
        echo "updated ./bashrc to set FLEPI_PATH=$FLEPI_PATH"
    fi
else
    echo "FLEPI_PATH=$FLEPI_PATH environment variable already set."
    # TODO: ensure that FLEPI_PATH is set in bashrc? bash_profile?
fi

# install necessary system dependencies
sudo apt install libudunits2-dev libssl-dev libfontconfig1-dev libxml2-dev libcurl4-openssl-dev libharfbuzz-dev libfribidi-dev libgdal-dev libcairo2-dev

# install the python package
pip install -e $FLEPI_PATH/flepimop/gempyor_pkg/

# install the R packages
sudo $FLEPI_PATH/build/setup.R $FLEPI_PATH