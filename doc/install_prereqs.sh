#!/bin/bash
# This will install the prerequisites necessary to build the documentation
sudo apt-get install doxygen

# We have to use the master version of breathe to avoid
# https://github.com/michaeljones/breathe/issues/261
# I've pinned the commit to what worked for me

if [[ "$VIRTUAL_ENV" != "" ]]
then
    pip install Sphinx git+https://github.com/michaeljones/breathe.git@d681785bb728eb02261263f26736bf6b2a929da5
else
    sudo pip install Sphinx git+https://github.com/michaeljones/breathe.git@d681785bb728eb02261263f26736bf6b2a929da5
fi
