#!/bin/bash
# This will install the prerequisites necessary to build the documentation
sudo apt-get install doxygen

# We have to use the master version of breathe to avoid
# https://github.com/michaeljones/breathe/issues/261
# I've pinned the commit to what worked for me
sudo pip install sphinx git+https://github.com/michaeljones/breathe.git@d681785bb728eb02261263f26736bf6b2a929da5
