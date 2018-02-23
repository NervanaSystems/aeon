#!/bin/bash
# This will install the prerequisites necessary to build the documentation
if [ $(id -u) != "0" ]; then
    echo >&2 "error: this installation script must be run as root"
    exit 1
fi
[ -f /etc/os-release ] && . /etc/os-release
if [ -z ${ID+x} ]; then
    echo >&2 "error: unknown operating system detected (missing /etc/os-release)"
    exit 1
fi
case $ID in
ubuntu)
    apt install -y python-pip doxygen
    ;;
centos)
    yum -y install python-pip doxygen
    ;;
*)
    echo >&2 "error: unsupported Linux distribution detected (ID=\"$(ID)\")"
    exit 1
    ;;
esac

# We have to use the master version of breathe to avoid
# https://github.com/michaeljones/breathe/issues/261
# I've pinned the commit to what worked for me

pip install --upgrade Sphinx breathe sphinxcontrib-httpdomain sphinxcontrib-napoleon
