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
    apt install -y doxygen python-sphinxcontrib-httpdomain python-pip
    ;;
centos)
    yum -y install doxygen python2-sphinxcontrib-httpdomain python-pip
    ;;
*)
    echo >&2 "error: unsupported Linux distribution detected (ID=\"$(ID)\")"
    exit 1
    ;;
esac

# We have to use the master version of breathe to avoid
# https://github.com/michaeljones/breathe/issues/261
# I've pinned the commit to what worked for me

pip install Sphinx git+https://github.com/michaeljones/breathe.git@d681785bb728eb02261263f26736bf6b2a929da5 sphinxcontrib-httpdomain
