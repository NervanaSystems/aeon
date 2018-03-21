.. ---------------------------------------------------------------------------
.. Copyright 2017-2018 Intel Corporation
.. 
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Getting Started
===============

Aeon does not support other than official Python distributions and may not work.

Installation
------------

First grab aeon's dependencies:

**Ubuntu (release 16.04 LTS and later)**::

  apt-get install git clang libcurl4-openssl-dev libopencv-dev libsox-dev libboost-filesystem-dev libboost-system-dev libssl-dev

Python 3.n::

    apt-get install python3-dev python3-pip python3-numpy

**Centos (release 7.2 and later)**::

  yum install epel-release
  yum install git clang gcc-c++ make cmake openssl-devel opencv-devel libcurl-devel sox-devel boost-devel boost-filesystem boost-system

Python 2.7::

    yum install python-pip python-devel

Python 3.n::

    yum install python-pip python34-pip python34-devel python34-opencv python34-numpy


**OSX (Assuming you followed neon's Homebrew based install)**::

  brew tap homebrew/science
  brew install opencv
  brew install sox
  brew install boost


**Then build the aeon libraries**::

    git clone https://github.com/NervanaSystems/aeon.git

For Python 2.7::

    cd aeon
    pip install -r requirements.txt
    mkdir -p build && cd $_ && cmake .. && pip install .

For Python 3.n::

    cd aeon
    pip3 install -r requirements.txt
    mkdir -p build && cd $_ && cmake .. && pip3 install .

Note: if installing system wide (as opposed to within a virtual environment) you may need to run `sudo`


Now continue on to the :doc:`user_guide` to get started using aeon. Or to the
:doc:`developer_guide` to developing custom loaders/transformers.

For distributed aeon please check `dependencies <service.html#dependencies>`_ and `building <service.html#building>`_ information.
