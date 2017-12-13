.. ---------------------------------------------------------------------------
.. Copyright 2017 Intel(R) Nervana(TM)
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Getting Started
===============

Installation
------------

First grab aeon's dependencies:

Ubuntu::

  apt-get install git clang libcurl4-openssl-dev libopencv-dev libsox-dev libboost-filesystem-dev libboost-system-dev

Centos::

  yum install epel-release
  yum install git clang gcc-c++ make cmake openssl-devel opencv-devel libcurl-devel sox-devel boost-devel boost-filesystem boost-system

OSX (Assuming you followed neon's Homebrew based install)::

  brew tap homebrew/science
  brew install opencv
  brew install sox
  brew install boost


Then build the aeon libraries::

    git clone https://github.com/NervanaSystems/aeon.git
    mkdir -p aeon/build && cd $_ && cmake .. && pip install .

Now continue on to the :doc:`user_guide` to get started using aeon. Or to the
:doc:`developer_guide` to developing custom loaders/transformers.

For distributed aeon please check `dependencies <service.html#dependencies>`_ and `building <service.html#building>`_ information.
