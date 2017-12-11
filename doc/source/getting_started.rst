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

For CentOS 7, it might be that the version of Clang available in the EPEL repository is too old
to understand the GCC flags that the Python extension build system imposes, so
we build the latest version of Clang from source (after first installing it's
high and low level build systems)::

    sudo -i
    # Build CMake
    cd $BUILDROOT
    wget https://cmake.org/files/v3.6/cmake-3.6.1.tar.gz
    tar xf cmake-3.6.1.tar.gz
    rm cmake-3.6.1.tar.gz
    cd cmake-3.6.1
    ./bootstrap && make -j && make install

    # Build Ninja
    cd $BUILDROOT
    git clone https://github.com/ninja-build/ninja.git && cd ninja
    ./configure.py --bootstrap && cp ninja /usr/local/bin

    # Build LLVM + Clang
    cd $BUILDROOT && rm -rf /ninja
    wget http://llvm.org/releases/3.9.0/llvm-3.9.0.src.tar.xz
    tar xf llvm-3.9.0.src.tar.xz && rm llvm-3.9.0.src.tar.xz
    cd llvm-3.9.0.src/tools
    wget http://llvm.org/releases/3.9.0/cfe-3.9.0.src.tar.xz
    tar xf cfe-3.9.0.src.tar.xz && rm cfe-3.9.0.src.tar.xz
    mkdir $BUILDROOT/llvmbuild
    cd $BUILDROOT/llvmbuild
    cmake -G "Ninja" -DCMAKE_BUILD_TYPE=release ../llvm-3.9.0.src
    ninja && ninja install


Then build the aeon libraries::

    git clone https://github.com/NervanaSystems/aeon.git
    mkdir -p aeon/build && cd $_ && cmake .. && pip install .

Now continue on to the :doc:`user_guide` to get started using aeon. Or to the
:doc:`developer_guide` to developing custom loaders/transformers.

For distributed aeon please check `dependencies <service.html#dependencies>`_ and `building <service.html#building>`_ information.
