.. ---------------------------------------------------------------------------
.. Copyright 2015 Nervana Systems Inc.
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

  sudo apt-get install libcurl4-openssl-dev clang libopencv-dev libsox-dev

OSX (Assuming you followed neon_'s Homebrew based install)::

  brew tap homebrew/science
  brew install opencv
  brew install sox

Note: Fedora and CentOS 7 are unsupported platforms for aeon, but we've found the following
instructions to be useful.

Fedora::

  sudo dnf install opencv-devel clang libcurl-devel sox-devel

For CentOS 7, the version of Clang available in the EPEL repository is too old
to understand the GCC flags that the Python extension build system imposes, so
we build the latest version of Clang from source (after first installing it's
high and low level build systems)::

    yum -y install libcurl-devel gcc gcc-c++ make pkgconfig opencv-devel git \
        python-virtualenv libyaml-devel epel-release hdf5-devel python-pip \
        sox-devel
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
  mkdir build_aeon && cd $_ && make ../aeon && make -j

To install the python bindings, *after* the library has been built, run the following from the ``build_aeon`` directory::
  pip install .


Now continue on to the :doc:`user_guide` to get started using aeon. Or to the
:doc:`developer_guide` to developing custom loaders/transformers.

