# Getting Started

## Installation

First grab aeon's dependencies:

### Ubuntu:

    sudo apt-get install libcurl4-openssl-dev clang libopencv-dev libsox-dev

### OSX:

    brew tap homebrew/science
    brew install opencv
    brew install sox

### Fedora:

    sudo dnf install opencv-devel clang libcurl-devel sox-devel

For CentOS 7, the version of Clang available in the EPEL repository is too old 
to understand the GCC flags that the Python extension build system imposes, so 
we build the latest version of Clang from source (after first installing it's 
high and low level build systems)::

    sudo -i
    # Build CMake
    cd $BUILDROOT
    curl -O https://cmake.org/files/v3.6/cmake-3.6.1.tar.gz
    tar xf cmake-3.6.1.tar.gz
    rm cmake-3.6.1.tar.gz
    cd cmake-3.6.1
    ./bootstrap && make -j && make install

    # Build Ninja
    cd $BUILDROOT
    git clone git://github.com/ninja-build/ninja.git && cd ninja
    ./configure.py --bootstrap && cp ninja /usr/local/bin

    # Build LLVM + Clang
    cd $BUILDROOT && rm -rf /ninja
    curl -O http://llvm.org/releases/3.9.0/llvm-3.9.0.src.tar.xz
    tar xf llvm-3.9.0.src.tar.xz && rm llvm-3.9.0.src.tar.xz
    cd llvm-3.9.0.src/tools
    curl -O http://llvm.org/releases/3.9.0/cfe-3.9.0.src.tar.xz
    tar xf cfe-3.9.0.src.tar.xz && rm cfe-3.9.0.src.tar.xz
    mkdir $BUILDROOT/llvmbuild
    cd $BUILDROOT/llvmbuild
    cmake -G "Ninja" -DCMAKE_BUILD_TYPE=release /llvm-3.9.0.src
    ninja && ninja install

### To install aeon:

    pip install git+https://github.com/NervanaSystems/aeon.git

Note: if installing system wide (as opposed to within a virtual environment) you may need to run `sudo`.

Now continue on to the [user guide](doc/source/user_guide.rst) to get started using aeon. Or to the
[developer guide](doc/source/developer_guide.rst) to developing custom loaders/transformers.
