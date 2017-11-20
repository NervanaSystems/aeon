# aeon

[aeon](https://github.com/NervanaSystems/aeon) is Intel Nervana's framework-independent deep learning dataloader committed to best performance. Designed for ease-of-use and extensibility.

- Supports common media types: image, video, audio. It is ready for use with classification, segmentation, localization, transcription and more.
- Loads and augments data in parallel to framework computation.
- [Examples](examples) folder contains simple scripts showing how to use it.

For fast iteration and model exploration, it is already used by fastest performance deep learning framework [neon](https://github.com/NervanaSystems/neon).

See the new features in our latest release. 

# Getting Started

## Installation

First grab Aeon's dependencies:

### Ubuntu (release 14.04 LTS and later):

    apt-get install git clang cmake python-dev python-pip libcurl4-openssl-dev libopencv-dev libsox-dev

##### For Python 3.n

    apt-get install python3-dev python3-pip

### CentOS (release 7.2 and later):

    yum install epel-release
    yum install git clang gcc-c++ make cmake opencv-devel libcurl-devel sox-devel

##### For Python 2.7

    yum install python-pip python-devel

##### For Python 3.n

    yum install python-pip python34-pip python34-devel python34-opencv python34-numpy

### OSX:

    brew tap homebrew/science
    brew install opencv
    brew install sox

### Fedora (before release 25):

    sudo dnf install opencv-devel clang libcurl-devel sox-devel

For CentOS 7, the version of Clang available in the EPEL repository is too old
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

### Fedora (release 25 and later):

    yum install git clang cmake python-pip opencv-devel libcurl-devel sox-devel python-devel python-opencv

    mkdir -p /usr/lib/rpm/redhat/ && touch /usr/lib/rpm/redhat/redhat-hardened-ld # Because of a Fedora bug in libtool

## Code coverage

    Code coverage in aeon depends on llvm-cov and lcov.
    Report will be generated in html-coverage-report/index.html

    Example:

    sudo apt-get install llvm lcov
    mkdir build
    cd build
    # COVERAGE flag must be set to generate coverage report
    cmake .. -DCOVERAGE=ON
    # If you want to generate report when unit test fails: make -i coverage
    make coverage

### To install Aeon:

    git clone https://github.com/NervanaSystems/aeon.git

##### For Python 2.7

    cd aeon
    pip install -r requirements.txt
    mkdir -p build && cd $_ && cmake .. && pip install .

##### For Python 3.n

    cd aeon
    pip3 install -r requirements.txt
    mkdir -p build && cd $_ && cmake .. && pip3 install .

Note: if installing system wide (as opposed to within a virtual environment) you may need to run `sudo`

Now continue on to the [user guide](http://aeon.nervanasys.com/index.html/user_guide.html) to get started using aeon. Or to the
[developer guide](http://aeon.nervanasys.com/index.html/developer_guide.html) to developing custom loaders/transformers.

# Documentation

The complete documentation for aeon is available [here](aeon.nervanasys.com).

# Support

For any bugs or feature requests please:

Search the open and closed [issues list](https://github.com/NervanaSystems/aeon/issues) to see if we're already working on what you have uncovered.
Check that your issue/request isn't framework related.
File a new [issue](https://github.com/NervanaSystems/aeon/issues) or submit a new [pull request](https://github.com/NervanaSystems/aeon/pulls) if you have some code you'd like to contribute

# License

We are releasing [aeon](https://github.com/NervanaSystems/aeon) under an open source [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome you to [contact us](info@nervanasys.com) with your use cases.
