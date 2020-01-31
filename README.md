# aeon

[aeon](https://github.com/NervanaSystems/aeon) is Intel Nervana's framework-independent deep learning dataloader committed to best performance. Designed for ease-of-use and extensibility.

- Supports common media types: image, video. It is ready for use with classification, segmentation, localization, transcription and more.
- Loads and augments data in parallel to framework computation.
- [Examples](examples) folder contains simple scripts showing how to use it.

For fast iteration and model exploration, it is already used by fastest performance deep learning framework [neon](https://github.com/NervanaSystems/neon).

See the new features in our latest release.

# Getting Started

## Installation

First grab aeon's dependencies:

### Ubuntu (release 16.04 LTS and later)

    apt-get install git clang libopencv-dev

##### For Python 3.n

    apt-get install python3-dev python3-pip python3-numpy

### CentOS (release 7.2 and later)

    yum install epel-release
    yum install git clang gcc-c++ make cmake opencv-devel

##### For Python 2.7

    yum install python-pip python-devel

##### For Python 3.n

    yum install python-pip python34-pip python34-devel python34-opencv python34-numpy

### OSX

    brew tap homebrew/science
    brew install opencv

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

### To install aeon

    git clone -b develop https://github.com/NervanaSystems/aeon.git

##### For Python 2.7

    cd aeon
    pip install -r requirements.txt
    mkdir -p build && cd $_ && cmake .. && pip install .

##### For Python 3.n

    cd aeon
    pip3 install -r requirements.txt
    mkdir -p build && cd $_ && cmake .. && pip3 install .

### To install aeon with the latest OpenCV version

##### Clone aeon repository

    git clone -b develop https://github.com/NervanaSystems/aeon.git

##### Build OpenCV

    mkdir -p aeon/build/3rdparty && cd $_
    git clone https://github.com/opencv/opencv.git
    cd opencv && mkdir build && cd $_

    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/installation/OpenCV -DINSTALL_C_EXAMPLES=OFF -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_TBB=OFF -DWITH_V4L=OFF -DWITH_QT=OFF -DWITH_OPENGL=OFF -DBUILD_EXAMPLES=OFF -DWITH_CUDA=OFF -DOPENCV_FORCE_3RDPARTY_BUILD=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DBUILD_ZLIB=OFF -DBUILD_TIFF=OFF  -DBUILD_JASPER=OFF -DBUILD_JPEG=OFF -DBUILD_PNG=OFF -DBUILD_OPENEXR=OFF -DBUILD_WEBP=OFF -DBUILD_opencv_gpu=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DOPENCV_PC_FILE_NAME=opencv.pc ..

    make -j8 # for example it runs 8 jobs in parallel
    make install

##### Install aeon

    cd ../../../
    cmake -DCMAKE_PREFIX_PATH=$(pwd)/3rdparty/opencv/build/installation/OpenCV ..

##### For Python 2.7

    pip install -r ../requirements.txt
    pip install .

##### For Python 3.n

    pip3 install -r ../requirements.txt
    pip3 install .

You can also build wheel package with command `python setup.py bdist_wheel`

Note: if installing system wide (as opposed to within a virtual environment) you may need to run `sudo`

Now continue on to the [user guide](http://aeon.nervanasys.com/index.html/user_guide.html) to get started using aeon. Or to the
[developer guide](http://aeon.nervanasys.com/index.html/developer_guide.html) to developing custom loaders/transformers.

# Documentation

The complete documentation for aeon is available [here](http://aeon.nervanasys.com).

# Support

For any bugs or feature requests please:

Search the open and closed [issues list](https://github.com/NervanaSystems/aeon/issues) to see if we're already working on what you have uncovered.
Check that your issue/request isn't framework related.
File a new [issue](https://github.com/NervanaSystems/aeon/issues) or submit a new [pull request](https://github.com/NervanaSystems/aeon/pulls) if you have some code you'd like to contribute

# License

We are releasing [aeon](https://github.com/NervanaSystems/aeon) under an open source [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License. We welcome you to [contact us](info@nervanasys.com) with your use cases.
