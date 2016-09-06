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

First, if you haven't already, install neon_. Then grab aeon's dependencies:

Ubuntu::

  sudo apt-get install -y libcurl4-openssl-dev clang libopencv-dev

OSX::

  brew tap homebrew/science
  brew install opencv

CentOS/Fedora::

  sudo dnf install opencv-devel clang

Then install aeon under your neon virtualenv::

  git clone https://github.com/NervanaSystems/aeon.git
  cd aeon
  python setup.py install

If your neon is installed system wide, you can instead install aeon system wide
as well using :code:`sudo python setup.py install`.

Now continue on to the :doc:`user_guide` to get started using aeon. Or to the
:doc:`developer_guide` to developing custom loaders/transformers.

.. _neon: https://github.com/NervanaSystems/neon
