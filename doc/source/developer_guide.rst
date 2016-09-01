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

Developer's Guide
=================

Setup
-----

You'll probably want to install gtest_ for running and implementing your C++ 
unit tests.

In Ubuntu::

  sudo apt-get install libgtest-dev
  cd /usr/lib/gtest
  sudo cmake .  sudo make
  sudo mv libg* /usr/local/lib/

Mac OS X::

  brew install cmake
  git clone https://github.com/google/googletest.git
  cd googletest
  cmake .
  make -j
  make install
