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

Distributed Data Loading
=========================

Aeon Service
-------------
``aeon-service`` is a component managing distributed data loading sessions. It does all heavy computing of data extraction, transformation and augmentation. Clients talk with service via REST-like API. Client may either create new sessions or connect to existing ones.

.. _dependencies:

Dependencies
^^^^^^^^^^^^^
`cpprestsdk <https://github.com/Microsoft/cpprestsdk>`_ library is used to implement REST service.  It's available on Ubuntu 16.04, but not on Ubuntu 14.04 or Centos. If that's your problem, you need to build this library from sources. `This <https://github.com/Microsoft/cpprestsdk/wiki/How-to-build-for-Linux>`_ might be helpful to achieve that.  Version 2.4 is proved to be working on Centos 7.4.

On Ubuntu 16.04 and higher you need just:

.. code-block:: bash

    sudo apt install libcpprest-dev

When using RDMA transfer, you need to install `libfabric <https://github.com/ofiwg/libfabric>`_.
Installation for Centos:

.. code-block:: bash

    sudo yum install libfabric-devel

If it's not available in your OS, you need to build it from  `sources <https://github.com/ofiwg/libfabric>`_:

.. code-block:: bash

    git clone https://github.com/ofiwg/libfabric
    cd libfabric
    ./configure --prefix=<PATH_TO_LOCAL_DIR>
    make -j
    sudo make install

.. _building:

Building
^^^^^^^^^^^
The building of AEON client is enabled by default. To disable building AEON client please provide cmake flag ``-DENABLE_AEON_CLIENT=OFF``.

To enable building the AEON service (by default disabled) please provide cmake flag ``-DENABLE_AEON_SERVICE=ON``.
By providing the flag the cmake will start resolving dependencies required for AEON service tragets.
The "must have" component is C++ REST SDK from Microsoft (both runtime and development files).
The packages are available in all recent Linux operating systems, for other supported systems the ``cpprest`` package might not exist.
In such a case please build the library from source files. If you plan to install the library in non-standard directory, then let know
to the cmake its location by providing option ``-DCPPREST_DIR`` (root directory of runtime and development files).

To enable RDMA support (by default disabled) please provide the cmake flag ``-DENABLE_OPENFABRICS_CONNECTOR=ON``. The flag is recognized
by AEON service and AEON client targets. The "must have" component in this case is OpenFrabics Interface library (runtime and development
files). The packages are available for all recent Linux operating systems, and for other supported systems please build the library from
source files. If you plan to install the library in non-standard directory, then let know to the cmake its location by providing the
option ``-DOPENFABRICS_DIR`` (root directory of runtime and development files).

Sessions
^^^^^^^^^^^
Currently only data parallelism is supported. That means that each client will get different batch in every iteration step. If there are ``B`` batches in whole dataset, ``C`` clients and they are synchronized to process batch after batch, then each client will get only ``B/C`` batches.
To initialize dataloading session, you need to create one. It can be done in two ways. The first is to send ``POST`` request to ``/api/v1/dataset`` with config in the body. In response you will get session id. The second one is to create aeon object as usual, but providing ``remote`` object  without ``session_id`` field. In this case aeon will create new session, which can be passed to other clients. Session id can be fetched from aeon object field ``session_id``.
Session is destroyed along with dataloader object assuming that ``close_session`` is not set to ``false``.

You can use separate dataloading sessions just by not using ``session_id`` field in configuration.

RDMA
^^^^^^^^^^^^
To improve batch fetching speed, you may consider using RDMA transfer. For this, you need to install `libfabric <https://github.com/ofiwg/libfabric>`_ library.
On service side you need to provide ``address`` and ``port`` parameters to enable RDMA transfer.
On client side, you have to add parameters ``rdma_address`` and ``rdma_port`` in ``remote`` object to make use of RDMA transfer.
Currently RDMA works only on adapters supporting IB verbs, including Intel® Omni-Path Architecture adapters.

Parameters
^^^^^^^^^^^
``aeon-service`` is a separate executable binary. Below you can find it's parameters.

.. csv-table::
   :header: "Parameter", "Short version", "Type", "Default", "Description"
   :widths: 20, 10, 10, 10, 50
   :delim: |
   :escape: ~

   uri | u | string | *Required* | URI to listening interface in format 'protocol://host:port/'.
   address | a | string | ~"~" | IP address of RDMA interface.
   port | p | uint | 0 | Port number of RDMA interface.
   daemon | d | flag | \- | Run the process in background as a daemon process. By default the process runs in foreground.
   log | l | string | /var/log/aeon-service.log | Path to log file.
   version | v | flag | \- | Prints version.
   help | h | flag | \- | Prints help.

RESTful API can be found :doc:`here <service_api>`.

Client
-----------
Connection with service is configurable with ``remote`` object in aeon config. Below you can find it's fields. If ``remote`` object is absent, regular local data loading will happen.
Parameters of ``remote`` object from main aeon config:

.. csv-table::
   :header: "Parameter", "Type", "Default", "Description"
   :widths: 20, 10, 10, 50
   :delim: |
   :escape: ~

   address | string | *Required* | Service address to connect to.
   port | uint | *Required* | Service port to connect to.
   session_id | string | ~"~" | ID of shared session to connect to. If it's not provided, new session will be created.
   close_session | bool | true | If set to true, aeon will close session when aeon object is being destroyed.
   async | bool | true | async set to true makes batch loading to be double-buffered. Please note that async mode can make client fetch one batch more than requested.
   rdma_address | string | ~"~" | IP address of RDMA interface.
   rdma_port | uint | 0 | Port number of RDMA interface.

Usage
^^^^^^^^^^^^^
Single session usage has been presented in `cpp_iterator <https://github.com/NervanaSystems/aeon/tree/master/examples/cpp_remote_iterator>`_ and `python_remote_iterator <https://github.com/NervanaSystems/aeon/tree/master/examples/python_remote_iterator>`_.
Shared session usage can be found in `python_remote_iterator_shared <https://github.com/NervanaSystems/aeon/tree/master/examples/python_remote_iterator_shared>`_.
