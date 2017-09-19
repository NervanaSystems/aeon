.. ---------------------------------------------------------------------------
.. Copyright 2017 Nervana Systems Inc.
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

Distributed data loading
==========

Server
-----------
Server is a component managing distributed data loading sessions. It does all heavy computing of data extraction, transformation and augmentation. Clients talk with server via REST-like API. Client may either create new sessions or connect to existing ones.

Dependencies
^^^^^^
`cpprestsdk <https://github.com/Microsoft/cpprestsdk>`_ library is used to implement REST service.  It's available on ubuntu 16.04, but not on ubuntu 14.04 or centos. If that's your problem, you need to build this library from sources. `This <https://github.com/Microsoft/cpprestsdk/wiki/How-to-build-for-Linux>`_ might be helpful to achieve that.

Parameters
^^^^^^
``aeon-server`` is a separate binary. Below you can find it's parameters.

.. csv-table::
   :header: "Parameter", "Short version", "Type", "Required", "Description"
   :widths: 20, 10, 10, 10, 50
   :delim: |
   :escape: ~

   --address | a | string | true | Server address to listen on.
   --port | p | true | uint | Server port to listen on.
   --daemon | d | false | flag | Runs ``aeon-server`` as daemon.

Client
-----------
Connection with server is configurable with ``server`` object from aeon config. Below you can find it's fields. If ``server`` object is absent, usual local data loading will happen.
Parameters of server object from main aeon config:

.. csv-table::
   :header: "Parameter", "Type", "Default", "Description"
   :widths: 20, 10, 10, 50
   :delim: |
   :escape: ~

   address | string | *Required* | Server address to connect to.
   port | uint | *Required* | Server port to connect on.
   session_id | string | ~"~" | ID of shared session to connect to. If it's not provided, new session will be created.
