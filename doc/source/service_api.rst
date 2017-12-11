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
.. neon documentation master file

Service RESTful API
====================

   **URI structure**:

   ``http://(address):(port)/v1/dataset/(session_id)/(resource name)``

   Where:

   - **host**: *aeon-service* address
   - **port**: *aeon-service* port
   - **v1**: means it's the first version of API
   - **dataset**: namespace for all resources related to dataset
   - **session_id**: dataset session id
   - **resource name**: name of resource to be returned


.. _service_response:

   **Response format**:

   All endpoints with except of ``next`` return json named here ``service_response``.
   It consists of three fields:

   - **status** (enumeration):
       - ``"SUCCESS"``:  Request has been successfuly processed
       - ``"FAILURE"``: Some error occured. ``description`` field should contain some information.
       - ``"END_OF_DATASET"``: End of dataset has been reached and no new batch can be provided. User should reset session to start getting new batches.
   - **description** (optional): Provides description helpful to pass additional information about errors.
   - **data** (optional): Contains data requested by the user.

.. http:post:: /api/v1/dataset

   Creates new dataset session.
   Takes manifest as json parameter and creates new dataloading session. In response it sends session id in ``data`` json field.

   **Example request**:

   .. sourcecode:: bash

       curl -XPOST "http://example.com:34568/api/v1/dataset" -H "Content-Type: application/json" -d '{"manifest_filename":"~/test_data/manifest.tsv", "manifest_root": "~/test_data/", "batch_size": 8, "etl": [{"type": "image", "width": 28, "height": 28}, {"type": "label"}]}'


   **Example response**:

   .. sourcecode:: json

        {
           "data": {
              "id": "622"
           },
           "status": {
              "type": "SUCCESS"
           }
        }

   :statuscode 202: session was created
   :statuscode 400: bad config
   :statuscode 500: internal error


.. http:get:: /api/v1/dataset/(int:session_id)/names_and_shapes

   Provides data names and its shapes for session ``session_id``.

   **Example request**:

   .. sourcecode:: bash

        curl "http://example.com:34568/api/v1/dataset/622/names_and_shapes"

   **Example response**:

   .. sourcecode:: json

        {
           "data": {
              "names_and_shapes": {
                 "image": {
                    "byte_size": 2352,
                    "names": [
                       "channels",
                       "height",
                       "width"
                    ],
                    "otype": {
                       "cv_type": 0,
                       "name": "uint8_t",
                       "np_type": 2,
                       "size": 1
                    },
                    "shape": [
                       3,
                       28,
                       28
                    ]
                 },
                 "label": {
                    "byte_size": 4,
                    "names": [],
                    "otype": {
                       "cv_type": 4,
                       "name": "uint32_t",
                       "np_type": 6,
                       "size": 4
                    },
                    "shape": [
                       1
                    ]
                 }
              }
           },
           "status": {
              "type": "SUCCESS"
           }
        }


   :query session_id: session id
   :statuscode 200: no error
   :statuscode 404: there's no such session id
   :statuscode 500: internal error


.. http:get:: /api/v1/dataset/(int:session_id)/batch_size

   Provides batch size for session ``session_id``.

   **Example request**:

   .. sourcecode:: bash

        curl "http://example.com:34568/api/v1/dataset/622/batch_size"

   **Example response**:

   .. sourcecode:: json

        {
           "data": {
              "batch_size": "15"
           },
           "status": {
              "type": "SUCCESS"
           }
        }


   :query session_id: session id
   :statuscode 200: no error
   :statuscode 404: there's no such session id
   :statuscode 500: internal error


.. http:get:: /api/v1/dataset/(int:session_id)/batch_count

   Provides batch count for session ``session_id``.

   **Example request**:

   .. sourcecode:: bash

        curl "http://example.com:34568/api/v1/dataset/622/batch_count"

   **Example response**:

   .. sourcecode:: json

        {
           "data": {
              "batch_count": "15"
           },
           "status": {
              "type": "SUCCESS"
           }
        }


   :query session_id: session id
   :statuscode 200: no error
   :statuscode 404: there's no such session id
   :statuscode 500: internal error


.. http:get:: /api/v1/dataset/(int:session_id)/record_count

   Provides record count for session ``session_id``.

   **Example request**:

   .. sourcecode:: bash

        curl "http://example.com:34568/api/v1/dataset/622/record_count"

   **Example response**:

   .. sourcecode:: json

        {
           "data": {
              "record_count": "120"
           },
           "status": {
              "type": "SUCCESS"
           }
        }


   :query session_id: session id
   :statuscode 200: no error
   :statuscode 404: there's no such session id
   :statuscode 500: internal error


.. http:get:: /api/v1/dataset/(int:session_id)/next

   Provides next serialized batch data for session ``session_id``.
   This is the only request which does not return service_response_ json for successful response (status code 200). This is performance optimization. Returning service_response_ would require conversion to BASE64 format, which is quite costly when a lot of data is being transferred. All requests with status code different than 200 return service_response_ json.
   If RDMA is being used, then for successful response service_response_ is returned and data transfer happens via RDMA.

   **Example request**:

   .. sourcecode:: bash

        curl "http://example.com:34568/api/v1/dataset/622/next"


   :query session_id: session id
   :statuscode 200: batch fetch was successful
   :statuscode 404: there's no such session id or there is no more batch to provide (in this case status type will be ``END_OF_DATASET``)
   :statuscode 500: internal error


.. http:get:: /api/v1/dataset/(int:session_id)/reset

   Resets session ``session_id``.

   **Example request**:

   .. sourcecode:: bash

        curl "http://example.com:34568/api/v1/dataset/622/reset"

   **Example response**:

   .. sourcecode:: json

        {
           "status": {
              "type": "SUCCESS"
           }
        }


   :query session_id: session id
   :statuscode 200: session has been successfully reseted
   :statuscode 404: there's no such session id
   :statuscode 500: internal error


.. http:delete:: /api/v1/dataset/(int:session_id)

   Deletes session ``session_id``.

   **Example request**:

   .. sourcecode:: bash

        curl -XDELETE "http://example.com:34568/api/v1/dataset/622"

   **Example response**:

   .. sourcecode:: json

        {
           "status": {
              "type": "SUCCESS"
           }
        }


   :query session_id: session id
   :statuscode 200: session has been successfully delted
   :statuscode 404: there's no such session id
   :statuscode 500: internal error
