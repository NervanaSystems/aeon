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

boundingbox
===========

The object localization provider (``type=localization``) is designed to work with the Faster-RCNN model. The manifest should include paths to both the image but also the bounding box annotations:

.. code-block:: bash

    /image_dir/image0001.jpg    /annotations/0001.json
    /image_dir/image0002.jpg    /annotations/0002.json
    /image_dir/image0003.jpg    /annotations/0003.json

Each annotation is in the JSON format, which should have the main field "object" containing the bounding box, class, and difficulty of each object in the image. For example:


.. code-block:: bash

   {
       "object": [
           {
               "bndbox": {
                   "xmax": 262,
                   "xmin": 207,
                   "ymax": 75,
                   "ymin": 10
               },
               "difficult": false,
               "name": "tvmonitor",
           },
           {
               "bndbox": {
                   "xmax": 431,
                   "xmin": 369,
                   "ymax": 335,
                   "ymin": 127
               },
               "difficult": false,
               "name": "person",
           },
       ],
   }

To generate these json files from the XML format used by some object localization datasets such as PASCALVOC, see the main neon repository.

The dataloader generates on-the-fly the anchor targets required for training neon's Faster-RCNN model. Several important parameters control this anchor generation process:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   height (uint) | *Required* | Height of provisioned image (pixels)
   width (uint) | *Required* | Width of provisioned image (pixels)
   class_names (vector of strings) | *Required* | List of class names (e.g. [~"person~", ~"tvmonitor~"]). Should match the names provided in the json annotation files.
   max_bbox_count (uint) | *Required* | Maximum bounding boxes in the output buffer.
   name (string) | ~"~" | Name prepended to the output buffer name
   output_type (string)| ~"uint8_t~"| Output data type.

This provider creates a set of eleven buffers that are consumed by the Faster-RCNN model. Defining ``A`` as the number of anchor boxes that tile the final convolutional feature map, and ``N`` as the ``max_gt_boxes`` parameter, we have the provisioned buffers in this order:

.. csv-table::
   :header: "Buffer Name", "Shape", "Description"
   :widths: 20, 10, 45
   :delim: |

   boundingbox | (N * 4, 1) | Ground truth bounding box coordinates. Boxes are padded into a larger buffer.

For Faster-RCNN, we handle variable image sizes by padding an image into a fixed canvas to pass to the network. The image configuration is used as above with the added flags ``crop_enable`` set to False and ```fixed_aspect_ratio``` set to True. These settings place the largest possible image in the output canvas in the upper left corner. Note that the ``angle`` transformation is not supported.
