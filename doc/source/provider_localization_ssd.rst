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

SSD Localization
----------------

The object localization provider (``type=localization_ssd``) is designed to work with the Single Shot MultiBox Detector model. The manifest should include paths to both the image but also the bounding box annotations:

.. code-block:: bash

    @FILE	FILE
    /annotations/0001.json	/image_dir/image0001.jpg
    /annotations/0002.json	/image_dir/image0002.jpg
    /annotations/0003.json	/image_dir/image0003.jpg

Each annotation is in the JSON format, which should have the main field "object" containing the bounding box in pixel coordinates, class, and difficulty of each object in the image. For example:
Top-left corner of a bounding box is ``xmin,ymin`` and bottom-right corner is ``xmax,ymax``.


.. code-block:: bash

   {
       "object": [
           {
               "bndbox": {
                   "xmax": 299,
                   "xmin": 100,
                   "ymax": 299,
                   "ymin": 200
               },
               "difficult": false,
               "name": "tvmonitor",
           },
           {
               "bndbox": {
                   "xmax": 56,
                   "xmin": 0,
                   "ymax": 54,
                   "ymin": 24
               },
               "difficult": false,
               "name": "person",
           },
       ],
   }

To generate these json files from the XML format used by some object localization datasets such as PASCALVOC, see the main neon repository.

Input parameters:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   class_names (vector of strings) | *Required* | List of class names (e.g. [~"person~", ~"tvmonitor~"]). Should match the names provided in the json annotation files.
   height | *Required* | Input height of the network, to which the image should be scaled to fit.
   width | *Required* | Input height of the network, to which the image should be scaled to fit.
   output_type (string) | ~"float~" | Output data type.
   max_gt_boxes (long) | 64 | Maximum number of ground truth boxes in dataset. Used to buffer the ground truth boxes.

This provider creates a set of six buffers that are consumed by the SSD model. Defining ``N`` as the ``max_gt_boxes`` parameter, we have the provisioned buffers in this order:

.. csv-table::
   :header: "ID", "Buffer", "Shape", "Description"
   :widths: 5, 20, 10, 45
   :delim: |

   0 | im_shape | (2, 1) | Shape of the input image.
   1 | gt_boxes | (N * 4, 1) | Ground truth bounding box coordinates, in normalized coordinates (between 0 and 1, where 1 is the last pixel). Boxes are padded into a larger buffer of size N. The format is [xmin,ymin,xmax,ymax].
   2 | num_gt_boxes | (1, 1) | Number of ground truth bounding boxes.
   3 | gt_classes | (N, 1) | Class label for each ground truth box.
   4 | is_difficult | (N, 1) | Indicates if each ground truth box has the difficult metadata property.


For SSD, we handle variable image sizes by resizing (warping) an image to the input size of the network. Note that the ``angle`` transformation is not supported.
