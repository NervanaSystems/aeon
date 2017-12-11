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

Faster-RCNN Localization
=========================

The object localization provider module (``type=localization_rcnn``) is designed to work with the Faster-RCNN model. The manifest should include paths to both the image but also the bounding box annotations:

.. code-block:: bash

    @FILE	FILE
    /image_dir/image0001.jpg	/annotations/0001.json
    /image_dir/image0002.jpg	/annotations/0002.json
    /image_dir/image0003.jpg	/annotations/0003.json

Each annotation is in the JSON format, which should have the main field "object" containing the bounding box, class, and difficulty of each object in the image. For example:
Top-left corner of a bounding box is ``xmin,ymin`` and bottom-right corner is ``xmax,ymax``.


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
   name (string) | ~"~" | Name prepended to the output buffer name
   class_names (vector of strings) | *Required* | List of class names (e.g. [~"person~", ~"tvmonitor~"]). Should match the names provided in the json annotation files.
   rois_per_image (long) | 256 | Number of anchors per image used for training.
   scaling_factor (float) | 0.0625 | Feature map scaling of the convolutional network portion. Default scaling is shown for VGG-16 network.
   base_size (long) | 16 | Base length of anchor boxes
   ratios (vector) | [0.5, 1, 2] | List of aspect ratios used to generate anchor boxes.
   scales (vector) | [8, 16, 32] | List of area sizes used to generate anchor boxes.
   negative_overlap (float) | 0.3 | Negative anchors have less than this value with any ground truth box.
   positive_overlap (float) | 0.7 | Positive anchors have greater than this value with at least one ground truth box.
   foreground_fraction (float) | 0.5 | Maximal fraction of total anchors that are positive.
   output_type (string) | ~"float~" | Output data type.
   max_gt_boxes (long) | 64 | Maximum number of ground truth boxes in dataset. Used to buffer the ground truth boxes.

This provider creates a set of eleven buffers that are consumed by the Faster-RCNN model. Defining ``A`` as the number of anchor boxes that tile the final convolutional feature map, ``M`` as the ``max_gt_boxes`` parameter, and ``N`` as the batch_size, we have the provisioned buffers in this order:

.. csv-table::
   :header: "Buffer Name", "Shape", "Description"
   :widths: 20, 10, 45
   :delim: |

   bb_targets | (N, 4 * A) | Bounding box regressions for the region proposal network
   bb_targets_mask | (N, 4 * A) | Bounding box target masks. Only positive labels have non-zero elements.
   labels | (N, 2 * A) | Target positive/negative labels for the region proposal network.
   labels_mask | (N, 2 * A) | Mask for the labels buffer. Includes ``rois_per_image`` non-zero elements.
   im_shape | (N, 2) | Shape of the input image.
   gt_boxes | (N, M * 4) | Ground truth bounding box coordinates, already scaled by ``im_scale``. Boxes are padded into a larger buffer. The format is [xmin,ymin,xmax,ymax].
   num_gt_boxes | (N) | Number of ground truth bounding boxes.
   gt_classes | (N, M) | Class label for each ground truth box.
   im_scale | (N) | Scaling factor that was applied to the image.
   is_difficult | (N, M) | Indicates if each ground truth box has the difficult property.

For Faster-RCNN, we handle variable image sizes by padding an image into a fixed canvas to pass to the network. The image configuration is used as above with the added flags ``crop_enable`` set to False and ```fixed_aspect_ratio``` set to True. These settings place the largest possible image in the output canvas in the upper left corner. Note that the ``angle`` transformation is not supported.
