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

Image
=====

For image inputs, the dataloader has providers for classification, segmentation, and localization tasks. We support any image format that can be decoded with OpenCV. Our dataloader performs a series of customizable transformations on the image before provisioning the data to the model, shown in the schematic below:

.. image:: etl_image_transforms.png

1. Rotate the image by a random angle drawn from a uniform distribution between the provided ranges (parameter ``angle (int, int)``), and fill the padded regions with zeros.
2. Take a random crop of the image. The size of the crop is controlled by the parameters ``scale`` and ``do_area_scale``. Suppose the width is the short-side of the input image. By default, the crop width will then be a ``scale`` fraction of the width of the image. Optionally, if ``do_area_scale`` is enabled, then total area of the crop will be a ``scale`` fraction of the input image area. The proportions of the crop box match that of the output shape, unless horizontal_distortion is required.
3. Resize the cropped image to the desired output shape, defined by the parameters ``height`` and ``width``.
4. If required, apply any transformations (e.g. lighting, horizontal flip, photometric distortion)

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   height (uint) | *Required* | Height of provisioned image (pixels)
   width (uint) | *Required* | Width of provisioned image (pixels)
   output_type (string)| ~"uint8_t~"| Output data type.
   channel_major (bool)| True | Channel order of input image.
   channels (uint) | 3 | Number of channels in input image
   seed (int) | 0 | Random seed
   flip_enable (bool) | False | Apply horizontal flip with probability 0.5.
   scale (float, float) | (1, 1) | Fraction of image short-side length to take the crop.
   do_area_scale (bool) | False | Determine the crop size as a fraction of total area instead of short-side length.
   angle (int, int) | (0, 0) | Rotate image by a random angle drawn from the provided ranges. Angle is in degrees.
   lighting (float, float) | (0, 0) |
   horizontal_distortion (float, float) | (1, 1) | Change the aspect ratio by scaling the image width by a random factor.
   photometric (float, float) | (0, 0) | Apply randomly sampled contrast, brightness, saturation changes. Implements the colornoise perturbation as descrbed in Krizhevksy et al.
   center (bool) | False | Take the center crop of the image. If false, a randomly located crop will be taken.


The buffer provided to the model has shape ``(channels * height * width, bsz)``, where ``bsz`` is the batch size.

Classification
--------------

For classification (``type=image,label``), the manifest file should provide a path to the file, as well as to a file containing the label. For example:

.. code-block:: bash

    /image_dir/faces/naveen_rao.jpg,/classes/0.txt
    /image_dir/faces/arjun_bansal.jpg,/classes/0.txt
    /image_dir/faces/amir_khosrowshahi.jpg,/classes/0.txt
    /image_dir/fruits/apple.jpg,/classes/1.txt
    /image_dir/fruits/pear.jpg,/classes/1.txt
    /image_dir/animals/lion.jpg,/classes/2.txt
    /image_dir/animals/tiger.jpg,/classes/2.txt
    ...
    /image_dir/vehicles/toyota.jpg,/classes/3.txt

The label text files should contain a single integer between ``(0, num_classes-1)``.

The configuration for this provider accepts a few parameters:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   binary (bool) | True |
   output_type (string) | ~"int32_t~" | label data type

Note that data provisioned to the model is a ``(1, bsz)`` buffer, and not in one-hot format.

Segmentation
------------

Localization
------------

The object localization provider (``type=image,localization``) is designed to work with the Faster-RCNN model. The manifest should include paths to both the image but also the bounding box annotations:

.. code-block:: bash

    /image_dir/image0001.jpg,/annotations/0001.json
    /image_dir/image0002.jpg,/annotations/0002.json
    /image_dir/image0003.jpg,/annotations/0003.json

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

This provider creates a set of ten buffers that are consumed by the Faster-RCNN model. Defining ``A`` as the number of anchor boxes that tile the final convolutional feature map, and ``N`` as the ``max_gt_boxes`` parameter, we have the provisioned buffers in this order:

.. csv-table::
   :header: "Buffer", "Shape", "Description"
   :widths: 20, 10, 50
   :delim: |

   image_canvas | max_size * max_size | The Image is placed in the upper left corner of the canvas
   bb_targets | (4 * A, 1) | Bounding box regressions for the region proposal network
   bb_targets_mask | (4 * A, 1) | Bounding box target masks. Only positive labels have non-zero elements.
   labels | (2 * A, 1) | Target positive/negative labels for the region proposal network.
   labels_mask | (2 * A, 1) | Mask for the labels buffer. Includes ``rois_per_image`` non-zero elements.
   im_shape | (2, 1) | Shape of the input image.
   gt_boxes | (N * 4, 1) | Ground truth bounding box coordinates, already scaled by ``im_scale``. Boxes are padded into a larger buffer.
   num_gt_boxes | (1, 1) | Number of ground truth bounding boxes.
   gt_classes | (N, 1) | Class label for each ground truth box.
   im_scale | (1, 1) | Scaling factor that was applied to the image.
   is_difficult | (N, 1) | Indicates if each ground truth box has the difficult property.

For Faster-RCNN, we handle variable image sizes by padding an image into a fixed larger canvas of to pass to the network. The image configuration, instead of height and width as above, have two new required fields:

.. csv-table::
   :header: "Buffer", "Shape", "Description"
   :widths: 20, 10, 50
   :delim: |

   min_size (long) | *Required* | Minimize length of the input image's short-side.
   max_size (long) | *Required* | Maximum image length.

The input image is first scaled such that the short-side has length ``min_size``, then scaled to make sure that the maximum length does not exceed ``max_size``. The image is then placed in the upper left corner of a canvas with shape ``(max_size, max_size)`` for provisioning to the model. Note that transformations that change the image dimensions (``scale``, ``center``, ``angle``) are not supported.

