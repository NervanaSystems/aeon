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

image
=====

For image provision module is used for image classification, segmentation, and localization tasks. We support any image format that can be decoded with OpenCV.

The complete table of configuration parameters is shown below:

.. csv-table::
   :header: "Parameter", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   height (uint) | *Required* | Height of provisioned image (pixels)
   width (uint) | *Required* | Width of provisioned image (pixels)
   name (string) | ~"~" | Name prepended to the output buffer name
   output_type (string)| ~"uint8_t~"| Output data type.
   channels (uint) | 3 | Number of channels in input image
   channel_major (bool)| True | Load the pixel buffer in channel major order (that is, all pixels from blue channel contiguous, followed by all pixels from green channel, followed by all pixels from the red channel).  The alternative is to have the color channels for each pixel located adjacent to each other (b1g1r1b2g2r2 rather than b1b2g1g2r1r2).
   seed (int) | 0 | Random seed

The buffers provisioned to the model are:

.. csv-table::
   :header: "Buffer Name", "Shape", "Description"
   :widths: 20, 10, 45
   :delim: |
   :escape: ~

   image | ``(N, C, H, W)`` or ``(N, H, W, C)``| Extracted and transformed image, where ``C = channels``, ``H = height``, ``W = width``, and ``N = bsz`` (the batch size).  Layout is dependent on whether ``channel_major`` mode is true or not
