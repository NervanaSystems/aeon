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

Classification
==============

For classification, the manifest file should be a tab separated list providing a path to the file as well as a label index. For example:

.. code-block:: bash

    @FILE	ASCII_INT
    faces/naveen_rao.jpg	0
    faces/arjun_bansal.jpg	0
    faces/amir_khosrowshahi.jpg	0
    fruits/apple.jpg	1
    fruits/pear.jpg	1
    animals/lion.jpg	2
    animals/tiger.jpg	2
    ...
    vehicles/toyota.jpg	3

The label should contain a single integer between ``(0, num_classes-1)``.

Here is a c++ configuration example::

    int height = 224;
    int width = 224;
    int batch_size = 128;
    std::string manifest_root = "/test_data";
    std::string manifest      = "manifest.tsv";

    nlohmann::json js_image = {{"type", "image"},
                               {"height", height},
                               {"width", width},
                               {"channel_major", false}};
    nlohmann::json js_label = {{"type", "label"},
                               {"binary", false}};
    nlohmann::json js_aug = {{"type", "image"}
                             {"flip_enable", true}};
    nlohmann::json config = {{"manifest_root", manifest_root},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"etl", {js_image, js_label}},
                             {"augmentation", {js_aug}}};

    auto train_set = nervana::loader{config};


Segmentation
============

For segmentation problems (``type=image,pixelmask``), the input is an image, and the target output is a same-sized image where each pixel is assigned to a category. In the image below using the KITTI dataset, each pixel is assigned to object categories (sidewalk, road, car, etc.):

.. image:: segmentation_example.png

The manifest file contains paths to the input image, as well as the target image:

.. code-block:: bash

    @FILE	FILE
    image_dir/img1.jpg	mask_dir/mask1.png
    image_dir/img2.jpg	mask_dir/mask2.png
    image_dir/img3.jpg	mask_dir/mask3.png
    ...
    image_dir/imgN.jpg	mask_dir/maskN.png

Note that the target image should have a single channel only. If there are multiple channels, only the first channel from the target will be used. The image parameters are the same as above, and the pixelmask has zero configurations. Transformations such as photometric or lighting are applied to the input image only, and not applied to the pixel mask. The same cropping, flipping, and rotation settings are applied to both the image and the mask.

Here is a c++ configuration example::

    int height = 224;
    int width = 224;
    int batch_size = 128;
    std::string manifest_root = "/test_data";
    std::string manifest      = "manifest.tsv";

    nlohmann::json js_image = {{"type", "image"},
                               {"height", height},
                               {"width", width},
                               {"channel_major", false}};
    nlohmann::json js_target = {{"type", "pixelmap"},
                                {"height", height},
                                {"width", width}};
    nlohmann::json js_aug = {{"type", "image"}
                             {"flip_enable", true}};
    nlohmann::json config = {{"manifest_root", manifest_root},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"etl", {js_image, js_target}},
                             {"augmentation", {js_aug}}};

    auto train_set = nervana::loader{config};

Localization
============

The object localization provider (``type=image,localization``) is designed to work with the Faster-RCNN model. The manifest should include paths to both the image but also the bounding box annotations:

.. code-block:: bash

    @FILE	FILE
    image_dir/image0001.jpg	annotations/0001.json
    image_dir/image0002.jpg	annotations/0002.json
    image_dir/image0003.jpg	annotations/0003.json
    ...
    image_dir/imageN.jpg	annotations/N.json

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

The dataloader generates on-the-fly the anchor targets required for training neon's Faster-RCNN model. Several important parameters control this anchor generation process.

Here is a c++ configuration example::

    int height = 1000;
    int width = 1000;
    int batch_size = 1;
    std::string manifest_root = "/test_data";
    std::string manifest      = "manifest.tsv";
    std::vector<std::string> class_names = {"bicycle", "person"};

    nlohmann::json js_image = {{"type", "image"},
                               {"height", height},
                               {"width", width},
                               {"channel_major", false}};
    nlohmann::json js_local = {{"type", "localization"},
                               {"height", height},
                               {"width", width},
                               {"max_gt_boxes", 64},
                               {"class_names", class_names}};
    nlohmann::json js_aug = {{"type", "image"},
                             {"fixed_aspect_ratio", true},
                             {"crop_enable", false},
                             {"flip_enable", true}};
    nlohmann::json config = {{"manifest_root", manifest_root},
                             {"manifest_filename", manifest},
                             {"batch_size", batch_size},
                             {"iteration_mode", "INFINITE"},
                             {"etl", {js_image, js_local}},
                             {"augmentation", {js_aug}}};

    auto train_set = nervana::loader{config};


For Faster-RCNN, we handle variable image sizes by padding an image into a fixed canvas to pass to the network. The image configuration is used as above with the added flags ``crop_enable`` set to False and ```fixed_aspect_ratio``` set to True. These settings place the largest possible image in the output canvas in the upper left corner. Note that the ``angle`` transformation is not supported.
