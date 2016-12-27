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

User Guide
==========

The aeon dataloader is designed to deal with large datasets from different modalities, including image, video, and audio, that may be too large to load directly into memory. We use a macrobatching approach, where the data is loaded in chunks (macrobatches) that are then split further into minibatches to feed the model.

The dataloader was created to provide an easy interface to configure the loader for custom datasets, and also to load data from disk to neon with minimal latency. The basic workflow is show in the schematic below:

.. image:: aeon_workflow.png

Users first perform **ingest**, which means converting their data into a format supported by the dataloader (if needed), and generating a manifest file in comma-separated values (csv) format. This file tells the dataloader where the input and target data reside.

Given a configuration file, the aeon dataloader takes care of the rest (green box). During operation, the first time a dataset is encountered, the dataloader will **cache** the data into `cpio <https://en.wikipedia.org/wiki/Cpio>`_ format, allowing for quick subsequent reads. This is an optional but highly recommended step.

During **provision**, the dataloader reads the data from disk, performs any needed transformations on-the-fly, transfers the data to device memory (e.g. GPU), and provisions the data to the model as an input-target pair. We use a multi-threaded library to hide the latency of these disk reads and operations in the device compute.

The caching step can be skipped by providing an empty ``cache_directory`` configuration (see below). In that case, the dataloader will read directly from the source data files.

Note that **ingest** (creation of the manifest files and any data formatting needed) occurs outside aeon by the user since they are specific to the dataset.

Data format
-----------

As mentioned above, users interact with the dataloader by providing two items:

1. Manifest file, a comma-separated file (*.csv).
2. Configuration parameters, as a python dictionary.

Operations such as generating training/testing splits, or balancing labels for imbalanced datasets should be implemented outside of the dataloader by the user during **ingest** to create the appropriate manifest files. Several example ingest scripts are in the neon repository.

Manifest file
-------------

The manifest file provides the dataloader with an input and target pair. The
manifest file should contain one record per line,

.. code-block:: bash

    <path_to_input_1>,<path_to_target_1>
    <path_to_input_2>,<path_to_target_2>
    ...
    <path_to_input_N>,<path_to_target_N>

In the image classification case,

.. code-block:: bash

    /image_dir/faces/naveen_rao.jpg,0.txt
    /image_dir/faces/arjun_bansal.jpg,0.txt
    /image_dir/faces/amir_khosrowshahi.jpg,0.txt
    /image_dir/fruits/apple.jpg,1.txt
    /image_dir/fruits/pear.jpg,1.txt
    /image_dir/animals/lion.jpg,2.txt
    /image_dir/animals/tiger.jpg,2.txt
    ...
    /image_dir/vehicles/toyota.jpg,3.txt

Note that above, the target labels should be text files with a single label.

For audio transcription, paths to target transcriptions are included:

.. code-block:: bash

    audio_sample_1.wav,audio_transcript_1.txt
    audio_sample_2.wav,audio_transcript_2.txt
    audio_sample_3.wav,audio_transcript_3.txt

For example formats of different modalities and problems, see the image, audio, and video sections.

Configuration
-------------

The dataloader configuration consists of a base loader config, then individual configs for the different modalities. These configs are captured by a python dictionary. For example,

.. code-block:: python

    image_config = dict(height=224, width=224)
    label_config = dict(binary=True)
    config = dict(type="image,label",
                  image=image_config,
                  label=label_config,
                  manifest_filename='train.csv',
                  minibatch_size=128)

Importantly, the ``type`` key indicates to the dataloader which input data type to expect, and the ``image`` and ``label`` keys correspond to additional configuration dictionaries. The dataloader currently supports:

- image classification (``type="image,label"``),
- image segmentation (``"image,pixelmask"``),
- image localization (``"image,localization"``),
- image bounding box (``"image,boundingbox"``),
- stereo depthmap (``"stereo_image,blob"``),
- video classification (``"video,label"``),
- audio classification (``"audio,label"``), and
- audio transcription (``"audio,transcription``").

For inference, types that provide the input only (e.g. ``type="image"``) are also supported.

aeon is designed to be modular and developer-friendly, so its relatively easy to write your own dataloader type and register it with the dataloader. For more information, see our Developer Guide.

The possible base loader configurations are the following (configurations without a default are required)

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :escape: ~
   :delim: |

   type (string)| *Required* | Provider type (e.g. "image, label").
   manifest_filename (string)| *Required* | Path to the manifest file.
   minibatch_size (int)| *Required* | Minibatch size. In neon, typically accesible via ``be.bsz``.
   manifest_root (string) | ~"~" | If provided, ``manifest_root`` is prepended to all manifest items with relative paths, while manifest items with absolute paths are left untouched. 
   cache_directory (string)| ~"~" | If provided, the dataloader will cache the data into ``*.cpio`` files for fast disk reads.
   macrobatch_size (int)| 0 | Size of the macrobatch archive files.
   subset_fraction (float)| 1.0 | Fraction of the dataset to iterate over. Useful when testing code on smaller data samples.
   shuffle_every_epoch (bool) | False | Shuffles the dataset order for every epoch
   shuffle_manifest (bool)| False | Shuffles the manifest file once at start.
   single_thread (bool)| False | Execute on a single thread
   random_seed (int)| 0 | Set the random seed.

Example python usage
--------------------

While aeon can be used within a purely C++ environment, we have included a python class ``DataLoader`` for integration into a python environment. As an example of an image classification dataset, we first specify a python dictionary with confguration settings:

.. code-block:: python

    image_config = dict(height=224, width=224, flip_enable=True)
    label_config = dict(binary=False)
    config = dict(type="image,label",
                  image=image_config,
                  label=label_config,
                  manifest_filename='train.csv',
                  minibatch_size=128)

The above configuration will, for each image, take a random crop of 224x224 pixels, and perform a horizontal flip with probability 0.5. We then generate our dataloader:

.. code-block:: python

    from aeon import DataLoader
    from neon.backends import gen_backend

    be = gen_backend(backend='gpu')
    train = DataLoader(config, be)

The backend argument above from neon tells the dataloader where to place the buffers to provision to the model.
