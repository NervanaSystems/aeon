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

The dataloader was created to provide an easy interface to configure the loader for custom datasets, and also to load data from disk to neon with minimal latency. During operation, the dataloader reads the data from disk, performs any needed transformations on-the-fly, transfers the data to device memory (e.g. GPU, if needed), and provisions the data to the model. We use a multi-threaded library to hide the latency of these disk reads and operations in the device compute.

Optionally (but highly recommended), the first time a dataset is encountered, the dataloader can cache the data into cpio files, allowing for quick disk reads. This is useful for datasets with many small items (e.g. images). To skip this step, provide an empty ``cache_directory`` configuration (see below).

Data format
-----------

Users interact with the dataloader by providing two items:
1. Manifest file, a comma-separated file (*.csv).
2. Configuration parameters, as a python dictionary.

Manifest file
-------------

The manifest file provides the dataloader with an input and target pair. The manifest file should contain a header line (that is ignored). Subsequent lines will have on record per line, formatted as:

.. code-block:: bash

    filename, label
    <path_to_input_1>,<target_1>
    <path_to_input_2>,<target_2>
    ...
    <path_to_input_N>,<target_N>

In the image classification case, the target data can be provided directly in the CSV file:

.. code-block:: bash

    filename, label
    /image_dir/faces/naveen_rao.jpg, 0
    /image_dir/faces/arjun_bansal.jpg, 0
    /image_dir/faces/amir_khosrowshahi.jpg, 0
    /image_dir/fruits/apple.jpg, 1
    /image_dir/fruits/pear.jpg, 1
    /image_dir/animals/lion.jpg, 2
    /image_dir/animals/tiger.jpg, 2
    ...
    /image_dir/vehicles/toyota.jpg, 3

For audio transcription, paths to target transcriptions are included:

.. code-block:: bash

    filename, label
    audio_sample_1.wav, audio_transcript_1.txt
    audio_sample_2.wav, audio_transcript_2.txt
    audio_sample_3.wav, audio_transcript_3.txt

For example formats of different modalities and problems, see the image, audio, and video sections.

Configuration
-------------

The dataloader configuration consists of several loader configurations, then individual configs for the different modalities. These configs are captured by a python dictionary. For example,

.. code-block:: python

    image_config = dict(height=224, width=224)
    label_config = dict(binary=True)
    config = dict(type="image,label",
                  image=image_config,
                  label=label_config,
                  manifest_filename='train.csv',
                  minibatch_size=128)

Importantly, the ``type`` key indicates to the dataloader which input data type to expect, and the ``image`` and ``label`` keys correspond to additional configuration dictionaries. The dataloader currently supports image classification (``type="image,label")``, segmentation (``"image,pixelmask"``), localization (``"image,localization"``), and bounding box (``"image,boundingbox"``); video classification (``"video,label"``); audio classification (``"audio,label"``) and transcription (``"audio,transcription``"). For inference, types that provide the input only (e.g. ``type="image"``) are also supported.

aeon is designed to be modular and developer-friendly, so its relatively easy to write your own dataloader type and register it with the dataloader. For more information, see our Developer Guide.

The possible base loader configurations are the following:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 20, 40
   :escape: ~

   type (string), , Provider type (e.g. "image, label").
   manifest_filename (string), , Path to the manifest file.
   minibatch_size (int), , Minibatch size. In neon, typically accesible via ``be.bsz``.
   cache_directory (string), "", If provided, the dataloader will cache the data into ``*.cpio`` files for fast disk reads.
   macrobatch_size (int), 0, ?????
   subset_fraction (float), 1.0, Fraction of the dataset to iterate over. Useful when testing code on smaller data samples.
   shuffle_every_epoch (bool), False, Shuffles the dataset order for every epoch
   shuffle_manifest (bool), False, Shuffles the manifest file once at start.
   single_thread (bool), False, ?????
   random_seed (int), 0, Set the random seed.








