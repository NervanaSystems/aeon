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

video
=====

For training models using video data in neon, aeon requires the videos to be
MJPEG encoded. A full example of how to perform this initial preprocessing is
demonstrated in the `neon C3D example`_ in the neon repository based off the
C3D_ model architecture trained using the UCF101_ dataset. The preprocessing in
that example is achieved using the following ffmpeg_ command::

  ffmpeg -v quiet -i $VIDPATH \
         -an -vf scale=171:128 -framerate 25 \
         -c:v mjpeg -q:v 3 \
         -f segment -segment_time 0.64 -reset_timestamps 1 \
         -segment_list ${VIDPATH%.avi}.csv \
         -segment_list_entry_prefix `dirname $VIDPATH`/ \
         -y ${VIDPATH%.avi}_%02d.avi

Breaking this command down:

  - ``-an`` disables the audio stream
  - ``-vf scale=171:128`` scales the video frames to 171 by 128 pixels
  - ``-framerate 25`` sets the output framerate to 25 frames per second
  - ``-c:v mjpeg`` sets the output video codec to MJPEG
  - ``-q:v 3`` sets the output codec compression quality
  - ``-f segment ...`` splits video into equal length segments. See the
    `ffmpeg documentation
    <https://www.ffmpeg.org/ffmpeg-formats.html#segment_002c-stream_005fsegment_002c-ssegment>`_ for details
  - ``-y`` overwrite output file without prompting

Splitting the videos into equal length segments as we did here is not necessary
in general for the aeon ``DataLoader``, but is helpful for training this
particular model in neon.

Once preprocessing is complete, a sample manifest TSV file must be created with
the absolute paths of the videos and the classification labels. For example::

  @FILE	ASCII_INT
  /video_dir/video1_location.avi	1
  /video_dir/video2_location.avi	1
  /video_dir/video3_location.avi	4
  /video_dir/video4_location.avi	2

Where the first column contains absolute paths to the preprocessed MJPEG videos
and the second column contains numbers corresponding to a class label.

The configuration options for the video etl module are:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 20, 40
   :delim: |
   :escape: ~

   max_frame_count (uint) | *Required* | Maximum number of frames to extract from video. Shorter samples will be zero padded.
   frame (object) | *Required* | An :doc:`Image configuration <provider_image>` for each frame extracted from the video.
   name (string) | ~"~" | Name prepended to the output buffer name

The output buffer provisioned to the model from the video module is described below:

.. csv-table::
   :header: "Buffer Name", "Shape", "Description"
   :widths: 20, 10, 45
   :delim: |
   :escape: ~

   video | ``(N, C, D, H, W)`` | Where ``N`` is the batch size, ``C`` is the channel count, ``D`` is the number of frames, ``H`` is the height of each frame, ``W`` is the width of each frame.


Next in our model training python script, we create a ``DataLoader`` config
dictionary as described in the :doc:`user guide <user_guide>` but with an
appropriate entry for video options:

.. code-block:: python

    video_config = {"type": "video",
                    "max_frame_count": 16,
                    "frame": {"height": 112,
                              "width": 112}}
    label_config = {"type": "label",
                    "binary": False}

    augmentation_config = {"type": "image",
                           "scale": [0.875, 0.875]}

    aeon_config = {"manifest_filename": "train.csv",
                   "etl": (video_config, label_config),
                   "augmentation": (augmentation_config),
                   "batch_size": 128}


The last step is to then create the Python ``DataLoader`` object specifying a
set of transforms to apply to the input data.

.. code-block:: python

    import json
    from aeon import DataLoader

    train_set = DataLoader(aeon_config)

Again, for the full example consult the complete `neon C3D example`_ in the
neon repository.

.. _neon C3D example: https://github.com/NervanaSystems/neon/tree/master/examples/video-c3d
.. _C3D: http://arxiv.org/pdf/1412.0767v4.pdf
.. _UCF101: http://crcv.ucf.edu/data/UCF101.php
.. _ffmpeg: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

Citation
~~~~~~~~
::

  Learning Spatiotemporal Features with 3D Convolutional Networks
  http://arxiv.org/pdf/1412.0767v4.pdf

  http://vlg.cs.dartmouth.edu/c3d/

  https://github.com/facebook/C3D

