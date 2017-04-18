.. ---------------------------------------------------------------------------
.. Copyright 2015-2017 Nervana Systems Inc.
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

Video
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
  - ``c:v mjpeg`` sets the output video codec to MJPEG
  - ``-q:v 3`` sets the output codec compression quality
  - ``-f segment ...`` splits video into equal length segments. See the
    `ffmpeg documentation
    <https://www.ffmpeg.org/ffmpeg-formats.html#segment_002c-stream_005fsegment_002c-ssegment>`_ for details
  - ``-y`` overwrite output file without prompting

Splitting the videos into equal length segments as we did here is not necessary
in general for the aeon ``DataLoader``, but is helpful for training this
particular model in neon.

Once preprocessing is complete, a sample manifest tab-separated values (TSV) file must be created with
the absolute paths of the videos and the classification labels. For example::

  /video_dir/video1_location.avi    /labels/target_1.txt
  /video_dir/video2_location.avi    /labels/target_1.txt
  /video_dir/video3_location.avi    /labels/target_4.txt
  /video_dir/video4_location.avi    /labels/target_2.txt

Where the first column contains absolute paths to the preprocessed MJPEG videos
and the second column contains absolute paths to label files. The label files
in this case contain a single ASCII number indicating the correct class label
of this training example.

Next in our model training python script, we create a ``DataLoader`` config
dictionary as described in the :doc:`user guide <user_guide>` but with an
appropriate entry for video options:

.. code-block:: python

    config = dict(type="video,label",
                  video={'max_frame_count': 16,
                         'frame': {'height': 112,
                                   'width': 112,
                                   'scale': [0.875, 0.875]}},
                  label={'binary': False},
                  manifest_filename='train.csv',
                  batch_size=128)

The two current possible options for video configuration are:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 20, 40
   :delim: |
   :escape: ~

   max_frame_count (uint) | *Required* | Maximum number of frames to extract from video. Shorter samples will be zero padded.
   frame (object) | *Required* | An :doc:`Image configuration <image_etl>` for each frame extracted from the video.

The last step is to then create the Python ``DataLoader`` object specifying a
set of transforms to apply to the input data.

.. code-block:: python

    import json
    from aeon import DataLoader
    # config is defined in the code above
    dl = DataLoader(json.dumps(config))


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

