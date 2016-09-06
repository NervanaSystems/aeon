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

Audio
=====

For audio data, Aeon has providers for classification and transcription tasks (use ``audio,label`` and ``audio,transcription``, respectively). Currently, audio must first be converted to 16-bit, single channel, ``*.wav format``. This is done by the user before calling the dataloader. We would recommend using the command line utility sox_. For example, to convert ``*.aiff`` files to the correct format, one could use:

.. code-block:: bash

    find . -name '*.aiff'  | parallel 'sox {} -b 16 {.}.wav channels 1 rate 16k && rm {}'

This single line finds all ``*.aiff`` files and converts them in-place and in parallel to correctly formatted ``.wav`` files using sox.

From there, one must create the manifest file, specifying paths to both the audio and label files. For transcription, we have:

.. code-block:: bash

    audio_sample_1.wav,audio_transcript_1.txt
    audio_sample_2.wav,audio_transcript_2.txt
    audio_sample_3.wav,audio_transcript_3.txt

where each transcript file should contain ASCII characters for the target transcription. All characters provided in the transcript file, including special characters like newlines, are provisioned to the model.

When the data is provisioned to the model, Aeon supports transforming the wave file into three different feature spaces: spectrograms (``feature_type: specgram``), mel-frequency spectral coefficients (``mfsc``), or mel-frequency cepstral coefficients (``mfcc``).

.. TODO: desribe specgrams, mfsc, and mfcc
.. TODO: graphics of transforms

Feature spaces
--------------

Spectrograms
************

Mel-frequency spectral coefficients
***********************************

Mel-frequency cepstral coefficients
***********************************

.. TODO: describe noise augmentation
Noise augmentation
------------------


Configuration parameters
------------------------

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 20, 40
   :delim: |

    max_duration (string) | *Required* | Maximum duration of any audio clip ("seconds" or "samples", e.g. "4 seconds")
    frame_stride (string) | *Required* | Interval between consecutive frames ("seconds" or "samples")
    frame_length (string) | *Required* | Duration of each frame ("seconds" or "samples")
    sample_freq_hz (uint32_t) | 16000 | Sample rate of input audio in hertz
    feature_type (string) | ~"specgram~" | Feature space to represent audio. One of "specgram", "mfsc", or "mfcc"
    window_type (string) | ~"hann~" | Window type for spectrogram generation
    num_filters (uint32_t) | 64 | Number of filters to use for mel-frequency transform (used for feature_type = "mfsc" or "mfcc")
    num_cepstra (uint32_t) | 40 | Number of cepstra to use (only for feature_type = "mfcc")
    noise_index_file (string) | | File of pathnames to noisy audio files
    noise_level (uniform distribution, float) | (0.0, 0.5) | How much noise to add (a value of 1 would be 0 dB SNR)
    add_noise (bernoulli probability, float) | 0.0 | Probability of adding noise
    noise_index (uniform distribution, uint32_t) | 0, UINT32_MAX | Index into noise_index_file
    noise_offset_fraction (uniform distribution, float) | (1.0, 1.0) | Offset from start of noise file
    time_scale_fraction (uniform distribution, float) | (1.0, 1.0) | Simple linear time-warping
    output_type (string) | ~"uint8_t~" | Output data type. Currently only "uint8_t" is supported.
    seed (int) | 0 | Random seed

.. TODO: noise parameters in example config
.. code-block:: python

    audio_config = dict(sampling_freq=16000,
                        max_duration="3 seconds",
                        frame_length="256 samples",
                        frame_stride="128 samples",
                        window_type="hann")

A pre-trained model can be evaluated on new audio by performing "inference" without a target. For this example, the dataloader configuration would be:

.. code-block:: python

    dataloader_config = dict(type="audio,",
                             audio=audio_config,
                             manifest_filename="/path/to/manifest.csv",
                             minibatch_size=minibatch_size)

Where the manifest file is simply a list of filenames, one per line.

Classification
--------------

.. TODO: classification-specific config

Transcription
-------------

.. TODO: transcription-specific config

.. code-block:: python

    transcription_config = dict(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ-_!? .,()",
                                max_length=25)
    dataloader_config = dict(type="audio,transcription",
                             audio=audio_config,
                             transcription=transcription_config,
                             manifest_filename="/path/to/manifest.csv",
                             minibatch_size=minibatch_size)





.. _sox: http://sox.sourceforge.net/
.. _neon: https://github.com/NervanaSystems/neon
