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

For audio inputs, Aeon has providers for classification and transcription tasks. Currently, audio must first be converted to 16-bit, single channel, wav format. This is done through a process called "ingestion" and can usually be done very simply using the command line utility ``sox``. Once loaded, Aeon can transform the audio into three different feature spaces: spectrograms, mel-frequency spectral coefficients, or mel-frequency cepstral coefficients.

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 20, 40
   :delim: |

    max_duration (string) |  | Maximum duration of any audio clip ("seconds" or "samples", e.g. "4 seconds")
    frame_stride (string) | | Interval between consecutive frames ("seconds" or "samples")
    frame_length (string) | | Duration of each frame ("seconds" or "samples")
    sample_freq_hz (uint32_t) | 16000 | Sample rate of input audio in hertz
    feature_type (string) | specgram | Feature space to represent audio. One of "specgram", "mfsc", or "mfcc"
    window_type (string) | hann | Window type for spectrogram generation
    num_filters (uint32_t) | 64 | Number of filters to use for mel-frequency transform (used for feature_type = "mfsc" or "mfcc")
    num_cepstra (uint32_t) | 40 | Number of cepstra to use (only for feature_type = "mfcc")
    noise_index_file (string) | | File of pathnames to noisy audio files
    noise_level (uniform distribution, float) | (0.0, 0.5) | How much noise to add (a value of 1 would be 0 dB SNR)
    add_noise (bernoulli probability, float) | 0.0 | Probability of adding noise
    noise_index (uniform distribution, uint32_t) | 0, UINT32_MAX | Index into noise_index_file
    noise_offset_fraction (uniform distribution, float) | (1.0, 1.0) | Offset from start of noise file
    time_scale_fraction (uniform distribution, float) | (1.0, 1.0) | Simple linear time-warping
    type_string (string) | uint8_t | Input data type. Currently only "uint8_t" is supported.
    seed (int) | 0 | Random seed

((Still need: Example config, graphics showing transforms, longer description of each feature space, ))

.. a section on ingestion/format conversion. ((Needs: Example manifest, example sox script for ingest of a common dataset, any more details on required wav file format.))
.. -----

Classification
--------------


Transcription
-------------
