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

char_map
========
The character map provision module is useful for tasks involving text data.  Examples include text manipulation for natural language processing, or for producing reference transcriptions used for character-based speech recognition.  The processing pipeline is to simply take a sequence of characters and map them to a prescribed set of integer indices.

The mapping of characters to integers is done by providing an alphabet string, which determines the integer index by the position of the character in the alphabet.

The most common ways of providing the input text to aeon is either via pointer to transcription files::

    @FILE
    transcript_1.txt
    transcript_2.txt
    transcript_3.txt


or via strings directly in the manifest file::

    @STRING
    here is one text sequence one might process
    and here is another one
    each line is a different text sequence example

For the transcript file mode, all characters provided in the transcript file, including special characters like newlines, are provisioned to the model.

Transcription provisioning can be configured using the following parameters:

.. csv-table::
   :header: "Name", "Default", "Description"
   :widths: 20, 10, 50
   :delim: |
   :escape: ~

   alphabet (string)| *Required* | A string of symbols to be included in the target output (utf-8 input is supported)
   max_length (uint32_t) | *Required* | Maximum number of symbols in a target
   unknown_value (uint32_t) | 0 | Integer value to give to unknown characters. 0 causes them to be discarded.
   emit_length (bool) | False | Produce a buffer indicating the length of the input string
   output_type (string) | ~"uint32_t~" | transcript data type

.. code-block:: python

    transcription_config = {"type": "char_map",
                            "alphabet": "ABCDEFGHIJKLMNOPQRSTUVWXYZ-_!? .,()",
                            "max_length": 25}

    aeon_config = {"etl": (transcription_config),
                   "manifest_filename": "/path/to/manifest.tsv",
                   "batch_size": minibatch_size}


The buffers provisioned to the model are then:

.. csv-table::
    :header: "Buffer Name", "Shape", "Description"
    :widths: 20, 10, 45
    :delim: |
    :escape: ~

    char_map | ``(N, C)`` | Transcripts, where ``C = max transcript length``.
    char_map_length | ``(N)`` | Length of each transcript (``uint32_t``).  Only produced if ``emit_length`` is true in the configuration.
