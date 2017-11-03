.. ---------------------------------------------------------------------------
.. Copyright 2015-2017 Intel(R) Nervana(TM)
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
.. neon documentation master file

Nervana aeon
================

:Release: |version|
:Date: |today|

aeon_ is Intel Nervana_ ’s project to enable fast and flexible access to data when
training your neon_ neural network models. You probably want to get started
with :doc:`installation <getting_started>` then on to the :doc:`user guide <user_guide>` followed by the more detailed :doc:`developer's guide <developer_guide>`.

This project is evolving quickly, so feel free to send us feature requests and
bug reports at aeon_issues_ .

.. _nervana: http://nervanasys.com
.. _neon: https://github.com/NervanaSystems/neon
.. _aeon: https://github.com/NervanaSystems/aeon
.. _aeon_issues: https://github.com/NervanaSystems/aeon/issues


.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Introduction

   getting_started
   user_guide
   developer_guide
   service

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Providers

   provider_image
   provider_label
   provider_audio
   provider_localization_rcnn
   provider_localization_ssd
   provider_pixelmask
   provider_boundingbox
   provider_blob
   provider_video
   provider_charmap
   provider_labelmap

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Augmentation

   augment_image
   augment_audio

.. toctree::
   :hidden:
   :maxdepth: 0
   :caption: Config Examples

   config_examples

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Full APIs

   python_api
   cpp_api
   service_api
