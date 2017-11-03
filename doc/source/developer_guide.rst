.. ---------------------------------------------------------------------------
.. Copyright 2015 Intel(R) Nervana(TM)
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

Developer's Guide
=================

Custom data types
-----------------

All data types must define three interfaces:

.. doxygenclass:: nervana::interface::extractor
   :members:
   :undoc-members:
   :outline:
   :no-link:

.. doxygenclass:: nervana::interface::transformer
   :members:
   :undoc-members:
   :outline:
   :no-link:

.. doxygenclass:: nervana::interface::loader
   :members:
   :undoc-members:
   :outline:
   :no-link:

and a configuration class inheriting from:

.. doxygenclass:: nervana::interface::config
   :members: add_shape_type
   :undoc-members:
   :outline:
   :no-link:


The config should probably make use of the following three macros to define the
entries that make up the configuration options.

.. doxygendefine:: ADD_SCALAR
   :outline:
   :no-link:

.. doxygendefine:: ADD_IGNORE
   :outline:
   :no-link:

.. doxygendefine:: ADD_DISTRIBUTION
   :outline:
   :no-link:

For example, in ``nervana::video::config``, the following snippet adds the
appropriate config options for ``frame`` and ``max_frame_count``:

.. code-block:: c++

    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(max_frame_count, mode::REQUIRED),
        ADD_IGNORE(frame)
    };

Then in the ``config`` method, these attributes are then added using the
``add_shape_type`` method:

.. code-block:: c++

    config(nlohmann::json js) :
    frame(js["frame"])
    {
        if(js.is_null()) {
            throw std::runtime_error("missing video config in json config");
        }

        for(auto& info : config_list) {
            info->parse(js);
        }
        verify_config("video", config_list, js);

        // channel major only
        add_shape_type({frame.channels, max_frame_count, frame.height, frame.width},
                        frame.type_string);
        }


.. _gtest: https://github.com/google/googletest
