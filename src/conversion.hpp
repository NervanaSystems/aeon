/*
 Copyright 2017 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#pragma once
#define NP_NO_DEPRACATED_API NPY_1_7_API_VERSION
#define NUMPY_IMPORT_ARRAY_RETVAL

#include <Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "numpy/ndarrayobject.h"
#include "boundingbox.hpp"

namespace python
{
    void import_numpy();

    namespace conversion
    {
        namespace detail
        {
            cv::Mat to_mat(const PyObject* o);
            std::vector<nervana::boundingbox::box> to_boxes(const PyObject* o);
            PyObject* to_ndarray(const cv::Mat& mat);
            PyObject* to_list(const std::vector<nervana::boundingbox::box>& boxes);
        }

        template <typename T>
        struct convert
        {
            static T from_pyobject(const PyObject* from);
            static PyObject* to_pyobject(const T& from);
        };

        template <>
        struct convert<cv::Mat>
        {
            static cv::Mat from_pyobject(const PyObject* from) { return detail::to_mat(from); }
            static PyObject* to_pyobject(const cv::Mat& from)
            {
                cv::Mat to_convert = from.clone();
                python::import_numpy();
                return detail::to_ndarray(from);
            }
        };

        template <>
        struct convert<std::vector<nervana::boundingbox::box>>
        {
            static std::vector<nervana::boundingbox::box> from_pyobject(const PyObject* from)
            {
                return detail::to_boxes(from);
            }

            static PyObject* to_pyobject(const std::vector<nervana::boundingbox::box>& from)
            {
                python::import_numpy();
                return detail::to_list(from);
            }
        };
    }
}
