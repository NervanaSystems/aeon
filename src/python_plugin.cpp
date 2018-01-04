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
#include <Python.h>
#include <cstdlib>
#include "python_plugin.hpp"
#include "python_utils.hpp"

namespace nervana
{
    template <typename T>
    T plugin::augment(const std::string& methodname, const T& in_data)
    {
        python::ensure_gil gil;
        using convert = typename ::python::conversion::convert<T>;

        PyObject* arg = convert::to_pyobject(in_data);

        PyObject* ret_val = PyObject_CallMethodObjArgs(
            instance, PyUnicode_FromString(methodname.c_str()), arg, NULL);

        T out;
        if (ret_val != NULL)
        {
            out = convert::from_pyobject(ret_val);
        }
        else
        {
            PyObject *err_type, *err_value, *err_traceback;
            PyErr_Fetch(&err_type, &err_value, &err_traceback);
            const char* err_msg = PyBytes_AsString(err_value);

            std::stringstream ss;
            ss << "Python has failed with error message: " << err_msg << std::endl;
            PyErr_Restore(err_type, err_value, err_traceback);
            PyErr_Print();
            PyErr_Restore(err_type, err_value, err_traceback);
            throw std::runtime_error(ss.str());
        }

        return out;
    }

    plugin::plugin(const std::string& fname, const std::string& params)
        : filename(fname)
    {
        python::ensure_gil gil;
        handle = PyImport_ImportModule(filename.c_str());

        if (!handle)
        {
            PyObject *err_type, *err_value, *err_traceback;
            PyErr_Fetch(&err_type, &err_value, &err_traceback);
            const char* err_msg = PyBytes_AsString(err_value);

            std::stringstream ss;
            ss << "python module not loaded" << std::endl;
            ss << "Python has failed with error message: " << err_msg << std::endl;
            PyErr_Restore(err_type, err_value, err_traceback);
            PyErr_Print();
            PyErr_Restore(err_type, err_value, err_traceback);
            throw std::runtime_error(ss.str());
        }

        klass = PyObject_GetAttrString(handle, "plugin");

        if (!klass)
        {
            PyObject *err_type, *err_value, *err_traceback;
            PyErr_Fetch(&err_type, &err_value, &err_traceback);
            const char* err_msg = PyBytes_AsString(err_value);

            std::stringstream ss;
            ss << "python class not loaded" << std::endl;
            ss << "Python has failed with error message: " << err_msg << std::endl;
            PyErr_Restore(err_type, err_value, err_traceback);
            PyErr_Print();
            PyErr_Restore(err_type, err_value, err_traceback);
            throw std::runtime_error(ss.str());
        }

        PyObject* arg_tuple = PyTuple_New(1);
        PyTuple_SetItem(arg_tuple, 0, PyUnicode_FromString(params.c_str()));

        instance = PyObject_CallObject(klass, arg_tuple);
        if (!instance)
        {
            PyErr_Print();

            std::stringstream ss;
            ss << "Python plugin instance not created." << std::endl
               << "Module:" << fname.c_str() << std::endl
               << "Params:" << params.c_str() << std::endl;
            throw std::runtime_error(ss.str());
        }
    }

    void plugin::prepare()
    {
        python::ensure_gil gil;
        PyObject_CallMethodObjArgs(instance, PyUnicode_FromString("prepare"), NULL);
    }

    cv::Mat plugin::augment_image(const cv::Mat& m) { return augment("augment_image", m); }
    std::vector<boundingbox::box>
        plugin::augment_boundingbox(const std::vector<boundingbox::box>& boxes)
    {
        return augment("augment_boundingbox", boxes);
    }

    cv::Mat plugin::augment_audio(const cv::Mat& m) { return augment("augment_audio", m); }
    cv::Mat plugin::augment_pixel_mask(const cv::Mat& m)
    {
        return augment("augment_pixel_mask", m);
    }

    cv::Mat plugin::augment_depthmap(const cv::Mat& m) { return augment("augment_depthmap", m); }
}
