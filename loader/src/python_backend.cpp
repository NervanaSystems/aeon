/*
 Copyright 2016 Nervana Systems Inc.
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

#include "python_backend.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace nervana;
using namespace std;

// TODO: need to use outputs buffer to iterate through named fixed_buffer_map

python_backend::python_backend(loader_async* loader_source,
                               PyObject* py_obj_backend,
                               const vector<nervana::shape_type>& oshape_types)
    : async_manager<fixed_buffer_map, std::vector<PyObject*>>(loader_source)
    , m_py_obj_backend(py_obj_backend)
    , m_oshape_types{oshape_types}
{
    gil_state state;

    if (m_py_obj_backend == NULL)
    {
        throw std::runtime_error("Python Backend object does not exist");
    }

    Py_INCREF(m_py_obj_backend);

    m_f_consume = PyObject_GetAttrString(m_py_obj_backend, "consume");

    if (m_f_consume == NULL)
    {
        throw std::runtime_error("Backend has no 'consume' attribute");
    }

    if (!PyCallable_Check(m_f_consume))
    {
        throw std::runtime_error("Backend 'consume' function does not exist or is not callable");
    }

    setup_buffers();
}

void python_backend::setup_buffers()
{
    gil_state         state;
    PyOS_sighandler_t sighandler = PyOS_getsig(SIGINT);
    if (_import_array() < 0)
    {
        throw std::runtime_error("numpy.core.multiarray failed to import");
    }
    PyOS_setsig(SIGINT, sighandler);

    for (uint32_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < element_count(); ++j)
        {
            PyObject* pylist = PyList_New(1);
            Py_INCREF(Py_None);
            if (PyList_SetItem(pylist, 0, Py_None) != 0)
            {
                throw std::runtime_error("Error initializing list");
            }
            m_containers[i].push_back(pylist);
        }
    }
}

python_backend::~python_backend()
{
    gil_state state;
    for (auto& container : m_containers)
    {
        for (auto d : container)
        {
            Py_XDECREF(d);
        }
        container.clear();
    }

    Py_XDECREF(m_f_consume);
    Py_XDECREF(m_py_obj_backend);
    finalize();
}

std::vector<PyObject*>* python_backend::filler()
{
    gil_state state;
    std::vector<PyObject*>* outputs = get_pending_buffer();
    fixed_buffer_map* inputs = m_source->next();

    affirm(inputs->size() == outputs->size(), "number of input elements do not match number of output elements");

    for (size_t i = 0; i < inputs->size(); ++i)
    {
        PyObject* npy_buffer = wrap_buffer_as_np_array(inputs->at(i), m_oshape_types[i]);

        PyObject* pArgs = Py_BuildValue("OO", npy_buffer, outputs->at(i));

        if (pArgs == NULL)
        {
            throw std::runtime_error("Unable to build args");
        }

        PyObject* pRes = PyObject_CallObject(m_f_consume, pArgs);
        if (!pRes)
        {
            PyErr_Print();
        }

        Py_XDECREF(npy_buffer);

        Py_XDECREF(pArgs);
        Py_XDECREF(pRes);
    }
    return outputs;
}


PyObject* python_backend::wrap_buffer_as_np_array(buffer_fixed_size_elements buf, const nervana::shape_type& st)
{
    // For now, we will collapse everything into two dimensions
    std::vector<npy_intp> dims;
    dims.push_back(object_count());

    for (auto& d : st.get_shape())
    {
        dims.push_back(d);
    }

    PyObject* p_array = PyArray_SimpleNewFromData(dims.size(), &dims[0],
                                                  st.get_otype().get_np_type(), buf.data());

    if (p_array == NULL)
    {
        throw std::runtime_error("Unable to wrap buffer pool in as python object");
    }

    return p_array;
}
