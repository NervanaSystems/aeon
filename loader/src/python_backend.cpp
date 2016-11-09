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

python_backend::python_backend(PyObject* py_obj_backend) :
    _py_obj_backend(py_obj_backend)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    if (_py_obj_backend == NULL) {
        throw std::runtime_error("Python Backend object does not exist");
    }

    Py_INCREF(_py_obj_backend);
    _f_consume = PyObject_GetAttrString(_py_obj_backend, "consume");

    if (_f_consume == NULL) {
        throw std::runtime_error("Backend has no 'consume' attribute");
    }

    if (!PyCallable_Check(_f_consume)) {
        throw std::runtime_error("Backend 'consume' function does not exist or is not callable");
    }

    PyGILState_Release(gstate);
}

void python_backend::setup_buffers(const vector<nervana::shape_type>& oshape_types, int batchSize)
{
    _oshape_types = oshape_types;
    _batchSize = batchSize;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyOS_sighandler_t sighandler = PyOS_getsig(SIGINT);
    if (_import_array() < 0) {
        throw std::runtime_error("numpy.core.multiarray failed to import");
    }
    PyOS_setsig(SIGINT, sighandler);

    for (uint32_t i = 0; i < _oshape_types.size(); ++i)
    {
        _host_lists.push_back(initPyList());
        _dev_lists.push_back(initPyList());
    }

    PyGILState_Release(gstate);
}

PyObject* python_backend::get_shapes()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    uint32_t num_shapes = _oshape_types.size();
    PyObject* all_shapes = PyTuple_New(num_shapes);
    for (uint32_t idx = 0; idx < num_shapes; ++idx)
    {
        auto shapevec = _oshape_types[idx].get_shape();
        PyObject* this_shape = PyTuple_New(shapevec.size());
        for (uint32_t dim = 0; dim < shapevec.size(); dim++)
        {
            PyTuple_SetItem(this_shape, dim, Py_BuildValue("i", shapevec[dim]));
        }
        PyTuple_SetItem(all_shapes, idx, this_shape);
    }


    PyGILState_Release(gstate);

    return all_shapes;
}

PyObject* python_backend::initPyList(int length)
{
    PyObject* pylist = PyList_New(length);
    for (int i=0; i<length; ++i) {
        Py_INCREF(Py_None);
        if (PyList_SetItem(pylist, i, Py_None) != 0) {
            throw std::runtime_error("Error initializing list");
        }
    }
    return pylist;
}

void python_backend::clear_buffers()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    for (auto h: _host_lists) {
        Py_XDECREF(h);
    }
    for (auto d: _dev_lists) {
        Py_XDECREF(d);
    }
    _host_lists.clear();
    _dev_lists.clear();
    PyGILState_Release(gstate);
}

python_backend::~python_backend()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    Py_XDECREF(_f_consume);
    Py_XDECREF(_py_obj_backend);
    PyGILState_Release(gstate);

}

bool python_backend::use_pinned_memory()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    bool result = false;
    PyObject *pinned_mem = PyObject_GetAttrString(_py_obj_backend, "use_pinned_mem");
    if (pinned_mem != NULL) {
        if (PyObject_IsTrue(pinned_mem)) {
            result = true;
        }
        Py_DECREF(pinned_mem);
    }
    PyGILState_Release(gstate);
    return result;
}

// Copy to device.
void python_backend::call_backend_transfer(buffer_out_array &outBuf, int bufIdx)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    affirm(_host_lists.size() == _oshape_types.size(), "host lists size does not match oshape size");
    affirm(_dev_lists.size() == _host_lists.size(), "dev list size does not match host lists size");

    for (uint32_t i=0; i<_host_lists.size(); i++) {
        wrap_buffer_pool(_host_lists[i], outBuf[i], bufIdx, _oshape_types[i]);
        PyObject* pArgs  = Py_BuildValue("iOO", bufIdx, _host_lists[i], _dev_lists[i]);

        if (pArgs == NULL) {
            throw std::runtime_error("Unable to build args");
        }
        PyObject* pRes = PyObject_CallObject(_f_consume, pArgs);
        if (!pRes) {
            PyErr_Print();
        }
        Py_XDECREF(pArgs);
        Py_XDECREF(pRes);
    }
    PyGILState_Release(gstate);
}

PyObject* python_backend::get_host_tuple(int bufIdx)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();


    int provider_count = _dev_lists.size();
    PyObject *result = PyTuple_New(provider_count);
    if (result == NULL)
    {
        throw std::runtime_error("couldn't make tuple");
    }

    for (int pidx = 0; pidx < provider_count; pidx++) {
        PyObject* value = PyList_GetItem(_dev_lists[pidx], bufIdx);

        if (value == NULL) {
            throw std::runtime_error("Bad Index");
        }
        Py_INCREF(value);
        PyTuple_SetItem(result, pidx, value);
    }

    PyGILState_Release(gstate);
    return result;
}

void python_backend::wrap_buffer_pool(PyObject *list, buffer_out *buf, int bufIdx,
                                        const nervana::shape_type& st)
{
    PyObject *hdItem = PyList_GetItem(list, bufIdx);

    if (hdItem == NULL) {
        throw std::runtime_error("Bad Index");
    }
    if (hdItem != Py_None) {
        return;
    }

    // For now, we will collapse everything into two dimensions
    int all_dims = std::accumulate(st.get_shape().begin(),
                                   st.get_shape().end(),
                                   1, std::multiplies<uint32_t>());
    int nd = 2;
    npy_intp dims[2] = {_batchSize, all_dims};
    if (st.flatten_all_dims()) {
        dims[0] = _batchSize * all_dims;
        dims[1] = 1;
    }

    PyObject *p_array = PyArray_SimpleNewFromData(nd, dims, st.get_otype().np_type, buf->data());

    if (p_array == NULL) {
        throw std::runtime_error("Unable to wrap buffer pool in as python object");
    }

    if (PyList_SetItem(list, bufIdx, p_array) != 0) {
        throw std::runtime_error("Unable to add python array to list");
    }
}

