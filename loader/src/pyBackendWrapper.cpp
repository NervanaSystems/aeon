#include "pyBackendWrapper.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace nervana;
using namespace std;

pyBackendWrapper::pyBackendWrapper(PyObject* pBackend,
                                   vector<shared_ptr<nervana::interface::config>> configs,
                                   int batchSize)
: _configs(configs), _batchSize(batchSize), _pBackend(pBackend)
{
    if (_pBackend == NULL) {
        throw std::runtime_error("Python Backend object does not exist");
    }

    Py_INCREF(_pBackend);
    _f_consume = PyObject_GetAttrString(_pBackend, "consume");

    if (!PyCallable_Check(_f_consume)) {
        throw std::runtime_error("Backend 'consume' function does not exist or is not callable");
    }

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyOS_sighandler_t sighandler = PyOS_getsig(SIGINT);
    import_array();
    PyOS_setsig(SIGINT, sighandler);
    PyGILState_Release(gstate);

    for (uint i = 0; i < _configs.size(); ++i)
    {
        _host_lists.push_back(initPyList());
        _dev_lists.push_back(initPyList());
    }
}

PyObject* pyBackendWrapper::initPyList(int length)
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

pyBackendWrapper::~pyBackendWrapper()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    for (auto h: _host_lists) {
        Py_XDECREF(h);
    }
    for (auto d: _dev_lists) {
        Py_XDECREF(d);
    }

    Py_XDECREF(_f_consume);
    Py_XDECREF(_pBackend);
    PyGILState_Release(gstate);

}

bool pyBackendWrapper::use_pinned_memory()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    bool result = false;
    PyObject *pinned_mem = PyObject_GetAttrString(_pBackend, "use_pinned_mem");
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
void pyBackendWrapper::call_backend_transfer(buffer_out_array &outBuf, int bufIdx)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    assert(_host_lists.size() == _configs.size());
    assert(_dev_lists.size() == _host_lists.size());

    for (uint i=0; i<_host_lists.size(); i++) {
        wrap_buffer_pool(_host_lists[i], outBuf[i], bufIdx, _configs[i]);
        PyObject* pArgs  = Py_BuildValue("iOO", bufIdx, _host_lists[i], _dev_lists[i]);

        if (pArgs == NULL) {
            throw std::runtime_error("Unable to build args");
        }
        PyObject* pRes = PyObject_CallObject(_f_consume, pArgs);
        Py_XDECREF(pArgs);
        Py_XDECREF(pRes);
    }
    PyGILState_Release(gstate);
}

PyObject* pyBackendWrapper::get_dtm_tgt_pair(int bufIdx)
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

void pyBackendWrapper::wrap_buffer_pool(PyObject *list, buffer_out *buf, int bufIdx,
                                        const shared_ptr<nervana::interface::config> &cfg)
{
    PyObject *hdItem = PyList_GetItem(list, bufIdx);

    if (hdItem == NULL) {
        throw std::runtime_error("Bad Index");
    }
    if (hdItem != Py_None) {
        return;
    }

    // For now, we will collapse everything into two dimensions
    int all_dims = std::accumulate(cfg->get_shape().begin(),
                                   cfg->get_shape().end(),
                                   1, std::multiplies<uint32_t>());
    int nd = 2;
    npy_intp dims[2] = {_batchSize, all_dims};

    PyObject *p_array = PyArray_SimpleNewFromData(nd, dims, cfg->get_type().np_type, buf->data());

    // Note: Reverting this back to default strides, it appears correct (AP)
    // update strides.  not sure why this isn't happening correctly
    // inside PyArray_SimpleNewFromData
    // npy_intp* strides = PyArray_STRIDES((PyArrayObject*)p_array);
    // strides[1] *= dims[0];

    if (p_array == NULL) {
        throw std::runtime_error("Unable to wrap buffer pool in as python object");
    }

    if (PyList_SetItem(list, bufIdx, p_array) != 0) {
        throw std::runtime_error("Unable to add python array to list");
    }
}

