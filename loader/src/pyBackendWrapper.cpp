#include "pyBackendWrapper.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace nervana;
using namespace std;

pyBackendWrapper::pyBackendWrapper(PyObject* pBackend,
                                   std::shared_ptr<nervana::interface::config> dtm_config,
                                   std::shared_ptr<nervana::interface::config> tgt_config,
                                   int batchSize)
: _dtm_config(dtm_config), _tgt_config(tgt_config), _batchSize(batchSize), _pBackend(pBackend)
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

    _host_dlist = initPyList();
    _host_tlist = initPyList();
    _dev_dlist  = initPyList();
    _dev_tlist  = initPyList();
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
    Py_XDECREF(_host_dlist);
    Py_XDECREF(_host_tlist);
    Py_XDECREF(_dev_dlist);
    Py_XDECREF(_dev_tlist);
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
    wrap_buffer_pool(_host_dlist, outBuf[0], bufIdx, _dtm_config);
    wrap_buffer_pool(_host_tlist, outBuf[1], bufIdx, _tgt_config);

    PyObject* dArgs  = Py_BuildValue("iOO", bufIdx, _host_dlist, _dev_dlist);

    if (dArgs == NULL) {
        throw std::runtime_error("Unable to build args");
    }
    PyObject* dRes = PyObject_CallObject(_f_consume, dArgs);
    Py_XDECREF(dArgs);
    Py_XDECREF(dRes);

    PyObject* tArgs  = Py_BuildValue("iOO", bufIdx, _host_tlist, _dev_tlist);
    PyObject* tRes = PyObject_CallObject(_f_consume, tArgs);
    if (!tRes) {
        PyErr_Print();
    }
    Py_XDECREF(tArgs);
    Py_XDECREF(tRes);
    PyGILState_Release(gstate);

}

PyObject* pyBackendWrapper::get_dtm_tgt_pair(int bufIdx)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *dItem = PyList_GetItem(_dev_dlist, bufIdx);
    PyObject *hItem = PyList_GetItem(_dev_tlist, bufIdx);
    if (dItem == NULL || hItem == NULL) {
        throw std::runtime_error("Bad Index");
    }

    PyObject* dtm_tgt_pair = Py_BuildValue("OO", dItem, hItem);
    PyGILState_Release(gstate);
    return dtm_tgt_pair;
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

    // update strides.  not sure why this isn't happening correctly
    // inside PyArray_SimpleNewFromData
    npy_intp* strides = PyArray_STRIDES((PyArrayObject*)p_array);
    strides[1] *= dims[0];

    if (p_array == NULL) {
        throw std::runtime_error("Unable to wrap buffer pool in as python object");
    }

    if (PyList_SetItem(list, bufIdx, p_array) != 0) {
        throw std::runtime_error("Unable to add python array to list");
    }
}

