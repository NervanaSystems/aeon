#include "pyBackendWrapper.hpp"
using namespace nervana;

pyBackendWrapper::pyBackendWrapper(PyObject* pBackend,
                                   count_size_type* dtmInfo,
                                   count_size_type* tgtInfo,
                                   int batchSize)
: _dtmInfo(dtmInfo), _tgtInfo(tgtInfo), _batchSize(batchSize), _pBackend(pBackend)
{
    if (_pBackend == NULL) {
        throw std::runtime_error("Python Backend object does not exist");
    }

    // PyObject* tstr = PyObject_GetAttrString(_pBackend, "test_string");
    // if (!PyString_Check(tstr))
    //     throw std::runtime_error("Backend string check failed");

    // printf("Got this string from backend object: %s\n", PyString_AsString(tstr));
    // Py_XDECREF(tstr);
    // PyThreadState tstate = PyEval_SaveThread();
    // PyGILState_STATE gstate;
    // gstate = PyGILState_Ensure();

    _f_consume = PyObject_GetAttrString(_pBackend, "consume");

    if (!PyCallable_Check(_f_consume)) {
        printf("Backend 'consume' function does not exist or is not callable\n");
        throw std::runtime_error("Backend 'consume' function does not exist or is not callable");
    }

    _host_dlist = initPyList();
    _host_tlist = initPyList();
    _dev_dlist  = initPyList();
    _dev_tlist  = initPyList();
    printf("List check result %d, %d\n", PyList_Check(_host_dlist), __LINE__);

    // PyGILState_Release(gstate);

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
    printf("Completed init for list\n");
    return pylist;
}

pyBackendWrapper::~pyBackendWrapper()
{
    Py_XDECREF(_pBackend);
    Py_XDECREF(_host_dlist);
    Py_XDECREF(_host_tlist);
    Py_XDECREF(_dev_dlist);
    Py_XDECREF(_dev_tlist);
    Py_XDECREF(_f_consume);
}

bool pyBackendWrapper::use_pinned_memory()
{
    return false;
    // PyGILState_STATE gstate;
    // gstate = PyGILState_Ensure();

    // PyObject *pinned_mem = PyObject_GetAttrString(_pBackend, "use_pinned_mem");
    // if (pinned_mem == NULL)
    //     return false;

    // bool result = false;
    // if (PyObject_IsTrue(pinned_mem)) {
    //     result = true;
    // }
    // Py_DECREF(pinned_mem);
    // PyGILState_Release(gstate);

    // return result;
}

// Copy to device.
void pyBackendWrapper::call_backend_transfer(BufferPair &outBuf, int bufIdx)
{
    // PyThreadState tstate = PyEval_SaveThread();
    // PyGILState_STATE gstate;
    // gstate = PyGILState_Ensure();
    // printf("gil state %d\n", gstate);
    // PyGILState_Release(gstate);
    printf("Hey look here %d \n", __LINE__);
    // int len_dlist = (int) PyList_Size(_host_dlist);
    // printf("Length of list %d\n", len_dlist);
    wrap_buffer_pool(_host_dlist, outBuf.first, bufIdx, _dtmInfo);
    wrap_buffer_pool(_host_tlist, outBuf.second, bufIdx, _tgtInfo);

    PyObject* dArgs  = Py_BuildValue("iOO", bufIdx, _host_dlist, _dev_dlist);

    if (dArgs == NULL)
        printf("Something went wrong here\n");
    PyObject* dRes = PyObject_CallObject(_f_consume, dArgs);
    Py_XDECREF(dArgs);
    Py_XDECREF(dRes);

    PyObject* tArgs  = Py_BuildValue("iOO", bufIdx, _host_tlist, _dev_tlist);
    PyObject* tRes = PyObject_CallObject(_f_consume, tArgs);
    Py_XDECREF(tArgs);
    Py_XDECREF(tRes);

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

void pyBackendWrapper::wrap_buffer_pool(PyObject *list, Buffer *buf, int bufIdx,
                                        count_size_type *typeInfo)
{
    printf("Hey look here %d \n", __LINE__);
    PyObject *hdItem = PyList_GetItem(list, bufIdx);
    printf("Hey look here %d \n", __LINE__);

    if (hdItem == NULL) {
        throw std::runtime_error("Bad Index");
    }
    if (hdItem != Py_None) {
        return;
    }
    int nd = 2;
    npy_intp dims[2] = {_batchSize, typeInfo->count};
    int nptype  = npy_type_map[typeInfo->type[0]];

    PyObject *p_array = PyArray_SimpleNewFromData(nd, dims, nptype,
                                                  static_cast<void *>(buf->_data));
    if (p_array == NULL) {
        throw std::runtime_error("Unable to wrap buffer pool in as python object");
    }
    Py_INCREF(p_array);

    if (PyList_SetItem(list, bufIdx, p_array) != 0) {
        throw std::runtime_error("Unable to add python array to list");
    }
}

