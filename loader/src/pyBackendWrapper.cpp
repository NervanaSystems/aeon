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

    _f_consume = PyObject_GetAttrString(_pBackend, "consume");

    if (!PyCallable_Check(_f_consume)) {
        throw std::runtime_error("Backend 'consume' function does not exist or is not callable");
    }
    initPyList(_host_dlist);
    initPyList(_host_tlist);
    initPyList(_dev_dlist);
    initPyList(_dev_tlist);
}

void pyBackendWrapper::initPyList(PyObject *pylist, int length)
{
    pylist = PyList_New(length);
    for (int i=0; i<length; ++i) {
        Py_INCREF(Py_None);
        if (PyList_SetItem(pylist, i, Py_None) != 0) {
            throw std::runtime_error("Error initializing list");
        }
    }
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
    PyObject *pinned_mem = PyObject_GetAttrString(_pBackend, "use_pinned_mem");
    bool result = false;

    if (pinned_mem != NULL && PyBool_Check(pinned_mem)) {
        result = (pinned_mem == Py_True) ? true : false;
    }
    Py_XDECREF(pinned_mem);
    return result;
}

// Copy to device.
void pyBackendWrapper::call_backend_transfer(BufferPair &outBuf, int bufIdx)
{
    wrap_buffer_pool(_host_dlist, outBuf.first, bufIdx, _dtmInfo);
    wrap_buffer_pool(_host_tlist, outBuf.second, bufIdx, _tgtInfo);

    PyObject* dArgs  = Py_BuildValue("iOO", bufIdx, _host_dlist, _dev_dlist);
    PyObject* dRes = PyObject_CallObject(_f_consume, dArgs);
    Py_XDECREF(dArgs);
    Py_XDECREF(dRes);

    PyObject* tArgs  = Py_BuildValue("iOO", bufIdx, _host_tlist, _dev_tlist);
    PyObject* tRes = PyObject_CallObject(_f_consume, tArgs);
    Py_XDECREF(tArgs);
    Py_XDECREF(tRes);
}


void pyBackendWrapper::wrap_buffer_pool(PyObject *list, Buffer *buf, int bufIdx,
                                        count_size_type *typeInfo)
{
    PyObject *hdItem = PyList_GetItem(list, bufIdx);
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

// void pyBackendWrapper::call_backend_transfer(const std::shared_ptr<BufferPool>& b_pool,
//                                              int bufIdx)
// {
//     if(PyList_Size(_host_dlist) == 0) {
//         wrap_buffer_pool(_host_dlist, _dtmInfo, b_pool, true);
//     }

//     if(PyList_Size(_host_tlist) == 0) {
//         wrap_buffer_pool(_host_tlist, _tgtInfo, b_pool, false);
//     }

//     PyObject* dArgs  = Py_BuildValue("iOO", bufIdx, _host_dlist, _dev_dlist);
//     PyObject* dRes = PyObject_CallObject(_f_consume, dArgs);
//     Py_XDECREF(dArgs);
//     Py_XDECREF(dRes);

//     PyObject* tArgs  = Py_BuildValue("iOO", bufIdx, _host_tlist, _dev_tlist);
//     PyObject* tRes = PyObject_CallObject(_f_consume, tArgs);
//     Py_XDECREF(tArgs);
//     Py_XDECREF(tRes);
// }


// void pyBackendWrapper::wrap_buffer_pool(PyObject* list, count_size_type *type_info,
//                                         const std::shared_ptr<BufferPool>& b_pool,
//                                         bool is_datum)
// {
//     assert(PyList_Check(list) == true);
//     assert(PyList_Size(list) == 0);

//     int nd = 2;
//     npy_intp dims[2] = {_batchSize, type_info->count};
//     int nptype  = npy_type_map[type_info->type[0]];

//     for (int i=0; i<b_pool->getCount(); ++i)
//     {
//         auto buf_ptr = is_datum ? b_pool->getPair(i).first : b_pool->getPair(i).second;
//         void *ptr = static_cast<void *>(buf_ptr->_data);

//         PyObject *p_array = PyArray_SimpleNewFromData(nd, dims, nptype, ptr);

//         if (p_array == NULL) {
//             throw std::runtime_error("Unable to wrap buffer pool in as python object");
//         }
//         // These appends automatically increment references on the npy arrays
//         if (PyList_Append(list, p_array) != 0) {
//             throw std::runtime_error("Unable to add python array to list");
//         }
//     }
// }

