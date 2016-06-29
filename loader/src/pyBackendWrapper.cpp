#include "pyBackendWrapper.hpp"
using namespace nervana;

static std::map<char, int> npy_type_map = {{NPY_BOOLLTR, NPY_BOOL},
                                           {NPY_BYTELTR, NPY_BYTE},
                                           {NPY_UBYTELTR, NPY_UBYTE},
                                           {NPY_SHORTLTR, NPY_SHORT},
                                           {NPY_USHORTLTR, NPY_USHORT},
                                           {NPY_INTLTR, NPY_INT},
                                           {NPY_UINTLTR, NPY_UINT},
                                           {NPY_LONGLTR, NPY_LONG},
                                           {NPY_ULONGLTR, NPY_ULONG},
                                           {NPY_LONGLONGLTR, NPY_LONGLONG},
                                           {NPY_ULONGLONGLTR, NPY_ULONGLONG},
                                           {NPY_HALFLTR, NPY_HALF},
                                           {NPY_FLOATLTR, NPY_FLOAT},
                                           {NPY_DOUBLELTR, NPY_DOUBLE},
                                           {NPY_LONGDOUBLELTR, NPY_LONGDOUBLE},
                                           {NPY_CFLOATLTR, NPY_CFLOAT},
                                           {NPY_CDOUBLELTR, NPY_CDOUBLE},
                                           {NPY_CLONGDOUBLELTR, NPY_CLONGDOUBLE},
                                           {NPY_OBJECTLTR, NPY_OBJECT},
                                           {NPY_STRINGLTR, NPY_STRING},
                                           {NPY_STRINGLTR2, NPY_STRING},
                                           {NPY_UNICODELTR, NPY_UNICODE},
                                           {NPY_VOIDLTR, NPY_VOID},
                                           {NPY_DATETIMELTR, NPY_DATETIME},
                                           {NPY_TIMEDELTALTR, NPY_TIMEDELTA},
                                           {NPY_CHARLTR, NPY_CHAR}};

pyBackendWrapper::pyBackendWrapper(PyObject* pBackend,
                                   count_size_type* dtmInfo,
                                   count_size_type* tgtInfo,
                                   int batchSize)
: _dtmInfo(dtmInfo), _tgtInfo(tgtInfo), _batchSize(batchSize), _pBackend(pBackend)
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
    Py_XDECREF(dItem);
    Py_XDECREF(hItem);

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
void pyBackendWrapper::call_backend_transfer(BufferPair &outBuf, int bufIdx)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    printf("**Marker %s at %s:%d \n", __PRETTY_FUNCTION__, __FILE__, __LINE__);
    wrap_buffer_pool(_host_dlist, outBuf.first, bufIdx, _dtmInfo);
    wrap_buffer_pool(_host_tlist, outBuf.second, bufIdx, _tgtInfo);

    PyObject* dArgs  = Py_BuildValue("iOO", bufIdx, _host_dlist, _dev_dlist);

    if (dArgs == NULL) {
        throw std::runtime_error("Unable to build args");
    }
    PyObject* dRes = PyObject_CallObject(_f_consume, dArgs);
    Py_XDECREF(dArgs);
    Py_XDECREF(dRes);

    PyObject* tArgs  = Py_BuildValue("iOO", bufIdx, _host_tlist, _dev_tlist);
    PyObject* tRes = PyObject_CallObject(_f_consume, tArgs);
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

void pyBackendWrapper::wrap_buffer_pool(PyObject *list, Buffer *buf, int bufIdx,
                                        count_size_type *typeInfo)
{
    PyObject *hdItem = PyList_GetItem(list, bufIdx);
    printf("**Marker %s at %s:%d \n", __PRETTY_FUNCTION__, __FILE__, __LINE__);

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

    if (PyList_SetItem(list, bufIdx, p_array) != 0) {
        throw std::runtime_error("Unable to add python array to list");
    }
}

