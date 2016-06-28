#pragma once
#include <map>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <assert.h>
#include "buffer.hpp"
#include "util.hpp"

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


class pyBackendWrapper {
public:
    pyBackendWrapper(PyObject*, nervana::count_size_type*, nervana::count_size_type*, int);
    bool use_pinned_memory();
    void call_backend_transfer(BufferPair &outBuf, int bufIdx);
    PyObject* get_dtm_tgt_pair(int bufIdx);

    ~pyBackendWrapper();

    nervana::count_size_type*   _dtmInfo;
    nervana::count_size_type*   _tgtInfo;
    int                         _batchSize;

private:
    pyBackendWrapper() {};
    PyObject* initPyList(int length=2);
    void wrap_buffer_pool(PyObject *list, Buffer *buf, int bufIdx,
                          nervana::count_size_type *typeInfo);

    PyObject*                   _pBackend;

    PyObject*                   _host_dlist;
    PyObject*                   _host_tlist;

    PyObject*                   _dev_dlist;
    PyObject*                   _dev_tlist;

    PyObject*                   _f_consume = NULL;
};
