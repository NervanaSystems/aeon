#pragma once
#include <map>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <assert.h>
#include "buffer.hpp"
#include "util.hpp"

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
