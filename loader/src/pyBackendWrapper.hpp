#pragma once
#include <map>
#include "etl_interface.hpp"
#include <Python.h>

#include <assert.h>
#include "buffer_in.hpp"
#include "util.hpp"

class pyBackendWrapper {
public:
    pyBackendWrapper(PyObject*,
                     std::shared_ptr<nervana::interface::config>,
                     std::shared_ptr<nervana::interface::config>,
                     int);

    bool use_pinned_memory();
    void call_backend_transfer(BufferArray &outBuf, int bufIdx);
    PyObject* get_dtm_tgt_pair(int bufIdx);

    ~pyBackendWrapper();

    std::shared_ptr<nervana::interface::config>   _dtm_config;
    std::shared_ptr<nervana::interface::config>   _tgt_config;
    int                         _batchSize;

private:
    pyBackendWrapper() {};
    PyObject* initPyList(int length=2);
    void wrap_buffer_pool(PyObject *list, Buffer *buf, int bufIdx,
                          const std::shared_ptr<nervana::interface::config>& config);

    PyObject*                   _pBackend;

    PyObject*                   _host_dlist;
    PyObject*                   _host_tlist;

    PyObject*                   _dev_dlist;
    PyObject*                   _dev_tlist;

    PyObject*                   _f_consume = NULL;
};
