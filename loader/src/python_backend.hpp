#pragma once
#include <map>
#include "interface.hpp"
#include <Python.h>

#include <assert.h>
#include "buffer_in.hpp"
#include "util.hpp"
#include "buffer_out.hpp"

class python_backend {
public:
    python_backend(PyObject*, const std::vector<nervana::shape_type>&, int);
    ~python_backend();

    bool use_pinned_memory();
    void call_backend_transfer(nervana::buffer_out_array &outBuf, int bufIdx);
    PyObject* get_host_tuple(int bufIdx);
    PyObject* get_shapes();
    const std::vector<nervana::shape_type>& _oshape_types;
    int                         _batchSize;
private:
    python_backend() = delete;
    PyObject* initPyList(int length=2);
    void wrap_buffer_pool(PyObject *list, nervana::buffer_out *buf, int bufIdx,
                          const nervana::shape_type& shape_type);

    PyObject*                   _py_obj_backend;

    std::vector<PyObject*>      _host_lists;
    std::vector<PyObject*>      _dev_lists;

    PyObject*                   _f_consume = NULL;
};
