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

#pragma once

#include <map>
#include "interface.hpp"
#include <Python.h>

#include "loader.hpp"
#include "buffer_batch.hpp"
#include "util.hpp"

namespace nervana
{
    class python_backend;
    class gil_state;
}

class nervana::gil_state
{
public:
    gil_state()
        : m_gstate{PyGILState_Ensure()}
    {
    }
    ~gil_state() { PyGILState_Release(m_gstate); }
private:
    PyGILState_STATE m_gstate;
};

class nervana::python_backend : public nervana::async_manager<nervana::fixed_buffer_map, std::vector<PyObject*>>
{
public:
    python_backend(loader_async* loader_source, PyObject* py_obj_backend,
                   const std::vector<nervana::shape_type>& oshape_types);
    ~python_backend();
    void setup_buffers();

    virtual std::vector<PyObject*>* filler() override;


private:
    python_backend() = delete;
    PyObject* wrap_buffer_as_np_array(buffer_fixed_size_elements buf, const nervana::shape_type& st);

    PyObject* m_py_obj_backend;
    std::vector<nervana::shape_type> m_oshape_types;
    PyObject* m_f_consume = NULL;
};
