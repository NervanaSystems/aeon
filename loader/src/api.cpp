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

#include <Python.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

#include "json.hpp"
#include "api.hpp"
#include "loader.hpp"

using namespace nervana;
using namespace std;

extern "C" {

typedef struct
{
    PyObject_HEAD;
    long int            m;
    long int            i;
    shared_ptr<loader>  m_loader;

    // I don't know why we need padding but it crashes without it.
    // TODO: figure out what is going on.
    uint64_t            m_padding[10];
} aeon_AeonDataloader;

PyObject* aeon_AeonDataloader_iter(PyObject* self)
{
    std::cout << __FILE__ << " " << __LINE__ << " aeon_AeonDataloader_iter" << std::endl;
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    Py_INCREF(self);
    return self;
}

PyObject* aeon_AeonDataloader_iternext(PyObject* self)
{
    std::cout << __FILE__ << " " << __LINE__ << " aeon_AeonDataloader_iternext" << std::endl;
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    if (p->i < p->m)
    {
        PyObject* tmp = Py_BuildValue("l", p->i);
        (p->i)++;
        return tmp;
    }
    else
    {
        /* Raising of standard StopIteration exception with empty value. */
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
}

static Py_ssize_t aeon_AeonDataloader_length(PyObject* self)
{
    std::cout << __FILE__ << " " << __LINE__ << " aeon_AeonDataloader_length" << std::endl;
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    return p->m_loader->itemCount();
}

static PyTypeObject aeon_AeonDataloaderType = {
    PyObject_HEAD_INIT(NULL) 0, /*ob_size*/
    "aeon.AeonDataloader",      /*tp_name*/
    sizeof(aeon_AeonDataloader) /*tp_basicsize*/
};

static PySequenceMethods aeon_AeonDataloader_sequence_methods = {
    aeon_AeonDataloader_length /* sq_length */
};

static PyMethodDef module_methods[] = {
    {NULL} /* Sentinel */
};

static std::string dictionary_to_string(PyObject* dict)
{
    PyObject*         _key;
    PyObject*         _value;
    Py_ssize_t        pos = 0;
    std::stringstream ss;

    ss << "{";
    bool first = true;
    while (PyDict_Next(dict, &pos, &_key, &_value))
    {
        /* do something interesting with the values... */
        if (first)
        {
            first = false;
        }
        else
        {
            ss << ", ";
        }
        std::string key = PyString_AsString(_key);
        std::string value;
        if (PyDict_Check(_value))
        {
            value = dictionary_to_string(_value);
        }
        else
        {
            if (PyString_Check(_value))
            {
                value = PyString_AsString(_value);
                value = "\"" + value + "\"";
            }
            else if (PyBool_Check(_value))
            {
                value = (PyObject_IsTrue(_value) ? "true" : "false");
            }
            else
            {
                PyObject* objectsRepresentation = PyObject_Repr(_value);
                value                           = PyString_AsString(objectsRepresentation);
            }
        }
        ss << "\"" << key << "\": " << value;
    }
    ss << "}";
    return ss.str();
}

static PyObject* AeonDataloader_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    std::cout << __FILE__ << " " << __LINE__ << " AeonDataloader_new" << std::endl;
    long int             m;
    aeon_AeonDataloader* p;

    PyObject* dict;
    PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict);

    std::string    dict_string = dictionary_to_string(dict);
    nlohmann::json json_config = nlohmann::json::parse(dict_string);

    /* I don't need python callable __init__() method for this iterator,
       so I'll simply allocate it as PyObject and initialize it by hand. */

    p = PyObject_New(aeon_AeonDataloader, &aeon_AeonDataloaderType);
    if (!p)
    {
        return NULL;
    }

    /* I'm not sure if it's strictly necessary. */
    if (!PyObject_Init((PyObject*)p, &aeon_AeonDataloaderType))
    {
        Py_DECREF(p);
        return NULL;
    }

    p->m = 5;
    p->i = 0;
    std::cout << __FILE__ << " " << __LINE__ << std::endl;
    p->m_loader = make_shared<loader>(json_config.dump());
    p->m_loader = shared_ptr<loader>(new loader(json_config.dump()));
    std::cout << __FILE__ << " " << __LINE__ << std::endl;
    return (PyObject*)p;
}

static PyObject* aeon_shapes(PyObject* self, PyObject*)
{
    std::cout << __FILE__ << " " << __LINE__ << " aeon_shapes" << std::endl;
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    //    uint32_t num_shapes = _oshape_types.size();
    uint32_t  num_shapes = 2;
    PyObject* all_shapes = PyTuple_New(num_shapes);
    for (uint32_t idx = 0; idx < num_shapes; ++idx)
    {
        //        auto shapevec = _oshape_types[idx].get_shape();
        std::vector<int> shapevec   = {2, 3};
        PyObject*        this_shape = PyTuple_New(shapevec.size());
        for (uint32_t dim = 0; dim < shapevec.size(); dim++)
        {
            PyTuple_SetItem(this_shape, dim, Py_BuildValue("i", shapevec[dim]));
        }
        PyTuple_SetItem(all_shapes, idx, this_shape);
    }

    return all_shapes;
}

static PyObject* aeon_get_buffer_names(PyObject* self, PyObject*)
{
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    std::cout << __FILE__ << " " << __LINE__ << " aeon_get_buffer_names" << std::endl;
    p->i = 0;
    return Py_None;
}

static PyObject* aeon_get_buffer_shape(PyObject* self, PyObject* args)
{
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    std::cout << __FILE__ << " " << __LINE__ << " aeon_get_buffer_shape" << std::endl;
    char* s;
    PyArg_ParseTuple(args, "s", &s);
    std::string name = s;
    std::cout << __FILE__ << " " << __LINE__ << " buffer name " << name << std::endl;
    p->i = 0;
    return Py_None;
}

static PyObject* aeon_reset(PyObject* self, PyObject*)
{
    aeon_AeonDataloader* p = (aeon_AeonDataloader*)self;
    std::cout << __FILE__ << " " << __LINE__ << " aeon_reset" << std::endl;
    p->i = 0;
    return Py_None;
}

static PyMethodDef AeonMethods[] = {
    //    {"AeonDataloader",  aeon_myiter, METH_VARARGS, "Iterate from i=0 while i<m."},
    {"shapes", aeon_shapes, METH_NOARGS, "Get output shapes"},
    {"get_buffer_names", aeon_get_buffer_names, METH_NOARGS, "Get output buffer names"},
    {"get_buffer_shape", aeon_get_buffer_shape, METH_VARARGS, "Get shape of named buffer"},
    {"reset", aeon_reset, METH_NOARGS, "Reset iterator"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC initaeon(void)
{
    std::cout << __FILE__ << " " << __LINE__ << " initaeon" << std::endl;

    PyObject* m;

    aeon_AeonDataloaderType.tp_new         = &AeonDataloader_new;
    aeon_AeonDataloaderType.tp_as_sequence = &aeon_AeonDataloader_sequence_methods;
    aeon_AeonDataloaderType.tp_methods     = AeonMethods;
    aeon_AeonDataloaderType.tp_flags       = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER;
    aeon_AeonDataloaderType.tp_doc         = "Internal myiter iterator object.";
    aeon_AeonDataloaderType.tp_iter        = aeon_AeonDataloader_iter;     // __iter__() method
    aeon_AeonDataloaderType.tp_iternext    = aeon_AeonDataloader_iternext; // next() method

    if (PyType_Ready(&aeon_AeonDataloaderType) < 0)
    {
        return;
    }

    m = Py_InitModule("aeon", module_methods);

    Py_INCREF(&aeon_AeonDataloaderType);
    PyModule_AddObject(m, "AeonDataloader", (PyObject*)&aeon_AeonDataloaderType);
}
}
