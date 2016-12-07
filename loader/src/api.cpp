// /*
//  Copyright 2016 Nervana Systems Inc.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//       http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// */

#include <iostream>
#include <sstream>
// #include <memory>

#include "api.hpp"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace nervana;
using namespace std;


extern "C" {
#define DL_get_loader(v) (((aeon_Dataloader*)(v))->m_loader)
#define DL_get_i(v) (((aeon_Dataloader*)(v))->m_i)


static PyObject* wrap_buffer_as_np_array(const buffer_fixed_size_elements* buf);


typedef struct
{
    PyObject_HEAD;
    loader* m_loader;
    uint32_t m_i;
    // I don't know why we need padding but it crashes without it.
    // TODO: figure out what is going on.
    uint64_t m_padding[10];
} aeon_Dataloader;

static PyObject* Dataloader_iter(PyObject* self)
{
    INFO << " aeon_Dataloader_iter";
    Py_INCREF(self);
    return self;
}

static PyObject* Dataloader_iternext(PyObject* self)
{
    INFO << " aeon_Dataloader_iternext";
    PyObject* result = NULL;
    if (DL_get_loader(self)->get_current_iter() != DL_get_loader(self)->get_end_iter())
    {
        // d will be const fixed_buffer_map&
        const fixed_buffer_map& d = *(DL_get_loader(self)->get_current_iter());
        auto names = DL_get_loader(self)->get_buffer_names();

        result = PyDict_New();

        for (auto&& nm : names)
        {
            PyObject* wrapped_buf = wrap_buffer_as_np_array(d[nm]);
            if (PyDict_SetItemString(result, nm.c_str(), wrapped_buf) < 0)
            {
                ERR << "Error building shape string";
                PyErr_SetString(PyExc_RuntimeError, "Error building shape dict");
            }
        }
        DL_get_loader(self)->get_current_iter()++;
    }
    else
    {
        /* Raising of standard StopIteration exception with empty value. */
        PyErr_SetNone(PyExc_StopIteration);
    }

    return result;
}

static Py_ssize_t aeon_Dataloader_length(PyObject* self)
{
    INFO << " aeon_Dataloader_length " << DL_get_loader(self)->item_count();
    return DL_get_loader(self)->item_count();
}

static PySequenceMethods Dataloader_sequence_methods = {
    aeon_Dataloader_length /* sq_length */
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

static PyObject* wrap_buffer_as_np_array(const buffer_fixed_size_elements* buf)
{
    std::vector<npy_intp> dims;
    dims.push_back(buf->get_item_count());
    auto shape = buf->get_shape_type().get_shape();
    dims.insert(dims.end(), shape.begin(), shape.end());

    int nptype = buf->get_shape_type().get_otype().get_np_type();

    PyObject* p_array = PyArray_SimpleNewFromData(dims.size(), &dims[0], nptype, const_cast<char *>(buf->data()));

    if (p_array == NULL)
    {
        ERR << "Unable to wrap buffer as npy array";
        PyErr_SetString(PyExc_RuntimeError, "Unable to wrap buffer as npy array");
    }

    return p_array;
}

static void
Dataloader_dealloc(aeon_Dataloader* self)
{
    INFO << " Dataloader_dealloc";
    if (self->m_loader != nullptr)
    {
        delete self->m_loader;
    }
    self->ob_type->tp_free((PyObject*)self);
}

static PyObject*
Dataloader_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    INFO << " Dataloader_new";
    aeon_Dataloader* self = nullptr;

    static const char* keyword_list[] = {"config", nullptr};

    PyObject* dict;
    auto rc = PyArg_ParseTupleAndKeywords(args, kwds, "O", const_cast<char **>(keyword_list), &dict);

    if (rc)
    {
        std::string    dict_string = dictionary_to_string(dict);
        nlohmann::json json_config = nlohmann::json::parse(dict_string);

        INFO << " config " << json_config.dump(4);

        self = (aeon_Dataloader*)type->tp_alloc(type, 0);
        if (!self)
        {
            return NULL;
        }

        self->m_loader = new loader(json_config);
        self->m_i = 0;
    }

    return (PyObject*) self;
}

static int
Dataloader_init(aeon_Dataloader *self, PyObject *args, PyObject *kwds)
{
    return 0;
}

static PyObject* aeon_shapes(PyObject* self, PyObject *)
{
    INFO << " aeon_shapes";

    auto name_shape_list = DL_get_loader(self)->get_names_and_shapes();

    PyObject* py_shape_dict = PyDict_New();

    if (!py_shape_dict)
    {
        PyErr_SetNone(PyExc_RuntimeError);
    }
    else
    {
        for (auto&& name_shape : name_shape_list)
        {
            auto name = name_shape.first;
            auto shape = name_shape.second.get_shape();

            PyObject* py_shape_tuple = PyTuple_New(shape.size());

            if (!py_shape_tuple)
            {
                ERR << "Error building shape string";
                PyErr_SetString(PyExc_RuntimeError, "Error building shape");
            }
            else
            {
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    PyTuple_SetItem(py_shape_tuple, i, Py_BuildValue("i", shape[i]));
                }
            }

            if (PyDict_SetItemString(py_shape_dict, name.c_str(), py_shape_tuple) < 0)
            {
                ERR << "Error building shape string";
                PyErr_SetString(PyExc_RuntimeError, "Error building shape dict");
            }
        }
    }
    return py_shape_dict;
}

static PyObject* aeon_reset(PyObject* self, PyObject*)
{
    INFO << " aeon_reset";
    DL_get_i(self) = 0;
    return Py_None;
}

static PyMethodDef Dataloader_methods[] = {
    //    {"Dataloader",  aeon_myiter, METH_VARARGS, "Iterate from i=0 while i<m."},
    {"shapes", aeon_shapes, METH_NOARGS, "Get output shapes"},
    {"reset", aeon_reset, METH_NOARGS, "Reset iterator"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};


static PyTypeObject aeon_DataloaderType = {
    PyObject_HEAD_INIT(NULL)
    0, /*ob_size*/
    "aeon.Dataloader",       /*tp_name*/
    sizeof(aeon_Dataloader), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Dataloader_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    &Dataloader_sequence_methods,/*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_ITER | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Internal myiter iterator object.",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    Dataloader_iter,           /* tp_iter */
    Dataloader_iternext,       /* tp_iternext */
    Dataloader_methods,        /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Dataloader_init, /* tp_init */
    0,                         /* tp_alloc */
    Dataloader_new,            /* tp_new */
};

PyMODINIT_FUNC initaeon(void)
{
    INFO << " initaeon";

    PyObject* m;

    if (PyType_Ready(&aeon_DataloaderType) < 0)
    {
        return;
    }

    if (_import_array() < 0)
    {
        return;
    }

    m = Py_InitModule3("aeon", module_methods, "Dataloader containing module");

    Py_INCREF(&aeon_DataloaderType);
    PyModule_AddObject(m, "Dataloader", (PyObject*)&aeon_DataloaderType);
}
}
