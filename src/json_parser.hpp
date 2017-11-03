/*
 Copyright 2017 Intel(R) Nervana(TM)
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

/* This class handles python dictionary conversion to nlohman::json object.
 */
class JsonParser
{
private:
    enum Type
    {
        Bool,
        Int,
        Float,
        List,
        Tuple,
        Dict,
        String,
        None
    };

    /* This function handles py2 and py3 independent unpacking of string object 
     * (bytes or unicode) as an ascii std::string
     */
    std::string py23_string_to_ascii_string(PyObject* py_str)
    {
        PyObject*         s = NULL;
        std::stringstream ss;

        if (PyUnicode_Check(py_str))
        {
            s = PyUnicode_AsUTF8String(py_str);
        }
        else if (PyBytes_Check(py_str))
        {
            s = PyObject_Bytes(py_str);
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError, "Unexpected key type");
        }

        if (s != NULL)
        {
            ss << PyBytes_AsString(s);
            Py_XDECREF(s);
        }

        return ss.str();
    }

    Type getType(PyObject* object)
    {
        if (PyBool_Check(object))
            return Type::Bool;
        else if (PyFloat_Check(object))
            return Type::Float;
        else if (PyNumber_Check(object))
            return Type::Int;
        else if (PyList_Check(object))
            return Type::List;
        else if (PyTuple_Check(object))
            return Type::Tuple;
        else if (PyDict_Check(object))
            return Type::Dict;
        else if (PySequence_Check(object))
            return Type::String;
        else if (object == Py_None)
            return Type::None;
        else
        {
            std::stringstream ss;
            ss << "Unexpected type in config dictionary:" << std::endl
               << "Value: " << py23_string_to_ascii_string(PyObject_Repr(object)) << std::endl
               << "Type: " << py23_string_to_ascii_string(PyObject_Repr(PyObject_Type(object)))
               << std::endl;
            throw std::invalid_argument(ss.str());
        }
    }

    void push_value(nlohmann::json& arr, PyObject* value)
    {
        switch (getType(value))
        {
        case Type::Bool: arr.push_back((bool)(value == Py_True)); break;
        case Type::Int: arr.push_back((int)PyLong_AsLong(value)); break;
        case Type::Float: arr.push_back((float)PyFloat_AsDouble(value)); break;
        case Type::List: arr.push_back(parse_list(value)); break;
        case Type::Tuple: arr.push_back(parse_tuple(value)); break;
        case Type::Dict:
        {
            auto json = nlohmann::json::object();
            parse_dict(json, value);
            arr.push_back(json);
            break;
        }
        case Type::String: arr.push_back(py23_string_to_ascii_string(value)); break;
        case Type::None: arr.push_back(nullptr); break;
        default:
            throw std::runtime_error("Unexpected return value from recognize function.");
            break;
        }
    }

    nlohmann::json parse_list(PyObject* list)
    {
        nervana::affirm(PyList_Check(list), "Input argument must be list.");
        auto arr = nlohmann::json::array();
        for (Py_ssize_t i = 0; i < PyList_Size(list); ++i)
        {
            PyObject* value = PyList_GetItem(list, i);
            push_value(arr, value);
        }
        return arr;
    }

    nlohmann::json parse_tuple(PyObject* tuple)
    {
        nervana::affirm(PyTuple_Check(tuple), "Input argument must be tuple.");
        auto arr = nlohmann::json::array();
        for (Py_ssize_t i = 0; i < PyTuple_Size(tuple); ++i)
        {
            PyObject* value = PyTuple_GetItem(tuple, i);
            push_value(arr, value);
        }
        return arr;
    }

    void parse_dict(nlohmann::json& json, PyObject* dict)
    {
        nervana::affirm(PyDict_Check(dict), "Input argument must be dictionary.");
        PyObject * key, *value;
        Py_ssize_t pos = 0;

        while (PyDict_Next(dict, &pos, &key, &value))
        {
            std::string ascii_key = py23_string_to_ascii_string(key);

            switch (getType(value))
            {
            case Type::Bool: json[ascii_key]  = (bool)(value == Py_True); break;
            case Type::Int: json[ascii_key]   = (int)PyLong_AsLong(value); break;
            case Type::Float: json[ascii_key] = (float)PyFloat_AsDouble(value); break;
            case Type::List: json[ascii_key]  = parse_list(value); break;
            case Type::Tuple: json[ascii_key] = parse_tuple(value); break;
            case Type::Dict: parse_dict(json[ascii_key], value); break;
            case Type::String: json[ascii_key] = py23_string_to_ascii_string(value); break;
            case Type::None: json[ascii_key]   = nullptr; break;
            }
        }
    }

public:
    nlohmann::json parse(PyObject* dictionary)
    {
        if (!PyDict_Check(dictionary))
            throw std::invalid_argument("JsonParser::parse() can only take dictionary");
        auto json = nlohmann::json::object();
        parse_dict(json, dictionary);
        return json;
    }
};
