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

#include "api.hpp"

using namespace nervana;

extern "C" {

extern const char* get_error_message()
{
    return last_error_message.c_str();
}

extern void* start(const char* loaderConfigString, PyObject* pbackend)
{
    static_assert(sizeof(int) == 4, "int is not 4 bytes");
    try
    {
        loader* data_loader = new loader(loaderConfigString, pbackend);

        int result = data_loader->start();
        if (result != 0)
        {
            std::stringstream ss;
            ss << "Could not start data loader. Error " << result;
            last_error_message = ss.str();
            delete data_loader;

            return 0;
        }
        return reinterpret_cast<void*>(data_loader);
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();

        return 0;
    }
}

extern int error()
{
    try
    {
        throw std::runtime_error("abc error");
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();
        return -1;
    }
}

extern PyObject* next(loader* data_loader, int bufIdx)
{
    try
    {
        return data_loader->next(bufIdx);
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();

        Py_INCREF(Py_None);
        return Py_None;
    }
}

extern PyObject* shapes(loader* data_loader)
{
    try
    {
        return data_loader->shapes();
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();

        Py_INCREF(Py_None);
        return Py_None;
    }
}

extern int itemCount(loader* data_loader)
{
    try
    {
        return data_loader->itemCount();
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();
        return -1;
    }
}

extern int reset(loader* data_loader)
{
    try
    {
        return data_loader->reset();
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();
        return -1;
    }
}

extern int stop(loader* data_loader)
{
    try
    {
        data_loader->stop();
        delete data_loader;
        return 0;
    }
    catch (std::exception& ex)
    {
        last_error_message = ex.what();
        return -1;
    }
}
}
