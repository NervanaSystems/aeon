/*
 Copyright 2015 Nervana Systems Inc.
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
#include "cpio.hpp"
#include "pyLoader.hpp"

extern "C" {

static std::string last_error_message;

extern const char* get_error_message();
extern int error();
extern void* start(const char* loaderConfigString, PyObject* pbackend);
extern PyObject* next(PyLoader* loader, int bufIdx);
extern int reset(PyLoader* loader);
extern int stop(PyLoader* loader);
extern int itemCount(PyLoader* loader);

}
