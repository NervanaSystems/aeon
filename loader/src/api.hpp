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
#include "cpio.hpp"
#include "loader.hpp"

extern "C" {
static std::string last_error_message;

extern const char* get_error_message();
extern int         error();
extern void* start(const char* loaderConfigString, PyObject* pbackend);
extern PyObject* next(nervana::loader* data_loader, int bufIdx);
extern int reset(nervana::loader* data_loader);
extern int stop(nervana::loader* data_loader);
extern int itemCount(nervana::loader* data_loader);
extern PyObject* shapes(nervana::loader* data_loader);
}
