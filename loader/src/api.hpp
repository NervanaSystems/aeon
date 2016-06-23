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

#include "batchfile.hpp"

extern "C" {

static std::string last_error_message;

extern const char* get_error_message() {
    return last_error_message.c_str();
}

extern void* start(
                   int* itemCount,
                   int miniBatchSize,
                   const char* loaderConfigString,
                   DeviceParams* deviceParams
                   ) {
    static_assert(sizeof(int) == 4, "int is not 4 bytes");
    try {

        Loader* loader = new Loader(miniBatchSize, loaderConfigString, deviceParams);

        int result = loader->start();
        if (result != 0) {
            std::stringstream ss;
            ss << "Could not start data loader. Error " << result;
            last_error_message = ss.str();
            delete loader;
            return 0;
        }
        *itemCount = loader->itemCount();
        return reinterpret_cast<void*>(loader);
    } catch(std::exception& ex) {
        last_error_message = ex.what();
        return 0;
    }
}

extern int error() {
    try {
        throw std::runtime_error("abc error");
    } catch(std::exception& ex) {
        last_error_message = ex.what();
        return -1;
    }
}

extern int next(Loader* loader) {
    try {
        loader->next();
        return 0;
    } catch(std::exception& ex) {
        last_error_message = ex.what();
        return -1;
    }
}

extern int reset(Loader* loader) {
    try {
        return loader->reset();
    } catch(std::exception& ex) {
        last_error_message = ex.what();
        return -1;
    }
}

extern int stop(Loader* loader) {
    try {
        loader->stop();
        delete loader;
        return 0;
    } catch(std::exception& ex) {
        last_error_message = ex.what();
        return -1;
    }
}

}
