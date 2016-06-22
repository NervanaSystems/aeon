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
                   const char* manifestFilename,
                   const char* rootCacheDir,
                   const char* mediaConfigString,
                   DeviceParams* deviceParams,
                   int miniBatchSize,
                   int subsetPercent,
                   int macroBatchSize,
                   int randomSeed=0,
                   bool shuffleManifest=false,
                   bool shuffleEveryEpoch=false
                   ) {
    static_assert(sizeof(int) == 4, "int is not 4 bytes");
    try {
        // std::cout << manifestFilename << std::endl;
        // std::cout << rootCacheDir << std::endl;
        // std::cout << mediaConfigString << std::endl;
        // std::cout << miniBatchSize << std::endl;
        // std::cout << subsetPercent << std::endl;
        // std::cout << deviceParams << std::endl;
        // std::cout << macroBatchSize << std::endl;
        // std::cout << randomSeed << std::endl;
        // std::cout << shuffleManifest << std::endl;
        // std::cout << shuffleEveryEpoch << std::endl;

        nlohmann::json js = nlohmann::json::parse(mediaConfigString);
        // std::cout << "jsd " << js.dump(4) << std::endl;

        Loader* loader = new Loader(miniBatchSize,
                                    shuffleManifest,
                                    shuffleEveryEpoch,
                                    subsetPercent,
                                    mediaConfigString,
                                    deviceParams,
                                    manifestFilename,
                                    macroBatchSize,
                                    rootCacheDir,
                                    randomSeed);
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
