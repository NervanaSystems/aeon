/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <chrono>
#include <clocale>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "gen_image.hpp"
#include "file_util.hpp"
#include "json.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

gen_image image_dataset;
string    test_cache_directory;

static void CreateImageDataset()
{
    //    std::chrono::high_resolution_clock timer;
    //    auto start = timer.now();
    image_dataset.directory("image_data")
        .prefix("archive-")
        .macrobatch_max_records(500)
        // SetSize must be a multiple of (minibatchCount*batchSize) which is 8320 currently
        .dataset_size(1500)
        .ImageSize(128, 128)
        .create();
    //    auto end = timer.now();
    //    cout << "image dataset " << (chrono::duration_cast<chrono::milliseconds>(end - start)).count() << " msec" << endl;
}

static void DeleteDataset()
{
    image_dataset.delete_files();
}

extern "C" int main(int argc, char** argv)
{
    std::setlocale(LC_CTYPE, "");

    cout << "OpenCV version : " << CV_VERSION << endl;

    const char*   exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back((char*)exclude);
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc++;

    CreateImageDataset();
    test_cache_directory = nervana::file_util::make_temp_directory();

    ::testing::InitGoogleMock(&argc, argv_vector.data());
    int rc = RUN_ALL_TESTS();

    nervana::file_util::remove_directory(test_cache_directory);
    DeleteDataset();

    return rc;
}
