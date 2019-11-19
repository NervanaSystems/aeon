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

using namespace std;
using namespace nervana;

extern gen_image image_dataset;
extern string    test_cache_directory;

gen_image image_dataset;
string    test_cache_directory;

static void DeleteDataset()
{
    image_dataset.delete_files();
}

void exit_func(int s)
{
    //    cout << __FILE__ << " " << __LINE__ << "exit function " << s << endl;
    //    exit(-1);
}

extern "C" int main(int argc, char** argv)
{
    std::setlocale(LC_CTYPE, "");

    cout << "OpenCV version : " << CV_VERSION << endl;
    // mock_nds_server server;

    const char*   exclude = "--gtest_filter=-benchmark.*";
    vector<char*> argv_vector;
    argv_vector.push_back(argv[0]);
    argv_vector.push_back((char*)exclude);
    for (int i = 1; i < argc; i++)
    {
        argv_vector.push_back(argv[i]);
    }
    argc++;

    ::testing::InitGoogleMock(&argc, argv_vector.data());
    int rc = RUN_ALL_TESTS();

    return rc;
}
