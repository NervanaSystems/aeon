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

#include <iostream>
#include <chrono>

#include "gtest/gtest.h"

#include "gen_image.hpp"
#include "file_util.hpp"
#include "web_server.hpp"
#include "web_app.hpp"

using namespace std;

gen_image image_dataset;
string test_cache_directory;

static void CreateImageDataset()
{
//    std::chrono::high_resolution_clock timer;
//    auto start = timer.now();
    image_dataset.Directory("image_data")
            .Prefix("archive-")
            .MacrobatchMaxItems(500)
            // SetSize must be a multiple of (minibatchCount*batchSize) which is 8320 currently
            .DatasetSize(1500)
            .ImageSize(128,128)
            .Create();
//    auto end = timer.now();
//    cout << "image dataset " << (chrono::duration_cast<chrono::milliseconds>(end - start)).count() << " msec" << endl;
}

static void DeleteDataset() {
    image_dataset.Delete();
}

void exit_func(int s)
{
//    cout << __FILE__ << " " << __LINE__ << "exit function " << s << endl;
//    exit(-1);
}

void page_handler(web::page& page, const std::string& url)
{
    page.PageOK();
    page.SendString("<html>Hello World</html>");
}

void web_server()
{
    web::server ws;
    ws.RegisterPageHandler(page_handler);
    ws.Start(8086);
    sleep(10);
    ws.Stop();
}

extern "C" int main( int argc, char** argv )
{
//    struct sigaction sigIntHandler;
//    sigIntHandler.sa_handler = exit_func;
//    sigemptyset(&sigIntHandler.sa_mask);
//    sigIntHandler.sa_flags = 0;
//    sigaction(SIGINT, &sigIntHandler, NULL);


//    web_app app;

    web_server();

//    CreateImageDataset();
//    test_cache_directory = nervana::file_util::make_temp_directory();

//    ::testing::InitGoogleTest(&argc, argv);
//    int rc = RUN_ALL_TESTS();

//    nervana::file_util::remove_directory(test_cache_directory);
//    DeleteDataset();

//    return rc;
}
