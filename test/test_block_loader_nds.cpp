/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include <unistd.h>
#include <signal.h>
#include <sys/param.h>
#include <initializer_list>

#include <curl/curl.h>
#include <curl/easy.h>
#include <curl/curlbuild.h>

#include "gtest/gtest.h"

#include "file_util.hpp"
#include "web_server.hpp"
#include "json.hpp"
#include "cpio.hpp"
#include "helpers.hpp"

#define private public
#include "manifest_nds.hpp"

using namespace std;
using namespace nervana;

TEST(DISABLED_curl, test)
{
    network_client client("http://127.0.0.1:5000", "token", 1, 500, 1, 0);

    for (int i = 0; i < 1000; i++)
    {
        auto         url = client.load_block_url(i);
        stringstream ss;
        client.get(url, ss);
        cout << __FILE__ << " " << __LINE__ << " iteration " << i + 1 << " size " << ss.tellp()
             << endl;
    }
}

TEST(block_loader_nds, curl_stream)
{
    size_t block_size          = 16;
    size_t elements_per_record = 2;

    shared_ptr<manifest_nds> client(manifest_nds_builder()
                                        .base_url("http://127.0.0.1:5000")
                                        .token("token")
                                        .collection_id(1)
                                        .block_size(block_size)
                                        .elements_per_record(elements_per_record)
                                        .make_shared());

    stringstream stream;
    client->m_network_client.get("http://127.0.0.1:5000/test_pattern/", stream);

    stringstream expected;
    for (int i = 0; i < 1024; ++i)
    {
        expected << "0123456789abcdef";
    }
    ASSERT_EQ(stream.str(), expected.str());
}

TEST(block_loader_nds, curl_stream_filename)
{
    size_t block_size          = 16;
    size_t elements_per_record = 2;

    nlohmann::json json;
    json["url"]                     = "http://127.0.0.1:5000";
    json["params"]["token"]         = "token";
    json["params"]["collection_id"] = 1;
    json["params"]["tag"]           = "train";

    std::string   tmp_filename = nervana::file_util::tmp_filename();
    std::ofstream ofs(tmp_filename);

    json >> ofs;
    ofs.close();

    manifest_nds client = manifest_nds_builder()
                              .filename(tmp_filename)
                              .block_size(block_size)
                              .elements_per_record(elements_per_record)
                              .create();

    stringstream stream;
    client.m_network_client.get("http://127.0.0.1:5000/test_pattern/", stream);

    stringstream expected;
    for (int i = 0; i < 1024; ++i)
    {
        expected << "0123456789abcdef";
    }
    ASSERT_EQ(stream.str(), expected.str());
}

TEST(block_loader_nds, curl_stream_error)
{
    size_t block_size          = 16;
    size_t elements_per_record = 2;

    shared_ptr<manifest_nds> client(manifest_nds_builder()
                                        .base_url("http://127.0.0.1:5000")
                                        .token("token")
                                        .collection_id(1)
                                        .block_size(block_size)
                                        .elements_per_record(elements_per_record)
                                        .make_shared());

    stringstream stream;
    EXPECT_THROW(client->m_network_client.get("http://127.0.0.1:5000/error", stream),
                 std::runtime_error);
}

TEST(block_loader_nds, record_count)
{
    size_t block_size          = 16;
    size_t elements_per_record = 2;

    shared_ptr<manifest_nds> client(manifest_nds_builder()
                                        .base_url("http://127.0.0.1:5000")
                                        .token("token")
                                        .collection_id(1)
                                        .block_size(block_size)
                                        .elements_per_record(elements_per_record)
                                        .make_shared());

    // 200 and 5 are hard coded in the mock nds server
    ASSERT_EQ(client->record_count(), 200);
    ASSERT_EQ(client->block_count(), 5);
}

TEST(block_loader_nds, cpio)
{
    size_t block_size          = 16;
    size_t elements_per_record = 2;
    size_t block_count         = 3;

    shared_ptr<manifest_nds> client(manifest_nds_builder()
                                        .base_url("http://127.0.0.1:5000")
                                        .token("token")
                                        .collection_id(1)
                                        .block_size(block_size)
                                        .elements_per_record(elements_per_record)
                                        .make_shared());

    size_t record_number = 0;
    for (size_t block_number = 0; block_number < block_count; block_number++)
    {
        encoded_record_list* block = client->load_block(block_number);
        ASSERT_EQ(block_size, block->size());

        for (auto record : *block)
        {
            element_info info0(vector2string(record.element(0)));
            element_info info1(vector2string(record.element(1)));

            ASSERT_EQ(record_number, info0.record_number());
            ASSERT_EQ(record_number, info1.record_number());
            ASSERT_EQ(0, info0.element_number());
            ASSERT_EQ(1, info1.element_number());

            record_number++;
        }
    }
}

TEST(block_loader_nds, cpio_filename)
{
    size_t block_size          = 16;
    size_t elements_per_record = 2;
    size_t block_count         = 3;

    nlohmann::json json;
    json["url"]                     = "http://127.0.0.1:5000";
    json["params"]["token"]         = "token";
    json["params"]["collection_id"] = 1;
    json["params"]["tag"]           = "train";

    std::string   tmp_filename = nervana::file_util::tmp_filename();
    std::ofstream ofs(tmp_filename);

    json >> ofs;
    ofs.close();

    manifest_nds client = manifest_nds_builder()
                              .filename(tmp_filename)
                              .block_size(block_size)
                              .elements_per_record(elements_per_record)
                              .create();

    size_t record_number = 0;
    for (size_t block_number = 0; block_number < block_count; block_number++)
    {
        encoded_record_list* block = client.load_block(block_number);
        ASSERT_EQ(block_size, block->size());

        for (auto record : *block)
        {
            element_info info0(vector2string(record.element(0)));
            element_info info1(vector2string(record.element(1)));

            ASSERT_EQ(record_number, info0.record_number());
            ASSERT_EQ(record_number, info1.record_number());
            ASSERT_EQ(0, info0.element_number());
            ASSERT_EQ(1, info1.element_number());

            record_number++;
        }
    }
}

// TEST(block_loader_nds, multiblock_sequential)
// {

//     start_server();
//     int block_size = 5000;
//     int elements_per_record = 2;
//     auto client = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, block_size, 1, 0);
//     block_iterator_sequential iter(client);

//     for(int block_number=0; block_number<10; block_number++)
//     {
//         buffer_in_array dest(elements_per_record);
//         iter.read(dest);

//         vector<buffer_in*> data_buffer;
//         for (int element_number=0; element_number<elements_per_record; element_number++)
//         {
//             data_buffer.push_back(dest[element_number]);
//             ASSERT_EQ(block_size, data_buffer[element_number]->record_count());
//         }

//         for (int record_number=0; record_number<block_size; record_number++)
//         {
//             for (int element_number=0; element_number<elements_per_record; element_number++)
//             {
//                 const vector<char>& data_actual = data_buffer[element_number]->get_item(record_number);
//                 ASSERT_NE(0, data_actual.size());
//                 string data        = data_actual.data();
//                 auto tokens        = split(data, ':');
//                 int record_actual  = stod(tokens[0]);
//                 ASSERT_EQ(record_number, record_actual);
//                 int element_actual = stod(tokens[1]);
//                 ASSERT_EQ(element_number, element_actual);
//             }
//         }
//     }
// }

// TEST(block_loader_nds, multiblock_shuffled)
// {
//     start_server();
//     int block_size = 5000;
//     int elements_per_record = 2;
//     auto client = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, block_size, 1, 0);
//     block_iterator_shuffled iter(client);

//     for(int block_number=0; block_number<10; block_number++)
//     {
//         buffer_in_array dest(elements_per_record);
//         iter.read(dest);

//         vector<buffer_in*> data_buffer;
//         for (int element_number=0; element_number<elements_per_record; element_number++)
//         {
//             data_buffer.push_back(dest[element_number]);
//             ASSERT_EQ(block_size, data_buffer[element_number]->record_count());
//         }

//         for (int record_number=0; record_number<block_size; record_number++)
//         {
//             for (int element_number=0; element_number<elements_per_record; element_number++)
//             {
//                 const vector<char>& data_actual = data_buffer[element_number]->get_item(record_number);
//                 ASSERT_NE(0, data_actual.size());
//                 string data        = data_actual.data();
//                 auto tokens        = split(data, ':');
// //                int record_actual  = stod(tokens[0]);
// //                ASSERT_EQ(record_number, record_actual);
//                 int element_actual = stod(tokens[1]);
//                 ASSERT_EQ(element_number, element_actual);
//             }
//         }
//     }
// }

// //TEST(block_loader_nds, performance)
// //{
// //    //    generate_large_cpio_file();
// //    string                        cache_dir = file_util::make_temp_directory();
// //    chrono::high_resolution_clock timer;
// //    start_server();
// //    auto                    client   = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, 16, 1, 0);
// //    string                  cache_id = block_loader_random::randomString();
// //    string                  version  = "version123";
// //    auto                    cache    = make_shared<block_loader_cpio_cache>(cache_dir, cache_id, version, client);
// //    block_iterator_shuffled iter(cache);

// //    auto startTime = timer.now();
// //    for (int i = 0; i < 300; i++)
// //    {
// //        buffer_in_array dest(2);
// //        iter.read(dest);
// //    }
// //    auto endTime = timer.now();
// //    cout << "time " << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count() << " ms" << endl;
// //    file_util::remove_directory(cache_dir);
// //}
