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

#include <unistd.h>
#include <signal.h>
#include <sys/param.h>
#include <initializer_list>

#include <curl/curl.h>
#include <curl/easy.h>
#include <curl/curlbuild.h>

#include "gtest/gtest.h"

// cringe
#define private public
#include "manifest_nds.hpp"
#include "file_util.hpp"
#include "web_server.hpp"
#include "json.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

class curl_client
{
public:
    curl_client(const std::string& baseurl, const std::string& token, int collection_id, uint32_t block_size,
                                       int shard_count, int shard_index)
        : m_baseurl(baseurl)
        , m_token(token)
        , m_collection_id(collection_id)
        , m_shard_count(shard_count)
        , m_shard_index(shard_index)
        , m_macrobatch_size(block_size)
    {
        curl_global_init(CURL_GLOBAL_ALL);
    }

    ~curl_client()
    {
        curl_global_cleanup();
    }

    static size_t callback(void* ptr, size_t size, size_t nmemb, void* stream)
    {
        stringstream& ss = *(stringstream*)stream;
        // callback used by curl.  writes data from ptr into the
        // stringstream passed in to `stream`.

        ss.write((const char*)ptr, size * nmemb);
        return size * nmemb;
    }

    void get(const string& url, stringstream& stream)
    {
        // reuse curl connection across requests
        void* m_curl = curl_easy_init();

        // given a url, make an HTTP GET request and fill stream with
        // the body of the response

        curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, callback);
        curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &stream);

        // Perform the request, res will get the return code
        CURLcode res = curl_easy_perform(m_curl);

        // Check for errors
        long http_code = 0;
        curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code != 200 || res != CURLE_OK)
        {
            stringstream ss;
            ss << "HTTP GET on \n'" << url << "' failed. ";
            ss << "status code: " << http_code;
            if (res != CURLE_OK)
            {
                ss << " curl return: " << curl_easy_strerror(res);
            }

            curl_easy_cleanup(m_curl);
            throw std::runtime_error(ss.str());
        }

        curl_easy_cleanup(m_curl);
    }

    string load_block_url(uint32_t block_num)
    {
        stringstream ss;
        ss << m_baseurl << "/macrobatch/?";
        ss << "macro_batch_index=" << block_num;
        ss << "&macro_batch_max_size=" << m_macrobatch_size;
        ss << "&collection_id=" << m_collection_id;
        ss << "&shard_count=" << m_shard_count;
        ss << "&shard_index=" << m_shard_index;
        ss << "&token=" << m_token;
        return ss.str();
    }

    string metadata_url()
    {
        stringstream ss;
        ss << m_baseurl << "/object_count/?";
        ss << "macro_batch_max_size=" << m_macrobatch_size;
        ss << "&collection_id=" << m_collection_id;
        ss << "&shard_count=" << m_shard_count;
        ss << "&shard_index=" << m_shard_index;
        ss << "&token=" << m_token;
        return ss.str();
    }

private:
    const std::string m_baseurl;
    const std::string m_token;
    const int         m_collection_id;
    const int         m_shard_count;
    const int         m_shard_index;
    unsigned int      m_object_count;
    unsigned int      m_block_count;
    uint32_t          m_macrobatch_size;
};

TEST(DISABLED_curl,test)
{
    curl_client client("http://127.0.0.1:5000", "token", 1, 500, 1, 0);

    for (int i=0; i<1000; i++)
    {
        auto url = client.load_block_url(i);
        stringstream ss;
        client.get(url, ss);
        cout << __FILE__ << " " << __LINE__ << " iteration " << i+1 << " size " << ss.tellp() << endl;
    }
}

TEST(block_loader_nds, curl_stream)
{
    manifest_nds client = manifest_nds_builder().base_url("http://127.0.0.1:5000")
            .token("token").collection_id(1).block_size(16).elements_per_record(2).create();

    stringstream stream;
    client.get("http://127.0.0.1:5000/test_pattern/", stream);

    stringstream expected;
    for (int i = 0; i < 1024; ++i)
    {
        expected << "0123456789abcdef";
    }
    ASSERT_EQ(stream.str(), expected.str());
}

TEST(block_loader_nds, curl_stream_error)
{
    manifest_nds client = manifest_nds_builder().base_url("http://127.0.0.1:5000")
            .token("token").collection_id(1).block_size(16).elements_per_record(2).create();

    stringstream stream;
    EXPECT_THROW(client.get("http://127.0.0.1:5000/error", stream), std::runtime_error);
}

TEST(block_loader_nds, record_count)
{
    manifest_nds client = manifest_nds_builder().base_url("http://127.0.0.1:5000")
            .token("token").collection_id(1).block_size(16).elements_per_record(2).create();

    // 200 and 5 are hard coded in the mock nds server
    ASSERT_EQ(client.record_count(), 200);
    ASSERT_EQ(client.block_count(), 5);
}

TEST(block_loader_nds, cpio)
{
    manifest_nds client = manifest_nds_builder().base_url("http://127.0.0.1:5000")
            .token("token").collection_id(1).block_size(16).elements_per_record(2).create();

//     buffer_in_array dest(2);
//     ASSERT_EQ(dest.size(), 2);
//     ASSERT_EQ(dest[0]->record_count(), 0);

    encoded_record_list block = client.load_block(0);

//     ASSERT_EQ(dest[0]->record_count(), 2);
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
