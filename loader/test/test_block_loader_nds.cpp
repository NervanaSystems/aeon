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
#include "block_loader_nds.hpp"
#include "block_iterator_shuffled.hpp"
#include "block_iterator_sequential.hpp"
#include "block_loader_cpio_cache.hpp"
#include "block_loader_util.hpp"
#include "file_util.hpp"
#include "web_server.hpp"
#include "json.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

// NDSMockServer starts a python process in the constructor and kills the
// process in the destructor
class NDSMockServer
{
public:
//    NDSMockServer()
//    {
//        cout << "starting mock nds server ..." << endl;
//        pid_t pid = fork();
//        if(pid == 0) {
//            int i = system("../test/start_nds_server");
//            if(i) {
//                cout << "error starting nds_server: " << strerror(i) << endl;
//            }
//            exit(1);
//        }

//        // sleep for 3 seconds to let the nds_server come up
//        usleep(3 * 1000 * 1000);
//        _pid = pid;
//    }

//    ~NDSMockServer()
//    {
//        cout << "killing mock nds server ..." << endl;
//        // kill the python process running the mock NDS
//        stringstream stream;
//        block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);
//        client.get("http://127.0.0.1:5000/shutdown/", stream);
//        //  kill(_pid, 15);
//    }


    NDSMockServer()
    {
        page_request_handler fn = bind(&NDSMockServer::page_handler, this, placeholders::_1, placeholders::_2);
        m_server.register_page_handler(fn);
        m_server.start(5000);
    }

    ~NDSMockServer()
    {
        m_server.stop();
    }

    void set_elements_per_record(initializer_list<int> init)
    {
        m_elements_per_record = init;
    }

    void page_handler(web::page& page, const std::string& url)
    {
        if (url == "/object_count/")
        {
            nlohmann::json js = {
                {"record_count", 200},
                {"macro_batch_per_shard", 5}
            };
            string rc = js.dump();
            page.send_string(rc);
        }
        else if (url == "/macrobatch/")
        {
            map<string,string> args = page.args();
            int macro_batch_max_size = stod(args["macro_batch_max_size"]);
//            int macro_batch_index = stod(args["macro_batch_index"]);
//            int collection_id = stod(args["collection_id"]);
//            string token = args["token"];
            stringstream ss;
            {
                cpio::writer writer(ss);
                for (int record_number=0; record_number<macro_batch_max_size; record_number++)
                {
                    buffer_in_array bin{(uint32_t)m_elements_per_record.size()};
                    for (int element_number=0; element_number<m_elements_per_record.size(); element_number++)
                    {
                        vector<char> data(m_elements_per_record[element_number]);
                        stringstream ss;
                        ss << record_number << ":" << element_number;
                        string id = ss.str();
                        id.copy(data.data(), id.size());
                        data[id.size()] = 0;
                        bin[element_number]->add_item(data);
                    }
                    writer.write_all_records(bin);
                }
            }

            string cpio_data = ss.str();
            page.send_as_file(cpio_data.data(), cpio_data.size());
        }
        else if (url == "/test_pattern/")
        {
            for (int i=0; i<1024; i++)
            {
                page.send_string("0123456789abcdef");
            }
        }
        else if (url == "/error")
        {
            page.page_not_found();
        }
    }

private:
    web::server m_server;
    vector<int> m_elements_per_record = {1024, 8};
};

std::shared_ptr<NDSMockServer> mock_server;

size_t curl_client_callback(void* ptr, size_t size, size_t nmemb, void* stream)
{
    stringstream& ss = *(stringstream*)stream;
    // callback used by curl.  writes data from ptr into the
    // stringstream passed in to `stream`.

    ss.write((const char*)ptr, size * nmemb);
    return size * nmemb;
}

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

    void get(const string& url, stringstream& stream)
    {
        // reuse curl connection across requests
        void* m_curl = curl_easy_init();

        // given a url, make an HTTP GET request and fill stream with
        // the body of the response

        curl_easy_setopt(m_curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, curl_client_callback);
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

static void start_server()
{
    if (mock_server == nullptr)
    {
        mock_server = make_shared<NDSMockServer>();
    }
}

TEST(DISABLED_curl,test)
{
    start_server();
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
    start_server();
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

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
    start_server();
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    stringstream stream;
    EXPECT_THROW(client.get("http://127.0.0.1:5000/error", stream), std::runtime_error);
}

TEST(block_loader_nds, record_count)
{
    start_server();
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    // 200 and 5 are hard coded in the mock nds server
    ASSERT_EQ(client.record_count(), 200);
    ASSERT_EQ(client.block_count(), 5);
}

TEST(block_loader_nds, cpio)
{
    start_server();
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    buffer_in_array dest(2);
    ASSERT_EQ(dest.size(), 2);
    ASSERT_EQ(dest[0]->record_count(), 0);

    client.load_block(dest, 0);

//    ASSERT_EQ(dest[0]->record_count(), 2);
}

TEST(block_loader_nds, multiblock_sequential)
{

    start_server();
    int block_size = 5000;
    int elements_per_record = 2;
    auto client = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, block_size, 1, 0);
    block_iterator_sequential iter(client);

    for(int block_number=0; block_number<10; block_number++)
    {
        buffer_in_array dest(elements_per_record);
        iter.read(dest);

        vector<buffer_in*> data_buffer;
        for (int element_number=0; element_number<elements_per_record; element_number++)
        {
            data_buffer.push_back(dest[element_number]);
            ASSERT_EQ(block_size, data_buffer[element_number]->record_count());
        }

        for (int record_number=0; record_number<block_size; record_number++)
        {
            for (int element_number=0; element_number<elements_per_record; element_number++)
            {
                const vector<char>& data_actual = data_buffer[element_number]->get_item(record_number);
                ASSERT_NE(0, data_actual.size());
                string data        = data_actual.data();
                auto tokens        = split(data, ':');
                int record_actual  = stod(tokens[0]);
                ASSERT_EQ(record_number, record_actual);
                int element_actual = stod(tokens[1]);
                ASSERT_EQ(element_number, element_actual);
            }
        }
    }
}

TEST(block_loader_nds, multiblock_shuffled)
{
    start_server();
    int block_size = 5000;
    int elements_per_record = 2;
    auto client = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, block_size, 1, 0);
    block_iterator_shuffled iter(client);

    for(int block_number=0; block_number<10; block_number++)
    {
        buffer_in_array dest(elements_per_record);
        iter.read(dest);

        vector<buffer_in*> data_buffer;
        for (int element_number=0; element_number<elements_per_record; element_number++)
        {
            data_buffer.push_back(dest[element_number]);
            ASSERT_EQ(block_size, data_buffer[element_number]->record_count());
        }

        for (int record_number=0; record_number<block_size; record_number++)
        {
            for (int element_number=0; element_number<elements_per_record; element_number++)
            {
                const vector<char>& data_actual = data_buffer[element_number]->get_item(record_number);
                ASSERT_NE(0, data_actual.size());
                string data        = data_actual.data();
                auto tokens        = split(data, ':');
//                int record_actual  = stod(tokens[0]);
//                ASSERT_EQ(record_number, record_actual);
                int element_actual = stod(tokens[1]);
                ASSERT_EQ(element_number, element_actual);
            }
        }
    }
}

//TEST(block_loader_nds, performance)
//{
//    //    generate_large_cpio_file();
//    string                        cache_dir = file_util::make_temp_directory();
//    chrono::high_resolution_clock timer;
//    start_server();
//    auto                    client   = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, 16, 1, 0);
//    string                  cache_id = block_loader_random::randomString();
//    string                  version  = "version123";
//    auto                    cache    = make_shared<block_loader_cpio_cache>(cache_dir, cache_id, version, client);
//    block_iterator_shuffled iter(cache);

//    auto startTime = timer.now();
//    for (int i = 0; i < 300; i++)
//    {
//        buffer_in_array dest(2);
//        iter.read(dest);
//    }
//    auto endTime = timer.now();
//    cout << "time " << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count() << " ms" << endl;
//    file_util::remove_directory(cache_dir);
//}
