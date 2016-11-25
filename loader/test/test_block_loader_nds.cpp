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
        ws.register_page_handler(fn);
        ws.start(5000);
    }

    ~NDSMockServer()
    {
        ws.stop();
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
            string path = CURDIR "/../../test/test.cpio";
            page.send_file(path);
        }
        else if (url == "/test_pattern/")
        {
            for (int i=0; i<1024; i++)
            {
                page.send_string("0123456789abcdef");
            }
        }
        else if (url == "/shutdown/")
        {
        }
        else if (url == "/error")
        {
            page.page_not_found();
        }
    }

private:
    web::server ws;
    pid_t _pid;
};

std::shared_ptr<NDSMockServer> mock_server;

static void start_server()
{
    if (mock_server == nullptr)
    {
        mock_server = make_shared<NDSMockServer>();
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

TEST(block_loader_nds, object_count)
{
    start_server();
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    // 200 and 5 are hard coded in the mock nds server
    ASSERT_EQ(client.object_count(), 200);
    ASSERT_EQ(client.block_count(), 5);
}

TEST(block_loader_nds, cpio)
{
    start_server();
    block_loader_nds client("http://127.0.0.1:5000", "token", 1, 16, 1, 0);

    buffer_in_array dest(2);
    ASSERT_EQ(dest.size(), 2);
    ASSERT_EQ(dest[0]->get_item_count(), 0);

    client.load_block(dest, 0);

//    ASSERT_EQ(dest[0]->get_item_count(), 2);
}

string generate_large_cpio_file()
{
    char name[8192];
    realpath("test_big.cpio", name);
    string cpio_file(name);

    cpio::file_writer writer;
    writer.open(cpio_file);
    buffer_in_array buf(2);
    vector<char>    image_data(8000, 42);
    vector<char>    target_data(4, 0);
    buf[0]->add_item(image_data);
    buf[1]->add_item(target_data);
    for (int i = 0; i < 5000; i++)
    {
        writer.write_all_records(buf);
    }
    writer.close();
    return cpio_file;
}

TEST(block_loader_nds, multiblock_sequential)
{
    start_server();
    auto client = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, 16, 1, 0);
    block_iterator_sequential iter(client);

    for(int i=0; i<1; i++)
    {
        buffer_in_array dest(2);
        iter.read(dest);
        buffer_in* image_array = dest[0];

        for (int i=0; i<image_array->get_item_count(); i++)
        {
            const vector<char>& image_data = image_array->get_item(i);
            ASSERT_NE(0, image_data.size());
        }
    }
}

TEST(block_loader_nds, multiblock_shuffled)
{
    start_server();
    auto client = make_shared<block_loader_nds>("http://127.0.0.1:5000", "token", 1, 16, 1, 0);
    block_iterator_shuffled iter(client);

    for(int i=0; i<5; i++)
    {
        buffer_in_array dest(2);
        iter.read(dest);
        buffer_in* image_array = dest[0];

        for (int i=0; i<image_array->get_item_count(); i++)
        {
            const vector<char>& image_data = image_array->get_item(i);
            ASSERT_NE(0, image_data.size());
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
