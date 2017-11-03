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

#include <iostream>
#include <chrono>
#include <clocale>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "gen_image.hpp"
#include "file_util.hpp"
#include "web_server.hpp"
#include "web_app.hpp"
#include "log.hpp"
#include "json.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

extern gen_image image_dataset;
extern string    test_cache_directory;

gen_image image_dataset;
string    test_cache_directory;

// NDSMockServer starts a python process in the constructor and kills the
// process in the destructor
class mock_nds_server
{
public:
    mock_nds_server()
    {
        page_request_handler fn =
            bind(&mock_nds_server::page_handler, this, placeholders::_1, placeholders::_2);
        m_server.register_page_handler(fn);
        m_server.start(5000);
    }

    ~mock_nds_server() {}
    void set_elements_per_record(initializer_list<int> init) { m_elements_size_list = init; }
    void page_handler(web::page& page, const std::string& url)
    {
        if (url == "/object_count/")
        {
            nlohmann::json js = {{"record_count", 200}, {"macro_batch_per_shard", 5}};
            string         rc = js.dump();
            page.send_string(rc);
        }
        else if (url == "/macrobatch/")
        {
            map<string, string> args = page.args();
            size_t block_size    = stod(args["macro_batch_max_size"]);
            size_t block_index   = stod(args["macro_batch_index"]);
            size_t collection_id = stod(args["collection_id"]);
            string token         = args["token"];
            (void)block_index;
            (void)collection_id;
            (void)token; // silence warning
            stringstream ss;
            {
                size_t              record_start = block_size * block_index;
                cpio::writer        writer(ss);
                encoded_record_list record_list;
                for (size_t record_number = 0; record_number < block_size; record_number++)
                {
                    encoded_record record;
                    for (size_t element_number = 0; element_number < m_elements_size_list.size();
                         element_number++)
                    {
                        stringstream tmp;
                        tmp << record_number + record_start << ":" << element_number;
                        string id   = tmp.str();
                        auto   data = string2vector(id);
                        record.add_element(data);
                    }
                    record_list.add_record(record);
                }
                writer.write_all_records(record_list);
            }

            string cpio_data = ss.str();
            page.send_as_file(cpio_data.data(), cpio_data.size());
        }
        else if (url == "/test_pattern/")
        {
            for (int i = 0; i < 1024; i++)
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
    vector<int> m_elements_size_list = {1024, 32};
};

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

void exit_func(int s)
{
    //    cout << __FILE__ << " " << __LINE__ << "exit function " << s << endl;
    //    exit(-1);
}

void page_handler(web::page& page, const std::string& url)
{
    using std::chrono::system_clock;
    system_clock::time_point today = system_clock::now();
    time_t                   tt    = system_clock::to_time_t(today);

    page.page_ok();
    page.output_stream() << "<html>Now is " << ctime(&tt) << "</html>";
}

void web_server()
{
    web::server ws;
    ws.register_page_handler(page_handler);
    ws.start(8086);
    sleep(10);
    ws.stop();
}

extern "C" int main(int argc, char** argv)
{
    std::setlocale(LC_CTYPE, "");

    cout << "OpenCV version : " << CV_VERSION << endl;
    mock_nds_server server;

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
