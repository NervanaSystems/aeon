
/*******************************************************************************
* Copyright 2016-2019 Intel Corporation
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

#include <string>
#include <fstream>  

#include "gtest/gtest.h"
#include "file_util.hpp"
#include "block_loader_file.hpp"
#include "block_manager.hpp"
#include "loader.hpp"


using namespace std;
using namespace nervana;

using nlohmann::json;

TEST(benchmark, jpeg)
{ 
    
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
        return;
    }
    std::string manifest_filename = "train-index.tsv";
    string manifest = file_util::path_join(manifest_root, manifest_filename);
    
        bool     shuffle_manifest = true;
        float    subset_fraction  = 1.0;
        size_t   block_size       = 5000;
        uint32_t random_seed      = 1;
        uint32_t node_id          = 0;
        uint32_t node_count       = 0;
        uint32_t batch_size       = 64;
        auto     m_manifest_file  = make_shared<manifest_file>(manifest,
                                                          shuffle_manifest,
                                                          manifest_root,
                                                          subset_fraction,
                                                          block_size,
                                                          random_seed,
                                                          node_id,
                                                          node_count,
                                                          batch_size);

        auto record_count = m_manifest_file->record_count();
        if (record_count == 0)
        {
            throw std::runtime_error("manifest file is empty");
        }
        std::cout << "Manifest file record count: " << record_count << std::endl;
        std::cout << "Block count: " << record_count / block_size << std::endl;
        
        auto m_elements_per_record = m_manifest_file->elements_per_record();

        for (int i = 0; i < 100 ; i++)
        {   
            stopwatch            timer;
            auto block = m_manifest_file->next();
            for (auto element_list : *block)
            {
                const vector<manifest::element_t>& types = m_manifest_file->get_element_types();
                encoded_record                     record;
                for (int j = 0; j < m_elements_per_record; ++j)
                {
                    const string& element = element_list[j];
                    if  (types[j] == manifest::element_t::FILE)
                    {
                        timer.start();
                        auto buffer = file_util::read_file_contents(element);
                        timer.stop();
    //                    record.add_element(std::move(buffer));
                       // cout<<".";
                    }
                }
            }
            cout<<timer.get_total_milliseconds()<<"    "<< 5005*1000/timer.get_total_milliseconds() <<"\n";
        }
    
}

TEST(benchmark, block)
{
    char* cache_root    = getenv("TEST_CACHE");
    if (!cache_root)
    {
        cout << "Environment variable TEST_CACHE not found\n";
        return;
    }
    string pathname = string(cache_root);
    pathname +="/aeon_cache_7ed31e93/";

    stopwatch            timer;
    
    for (int i = 0; i< 250; i++)
    {
        string filename = pathname + string("block_")+std::to_string(i)+string(".cpio");
        
        std::ifstream filei(filename.c_str(), ios::binary);
        if (!filei.good())
        {
            cout<<filename<<"file not open\n";
            return;
        }
        timer.start();
        filei.seekg(0, ios::end);
        auto length = filei.tellg();
        filei.seekg(0, ios::beg);
    // allocate memory:
        auto    buffer = new char [length];
        // read data as a block:
        
        filei.read(buffer,length);
        timer.stop();
        cout<<timer.get_milliseconds()<<"    "<< 5005*1000/timer.get_milliseconds() <<"\n";
        
        
        
        filei.close();
        delete[] buffer;
    }   
     
}

TEST(benchmark, cache)
{
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    char* manifest_name = getenv("TEST_IMAGENET_MANIFEST");
    char* cache_root    = getenv("TEST_IMAGENET_CACHE");
    char* bsz           = getenv("TEST_IMAGENET_BATCH_SIZE");
    
    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
    }
    else
    {
        size_t batch_size = 128;
        if (bsz)
        {
            std::istringstream iss(bsz);
            iss >> batch_size;
        }

        std::string manifest_filename = "train-index.tsv";
        if (manifest_name)
        {
            std::istringstream iss(manifest_name);
            iss >> manifest_filename;
        }
        string manifest = file_util::path_join(manifest_root, manifest_filename);

        // chrono::high_resolution_clock                     timer;
        // chrono::time_point<chrono::high_resolution_clock> start_time;
        // chrono::time_point<chrono::high_resolution_clock> zero_time;
        // chrono::milliseconds                              total_time{0};

        bool     shuffle_manifest = false;
        bool     shuffle_enable   = false;
        float    subset_fraction  = 1.0;
        size_t   block_size       = 5000;
        uint32_t random_seed      = 0;
        uint32_t node_id          = 0;
        uint32_t node_count       = 0;
        auto     m_manifest_file  = make_shared<manifest_file>(manifest,
                                                          shuffle_manifest,
                                                          manifest_root,
                                                          subset_fraction,
                                                          block_size,
                                                          random_seed,
                                                          node_id,
                                                          node_count,
                                                          batch_size);

        auto record_count = m_manifest_file->record_count();
        if (record_count == 0)
        {
            throw std::runtime_error("manifest file is empty");
        }
        std::cout << "Manifest file record count: " << record_count << std::endl;
        std::cout << "Block count: " << record_count / block_size << std::endl;

        std::shared_ptr<block_loader_source> m_block_loader =
            make_shared<block_loader_file>(m_manifest_file, block_size);

        auto manager = make_shared<block_manager>(
            m_block_loader, block_size, cache_root, shuffle_enable, random_seed);

        //encoded_record_list* records;
        stopwatch            timer;
        timer.start();
        float  count       = 0;
        size_t iterations  = record_count / block_size;
        float  total_count = 0;
        float  total_time  = 0;
        for (size_t i = 0; i < iterations; i++)
        {
            auto records = manager->next();
            timer.stop();
            count      = records->size();
            float time = timer.get_microseconds() / 1000000.;
            cout << setprecision(0) << "block id=" << i << ", count=" << static_cast<int>(count)
                 << ", time=" << fixed << setprecision(6) << time << " images/second "
                 << setprecision(2) << count / time << "\n";
            total_count += count;
            total_time += time;
            timer.start();
        }
        cout << setprecision(0) << "total count=" << total_count
             << ", total time=" << setprecision(6) << total_time << ", average images/second "
             << total_count / total_time << endl;
    }
}

void benchmark_imagenet(json config, char* batch_delay, size_t batch_size)
{
    try
    {
        loader_factory factory;
        auto           train_set = factory.get_loader(config);

        size_t       total_batch   = ceil((float)train_set->record_count() / (float)batch_size);
        size_t       current_batch = 0;
        const size_t batches_per_output = 100;
        stopwatch    timer;
        timer.start();
        for (const nervana::fixed_buffer_map& x : *train_set)
        {
            (void)x;
            if (++current_batch % batches_per_output == 0)
            {
                timer.stop();
                float ms_time  = timer.get_milliseconds();
                float sec_time = ms_time / 1000.;

                cout << "batch " << current_batch << " of " << total_batch;
                cout << " time " << sec_time;
                cout << " " << batch_size * (float)batches_per_output / sec_time << " img/s";
                cout << "\t\taverage "
                     << batch_size * (float)batches_per_output /
                            ((float)timer.get_total_milliseconds() / timer.get_call_count() /
                             1000.0f)
                     << " img/s" << endl;
                timer.start();
            }
        }
    }
    catch (exception& err)
    {
        cout << "error processing dataset" << endl;
        cout << err.what() << endl;
    }
}

TEST(benchmark, imagenet)
{
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    char* manifest_name = getenv("TEST_IMAGENET_MANIFEST");
    char* cache_root    = getenv("TEST_IMAGENET_CACHE");
    char* address       = getenv("TEST_IMAGENET_ADDRESS");
    char* port          = getenv("TEST_IMAGENET_PORT");
    char* rdma_address  = getenv("TEST_IMAGENET_RDMA_ADDRESS");
    char* rdma_port     = getenv("TEST_IMAGENET_RDMA_PORT");
    char* session_id    = getenv("TEST_IMAGENET_SESSION_ID");
    char* async         = getenv("TEST_IMAGENET_ASYNC");
    char* batch_delay   = getenv("TEST_IMAGENET_BATCH_DELAY");
    char* bsz           = getenv("TEST_IMAGENET_BATCH_SIZE");
    char* iterations    = getenv("TEST_IMAGENET_ITERATIONS");

    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
    }
    else
    {
        int         height     = 224;
        int         width      = 224;
        size_t      batch_size = bsz ? atoi(bsz) : 128;
        std::string manifest =
            file_util::path_join(manifest_root, manifest_name ? manifest_name : "train-index.tsv");
        std::string iteration_mode       = iterations ? "COUNT" : "INFINITE";
        int         iteration_mode_count = iterations ? atoi(iterations) : 0;

        json image_config = {
            {"type", "image"}, {"height", height}, {"width", width}, {"channel_major", false}};
        json label_config = {{"type", "label"}, {"binary", false}};
        auto aug_config   = vector<json>{{{"type", "image"},
                                        {"scale", {0.5, 1.0}},
                                        {"saturation", {0.5, 2.0}},
                                        {"contrast", {0.5, 1.0}},
                                        {"brightness", {0.5, 1.0}},
                                        {"flip_enable", true}}};
        json config       = {{"manifest_root", manifest_root},
                       {"manifest_filename", manifest},
                       {"batch_size", batch_size},
                       {"iteration_mode", iteration_mode},
                       {"iteration_mode_count", iteration_mode_count},
                       {"cache_directory", cache_root ? cache_root : ""},
                       {"cpu_list", ""},
                       //{"web_server_port", 8086},
                       {"etl", {image_config, label_config}},
                       {"augmentation", aug_config}};

        if (address != NULL && port != NULL)
        {
            config["remote"]["address"] = address;
            config["remote"]["port"]    = std::stoi(port);
            if (session_id != NULL)
            {
                config["remote"]["session_id"] = session_id;
            }
            if (async != NULL)
            {
                bool b;
                istringstream(async) >> b;
                config["remote"]["async"] = b;
            }
            if (rdma_address != NULL && rdma_port != NULL)
            {
                config["remote"]["rdma_address"] = rdma_address;
                config["remote"]["rdma_port"]    = std::stoi(rdma_port);
            }
        }

        benchmark_imagenet(config, batch_delay, batch_size);
    }
}

TEST(benchmark, imagenet_paddle)
{
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    char* manifest_name = getenv("TEST_IMAGENET_MANIFEST");
    char* cache_root    = getenv("TEST_IMAGENET_CACHE");
    char* batch_delay   = getenv("TEST_IMAGENET_BATCH_DELAY");
    char* bsz           = getenv("TEST_IMAGENET_BATCH_SIZE");
    char* iterations    = getenv("TEST_IMAGENET_ITERATIONS");

    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
    }
    else
    {
        int         height     = 224;
        int         width      = 224;
        size_t      batch_size = bsz ? atoi(bsz) : 128;
        std::string manifest =
            file_util::path_join(manifest_root, manifest_name ? manifest_name : "train-index.tsv");
        std::string iteration_mode       = iterations ? "COUNT" : "INFINITE";
        int         iteration_mode_count = iterations ? atoi(iterations) : 0;

        json image_config = {{"type", "image"},
                             {"height", height},
                             {"width", width},
                             {"channels", 3},
                             {"output_type", "float"},
                             {"channel_major", true},
                             {"bgr_to_rgb", true}};

        json label_config = {{"type", "label"}, {"binary", false}};

        auto aug_config = vector<json>{{{"type", "image"},
                                        {"flip_enable", true},
                                        {"center", false},
                                        {"crop_enable", true},
                                        {"horizontal_distortion", {3. / 4., 4. / 3.}},
                                        {"do_area_scale", true},
                                        {"scale", {0.08, 1.0}},
                                        {"mean", {0.485, 0.456, 0.406}},
                                        {"stddev", {0.229, 0.224, 0.225}},
                                        {"resize_short_size", 0}}};
        json config     = {{"manifest_root", manifest_root},
                       {"manifest_filename", manifest},
                       {"shuffle_enable", true},
                       {"shuffle_manifest", true},
                       {"batch_size", batch_size},
                       {"iteration_mode", iteration_mode},
                       {"iteration_mode_count", iteration_mode_count},
                       {"cache_directory", cache_root ? cache_root : ""},
                       {"cpu_list", ""},
                       {"etl", {image_config, label_config}},
                       {"augmentation", aug_config}};

        benchmark_imagenet(config, batch_delay, batch_size);
    }
}

TEST(benchmark, decode_jpeg)
{
    stopwatch timer;
    size_t    manifest_size = 10000;
    string    image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");

    vector<char> image_data = file_util::read_file_contents(image_path);
    timer.start();
    for (size_t i = 0; i < manifest_size; i++)
    {
        cv::Mat output_img;
        cv::Mat input_img(1, image_data.size(), CV_8UC3, image_data.data());
        cv::imdecode(input_img, CV_8UC3, &output_img);
    }
    auto time = (float)timer.get_milliseconds() / 1000.;
    cout << ((float)manifest_size / time) << " images/second " << endl;
}

TEST(benchmark, read_jpeg)
{
    stopwatch timer;

    size_t       manifest_size = 10000;
    string       image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");
    stringstream manifest_stream;
    manifest_stream << "@FILE"
                    << "\n";
    for (size_t i = 0; i < manifest_size; i++)
    {
        manifest_stream << image_path << "\n";
    }
    manifest_file manifest{manifest_stream, false};

    vector<vector<string>>* block = nullptr;
    size_t                  count = 0;
    size_t                  mod   = 10000;
    timer.start();
    for (block = manifest.next(); block != nullptr; block = manifest.next())
    {
        for (const vector<string>& record : *block)
        {
            count++;
            vector<char> image_data = file_util::read_file_contents(record[0]);
            cv::Mat      output_img;
            cv::Mat      input_img(1, image_data.size(), CV_8UC3, image_data.data());
            cv::imdecode(input_img, CV_8UC3, &output_img);
            if (count % mod == 0)
            {
                auto time = (float)timer.get_milliseconds() / 1000.;
                cout << ((float)count / time) << " images/second " << endl;
            }
        }
    }
}

TEST(benchmark, load_jpeg)
{
    stopwatch timer;

    size_t manifest_size = 10000;
    string image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");

    timer.start();
    for (size_t i = 0; i < manifest_size; i++)
    {
        auto data = file_util::read_file_contents(image_path);
    }
    auto time = (float)timer.get_milliseconds() / 1000.;
    cout << "images/second " << ((float)manifest_size / time) << endl;
}

TEST(benchmark, load_jpeg_manifest)
{
    stopwatch timer;

    size_t       manifest_size = 10000;
    string       image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");
    stringstream manifest_stream;
    manifest_stream << "@FILE"
                    << "\n";
    for (size_t i = 0; i < manifest_size; i++)
    {
        manifest_stream << image_path << "\n";
    }
    manifest_file manifest{manifest_stream, false};

    vector<vector<string>>* block = nullptr;
    timer.start();
    for (block = manifest.next(); block != nullptr; block = manifest.next())
    {
        for (const vector<string>& record : *block)
        {
            auto data = file_util::read_file_contents(record[0]);
        }
    }
    auto time = (float)timer.get_milliseconds() / 1000.;
    cout << "images/second " << ((float)manifest_size / time) << endl;
}

TEST(benchmark, load_block_manager)
{
    char* cache_root = getenv("TEST_CACHE");
    if (!cache_root)
    {
        cout << "Environment variable TEST_CACHE not found\n";
        return;
    }
    stopwatch timer;
    bool      shuffle         = false;
    size_t    block_size      = 5000;

    size_t       manifest_size = 30000;
    string       image_path    = file_util::path_join(CURDIR, "test_data/img_2112_70.jpg");
    stringstream manifest_stream;
    manifest_stream << "@FILE"
                    << "\n";
    for (size_t i = 0; i < manifest_size; i++)
    {
        manifest_stream << image_path << "\n";
    }
    auto manifest = make_shared<manifest_file>(manifest_stream, false);

    auto loader = make_shared<block_loader_file>(manifest, block_size);

    block_manager manager{loader, 5000, cache_root, shuffle};

    encoded_record_list* records;
    timer.start();
    float  count      = 0;
    size_t iterations = (manifest_size / block_size) * 3;
    for (size_t i = 0; i < iterations; i++)
    {
        records = manager.next();
        timer.stop();
        count      = records->size();
        float time = timer.get_microseconds() / 1000000.;
        cout << "count=" << count << ", time=" << time << " images/second " << count / time << endl;
        timer.start();
    }
}

TEST(benchmark, manifest)
{
    string manifest_filename = file_util::tmp_filename();
    string cache_root        = "/this/is/supposed/to/be/long/so/we/make/it/so/";
    cout << "tmp manifest file " << manifest_filename << endl;

    chrono::high_resolution_clock timer;

    // Generate a manifest file
    {
        auto     startTime = timer.now();
        ofstream mfile(manifest_filename);
        mfile << "@FILE\tFILE" << std::endl;
        for (int i = 0; i < 10e6; i++)
        {
            mfile << cache_root << "image_" << i << ".jpg\t";
            mfile << cache_root << "target_" << i << ".txt\n";
        }
        auto endTime = timer.now();
        cout << "create manifest "
             << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count() << " ms"
             << endl;
    }

    // Parse the manifest file
    shared_ptr<manifest_file> manifest;
    {
        auto startTime = timer.now();
        manifest       = make_shared<manifest_file>(manifest_filename, false);
        auto endTime   = timer.now();
        cout << "load manifest "
             << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count() << " ms"
             << endl;
    }

    // compute the CRC
    {
        auto     startTime = timer.now();
        uint32_t crc       = manifest->get_crc();
        auto     endTime   = timer.now();
        cout << "manifest crc 0x" << setfill('0') << setw(8) << hex << crc << dec << endl;
        cout << "crc time "
             << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count() << " ms"
             << endl;
    }

    remove(manifest_filename.c_str());
}
