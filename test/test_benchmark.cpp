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

#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <iostream>
#include <ios>

#include "gtest/gtest.h"

#define private public

#include "loader.hpp"
#include "manifest_builder.hpp"
#include "gen_image.hpp"
#include "file_util.hpp"
#include "util.hpp"
#include "cache_file.h"

#if defined(ENABLE_AEON_SERVICE)
#include "service/service.hpp"
#include "client/loader_remote.hpp"
#endif

using namespace std;
using namespace nervana;

using nlohmann::json;

namespace 
{
    void benchmark_imagenet(json config, char* batch_delay, size_t batch_size)
    {
        try
        {
            loader_factory factory;
            auto           train_set = factory.get_loader(config);

            size_t       total_batch   = ceil((float)train_set->record_count() / (float)batch_size);
            size_t       current_batch = 0;
            const size_t batches_per_output = 50;
            stopwatch    timer;
            timer.start();
            for (const nervana::fixed_buffer_map& x : *train_set)
            {
                (void)x;
                if (++current_batch % batches_per_output == 0)
                {
                    timer.stop();
                    float ms_time = timer.get_milliseconds();
                    float sec_time = ms_time / 1000.;
                    
                    cout << "batch " << current_batch << " of " << total_batch;
                    cout << " time " << sec_time;
                    cout << " " << batch_size * (float)batches_per_output / sec_time << " img/s";
                    cout << "\t\taverage "
                            <<  batch_size * (float)batches_per_output / ((float)timer.get_total_milliseconds() 
                                /timer.get_call_count()/1000.0f)
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
}


TEST(benchmark, create_cache)
{
    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
        return;
    }
    std::string manifest_filename =
            file_util::path_join(manifest_root, "train-index.csv");

    auto manifest_file_src = make_shared<manifest_file>(manifest_filename, false, manifest_root);
    auto block_loader = make_shared<block_loader_file>(manifest_file_src, 1);

    //int record_count = 20; // manifest_file_src->record_count()
    int record_count = manifest_file_src->record_count();
    CacheFile cache_file("/home/ashvay/aeon_cache.bin", record_count);
    for (int i = 0; i < record_count; i++)
    {
       cout<<"["<<i<<"/"<<record_count<<"]"<<std::endl;
       encoded_record record = std::move(*block_loader->next()->begin());
       cache_file.add_record(record);
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
            file_util::path_join(manifest_root, manifest_name ? manifest_name : "train-index.csv");
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
        json config = {{"manifest_root", manifest_root},
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
            file_util::path_join(manifest_root, manifest_name ? manifest_name : "train-index.csv");
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
        json config = {{"manifest_root", manifest_root},
                       {"manifest_filename", manifest},
                       {"shuffle_enable", true},
                       {"shuffle_manifest", true},
                       {"batch_size", batch_size},
                       {"iteration_mode", iteration_mode},
                       {"iteration_mode_count", iteration_mode_count},
                       {"cache_directory", cache_root ? cache_root : ""},
                       {"cpu_list", ""},
                       {"etl", {image_config, label_config}},
                       {"random_seed", 1},
                       {"augmentation", aug_config}};

        benchmark_imagenet(config, batch_delay, batch_size);
    }
}

