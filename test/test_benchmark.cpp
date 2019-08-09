
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

#include "gtest/gtest.h"
#include "file_util.hpp"
#include "block_loader_file.hpp"
#include "block_manager.hpp"

using namespace std;
using namespace nervana;

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

        std::string manifest_filename = "train-index.csv";
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

        encoded_record_list* records;
        stopwatch            timer;
        timer.start();
        float  count       = 0;
        size_t iterations  = record_count / block_size;
        float  total_count = 0;
        float  total_time  = 0;
        for (size_t i = 0; i < iterations; i++)
        {
            records = manager->next();
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

// TODO(sfraczek): move benchmarks from test_loader.cpp and other test files here
