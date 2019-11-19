
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


using namespace std;
using namespace nervana;


TEST(benchmark, jpeg)
{

    char* manifest_root = getenv("TEST_IMAGENET_ROOT");
    if (!manifest_root)
    {
        cout << "Environment vars TEST_IMAGENET_ROOT not found\n";
        return;
    }
    std::string manifest_filename = "train-index.csv";
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

// TODO(sfraczek): move benchmarks from test_loader.cpp and other test files here
