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

#include <sstream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <iostream>
#include <iterator>

#include "block_loader_file_async.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

block_loader_file_async::block_loader_file_async(manifest_csv* mfst, uint32_t block_size)
    : async_manager<vector<string>,variable_buffer_array>(mfst)
    , m_block_size(block_size)
{
    m_elements_per_record = element_count();
    for (int k = 0; k < 2; ++k)
    {
        for (size_t j = 0; j < m_elements_per_record; ++j)
        {
            m_containers[k].emplace_back();
        }

    }
}

nervana::variable_buffer_array* block_loader_file_async::filler()
{
    variable_buffer_array* rc = get_pending_buffer();

    for (size_t i = 0; i < m_elements_per_record; ++i)
    {
        // These should be empty at this point
        rc->at(i).reset();
    }

    for (int i = 0; i < m_block_size; ++i)
    {
        auto filename_list = m_source->next();

        if (filename_list == nullptr)
        {
            rc = nullptr;
        }
        else
        {
            for (int j = 0; j < m_elements_per_record; ++j)
            {
                try
                {
                    auto buffer = file_util::read_file_contents(filename_list->at(j));
                    rc->at(j).add_item(std::move(buffer));
                }
                catch (std::exception& e)
                {
                    rc->at(j).add_exception(current_exception());
                }
            }
        }
    }
    return rc;
}
