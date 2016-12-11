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
#include "base64.hpp"

using namespace std;
using namespace nervana;

block_loader_file_async::block_loader_file_async(manifest_csv* manifest, size_t block_size)
    : block_loader_source_async(manifest)
    , m_block_size(block_size)
    , m_manifest(*manifest)
{
//    INFO << "file loader requested block size " << m_block_size;
    m_block_count = round((float)m_manifest.record_count() / (float)m_block_size);
    m_block_size = ceil((float)m_manifest.record_count() / (float)m_block_count);
//    INFO << "file loader new block size " << m_block_size;
//    INFO << "file loader record count " << m_manifest.record_count();
//    INFO << "file loader m_block_count " << m_block_count;
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
        auto element_list = m_source->next();

        if (element_list == nullptr)
        {
            break;
        }
        else
        {
            const vector<manifest::element_t>& types = m_manifest.get_element_types();
            for (int j = 0; j < m_elements_per_record; ++j)
            {
                try
                {
                    const string& element = element_list->at(j);
                    switch (types[j])
                    {
                    case manifest::element_t::FILE:
                    {
                        auto buffer = file_util::read_file_contents(element);
                         rc->at(j).add_item(std::move(buffer));
                        break;
                    }
                    case manifest::element_t::BINARY:
                    {
                        vector<char> buffer = string2vector(element);
                        vector<char> decoded = base64::decode(buffer);
                        rc->at(j).add_item(std::move(decoded));
                        break;
                    }
                    case manifest::element_t::STRING:
                    {
                        rc->at(j).add_item(element.data(), element.size());
                        break;
                    }
                    case manifest::element_t::ASCII_INT:
                    {
                        int32_t value = stod(element);
                        rc->at(j).add_item(&value, sizeof(value));
                        break;
                    }
                    case manifest::element_t::ASCII_FLOAT:
                    {
                        float value = stof(element);
                        rc->at(j).add_item(&value, sizeof(value));
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }
                catch (std::exception& e)
                {
                    rc->at(j).add_exception(current_exception());
                }
            }
        }
    }

    if (rc && rc->at(0).size() == 0)
    {
        rc = nullptr;
    }

//    if (rc) INFO << rc->at(0).size() << ", " << rc->at(1).size(); else INFO << "nullptr";
    return rc;
}
