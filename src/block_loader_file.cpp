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

#include <sstream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <iostream>
#include <iterator>

#include "block_loader_file.hpp"
#include "util.hpp"
#include "file_util.hpp"
#include "log.hpp"
#include "base64.hpp"

using namespace std;
using namespace nervana;

block_loader_file::block_loader_file(shared_ptr<manifest_file> manifest, size_t block_size)
    : async_manager<std::vector<std::vector<std::string>>, encoded_record_list>{manifest,
                                                                                "block_loader_file"}
    , m_block_size(block_size)
    , m_record_count{manifest->record_count()}
    , m_manifest(manifest)
{
    m_block_count         = round((float)m_manifest->record_count() / (float)m_block_size);
    m_block_size          = ceil((float)m_manifest->record_count() / (float)m_block_count);
    m_elements_per_record = manifest->elements_per_record();
}

nervana::encoded_record_list* block_loader_file::filler()
{
    m_state                 = async_state::wait_for_buffer;
    encoded_record_list* rc = get_pending_buffer();
    m_state                 = async_state::processing;

    rc->clear();

    m_state    = async_state::fetching_data;
    auto block = m_source->next();
    m_state    = async_state::processing;
    if (block != nullptr)
    {
        for (auto element_list : *block)
        {
            const vector<manifest::element_t>& types = m_manifest->get_element_types();
            encoded_record                     record;
            for (int j = 0; j < m_elements_per_record; ++j)
            {
                try
                {
                    const string& element = element_list[j];
                    switch (types[j])
                    {
                    case manifest::element_t::FILE:
                    {
                        auto buffer = file_util::read_file_contents(element);
                        record.add_element(std::move(buffer));
                        break;
                    }
                    case manifest::element_t::BINARY:
                    {
                        vector<char> buffer  = string2vector(element);
                        vector<char> decoded = base64::decode(buffer);
                        record.add_element(std::move(decoded));
                        break;
                    }
                    case manifest::element_t::STRING:
                    {
                        record.add_element(element.data(), element.size());
                        break;
                    }
                    case manifest::element_t::ASCII_INT:
                    {
                        int32_t value = stod(element);
                        record.add_element(&value, sizeof(value));
                        break;
                    }
                    case manifest::element_t::ASCII_FLOAT:
                    {
                        float value = stof(element);
                        record.add_element(&value, sizeof(value));
                        break;
                    }
                    }
                }
                catch (std::exception&)
                {
                    record.add_exception(current_exception());
                }
            }
            rc->add_record(std::move(record));
        }
    }

    if (rc && rc->size() == 0)
    {
        rc = nullptr;
    }

    m_state = async_state::idle;
    return rc;
}
