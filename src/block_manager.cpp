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

#include <exception>

#include "block_manager.hpp"
#include "file_util.hpp"
#include "cpio.hpp"

using namespace std;
using namespace nervana;

nervana::block_manager::block_manager(shared_ptr<block_loader_source> file_loader,
                                      size_t                          block_size,
                                      const string&                   cache_root,
                                      bool                            enable_shuffle,
                                      uint32_t                        seed)
    : async_manager<encoded_record_list, encoded_record_list>{file_loader, "block_manager"}
    , m_current_block_number{0}
    , m_block_size{file_loader->block_size()}
    , m_block_count{file_loader->block_count()}
    , m_record_count{file_loader->record_count()}
    , m_elements_per_record{file_loader->elements_per_record()}
{
    if (!cache_root.empty())
        m_cache.reset(new cache_system(file_loader->get_uid(),
                                       file_loader->block_count(),
                                       file_loader->elements_per_record(),
                                       cache_root,
                                       enable_shuffle,
                                       seed));
}

void block_manager::initialize()
{
    m_current_block_number = 0;
    if (m_cache)
        m_cache->restart();
    async_manager<encoded_record_list, encoded_record_list>::initialize();
}

nervana::encoded_record_list* block_manager::filler()
{
    m_state                    = async_state::wait_for_buffer;
    encoded_record_list* rc    = get_pending_buffer();
    m_state                    = async_state::processing;
    encoded_record_list* input = nullptr;

    rc->clear();

    if (m_cache && m_cache->is_complete())
    {
        m_cache->load_block(*rc);
    }
    else
    {
        m_state = async_state::fetching_data;
        input   = m_source->next();
        m_state = async_state::processing;
        if (input == nullptr)
        {
            rc = nullptr;
        }
        else
        {
            if (m_cache && m_cache->is_ownership())
                m_cache->store_block(*input);
            input->swap(*rc);
        }

        if (++m_current_block_number == m_block_count)
        {
            m_current_block_number = 0;
            m_source->reset();
            if (m_cache)
                m_cache->try_get_access();
        }
    }

    if (rc && rc->size() == 0)
    {
        rc = nullptr;
    }

    m_state = async_state::idle;
    return rc;
}
