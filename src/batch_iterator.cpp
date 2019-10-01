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

#include <string>

#include "batch_iterator.hpp"
#include "block_manager.hpp"
#include "batch_decoder.hpp"

#include "log.hpp"

using namespace nervana;
using namespace std;

batch_iterator::batch_iterator(shared_ptr<block_manager> blkl, size_t batch_size)
    : async_manager<encoded_record_list, encoded_record_list>(blkl, "batch_iterator")
    , m_batch_size(batch_size)
    , m_element_count(blkl->elements_per_record())
{
}

encoded_record_list* batch_iterator::filler()
{
    m_state                 = async_state::wait_for_buffer;
    encoded_record_list* rc = get_pending_buffer();
    m_state                 = async_state::processing;

    rc->clear();

    // This is for the first pass
    if (m_input_ptr == nullptr)
    {
        m_state     = async_state::fetching_data;
        m_input_ptr = m_source->next();
        m_state     = async_state::processing;
    }

    size_t remainder = m_batch_size;
    while (remainder > 0)
    {
        if (m_input_ptr == nullptr)
        {
            rc = nullptr;
            break;
        }

        size_t move_count;
        if (m_input_ptr->size() <= remainder)
        {
            move_count = m_input_ptr->size();
        }
        else
        {
            move_count = remainder; // Enough in the block to service this batch and more
        }

        // swap records one at a time
        m_input_ptr->move_to(*rc, move_count);

        remainder -= move_count;

        if (remainder > 0 || m_input_ptr->size() == 0)
        {
            m_state     = async_state::fetching_data;
            m_input_ptr = m_source->next();
            m_state     = async_state::processing;
        }
    }

    m_state = async_state::idle;

    return rc;
}

batch_iterator_fbm::batch_iterator_fbm(shared_ptr<batch_decoder>                  blkl,
                                       size_t                                     batch_size,
                                       const std::shared_ptr<provider_interface>& prov,
                                       bool                                       transpose)
    : async_manager<array_fixed_buffer_map, fixed_buffer_map>(blkl, "batch_iterator")
    , m_batch_size(batch_size)
    , m_transpose(transpose)
    , m_element_count(blkl->elements_per_record())
{
    m_element_count = elements_per_record();
    auto oshapes    = prov->get_output_shapes();

    for (unsigned int k = 0; k < m_buffers_number; ++k)
    {
        for (auto& sz : oshapes)
        {
            m_containers[k].add_item(sz.first, sz.second, batch_size, false);
        }
    }
}

fixed_buffer_map* batch_iterator_fbm::filler()
{
    m_state              = async_state::wait_for_buffer;
    fixed_buffer_map* rc = get_pending_buffer();
    m_state              = async_state::processing;

    if (m_input_ptr == nullptr)
    {
        m_src_index = 0;
        m_input_ptr = m_source->next();
        if (m_input_ptr == nullptr)
            return nullptr;
    }

    auto input_size = m_input_ptr->size();

    if (!m_transpose)
    {
        std::swap(*rc, (*m_input_ptr)[m_src_index]);
    }
    else
    {
        const string first_name          = (*m_input_ptr)[m_src_index].get_names()[0];
        size_t       input_data_size     = ((*m_input_ptr)[m_src_index])[first_name]->get_item_count();
        rc->copy((*m_input_ptr)[m_src_index], 0, 0, input_data_size, m_batch_size, true);
    }

    m_src_index++;

    if (input_size == m_src_index)
        m_input_ptr = nullptr;

    return rc;
}
