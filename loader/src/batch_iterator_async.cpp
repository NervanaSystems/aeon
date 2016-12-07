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

#include "batch_iterator_async.hpp"
#include "log.hpp"

using namespace nervana;

batch_iterator_async::batch_iterator_async(block_loader_file_async* blkl, size_t batch_size)
    : async_manager<variable_buffer_array, variable_buffer_array>(blkl)
    , m_batch_size(batch_size)
{
    m_element_count = element_count();
    for (int k = 0; k < 2; ++k)
    {
        for (size_t j = 0; j < m_element_count; ++j)
        {
            m_containers[k].emplace_back();
        }
    }
}

variable_buffer_array* batch_iterator_async::filler()
{
    variable_buffer_array* rc = get_pending_buffer();

    // This is for the first pass
    if (m_input_ptr == nullptr)
    {
        m_input_ptr = m_source->next();
    }

    // Empty this buffer so that it can be filled
    for (auto& ct : *rc)
    {
        ct.reset();
    }

    size_t number_needed = m_batch_size - rc->at(0).size();

    while (number_needed > 0)
    {
        if (m_input_ptr == nullptr)
        {
            rc = nullptr;
            break;
        }

        size_t move_count;
        if (m_input_ptr->at(0).size() <= number_needed)
        {
            move_count = m_input_ptr->at(0).size();
        }
        else
        {
            move_count = number_needed; // Enough in the block to service this batch and more
        }

        move_src_to_dst(m_input_ptr, rc, move_count);

        number_needed -= move_count;

        if (number_needed > 0)
        {
            m_input_ptr = m_source->next();
        }
    }

    return rc;
}

void batch_iterator_async::move_src_to_dst(variable_buffer_array* src_array_ptr, variable_buffer_array* dst_array_ptr, size_t count)
{
    for (size_t ridx = 0; ridx < m_element_count; ++ridx)
    {
        buffer_variable_size_elements& src = src_array_ptr->at(ridx);
        buffer_variable_size_elements& dst = dst_array_ptr->at(ridx);

        auto start_iter = src.begin();
        auto end_iter   = src.begin() + count;

        dst.append(make_move_iterator(start_iter), make_move_iterator(end_iter));
        src.erase(start_iter, end_iter);
    }
}
