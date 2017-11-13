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

#pragma once
#include <string>
#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "provider_interface.hpp"
#include "util.hpp"

/* block_loader_file
 *
 * Loads blocks of files from a Manifest into a BufferPair.
 *
 */

namespace nervana
{
    class batch_iterator;
    class batch_iterator_fbm; // batch iterator for fixed_buffer_map type
    class block_manager;
    class batch_decoder;
}

class nervana::batch_iterator : public async_manager<encoded_record_list, encoded_record_list>
{
public:
    batch_iterator(block_manager*, size_t batch_size);
    ~batch_iterator() { finalize(); }
    encoded_record_list* filler() override;

    size_t record_count() const override { return m_batch_size; }
    size_t elements_per_record() const override { return m_element_count; }
    void   initialize() override
    {
        m_input_ptr = nullptr;
        async_manager<encoded_record_list, encoded_record_list>::initialize();
    }

private:
    size_t               m_batch_size;
    size_t               m_element_count;
    encoded_record_list* m_input_ptr{nullptr};
};

class nervana::batch_iterator_fbm : public async_manager<fixed_buffer_map, fixed_buffer_map>
{
public:
    batch_iterator_fbm(batch_decoder*                             blkl,
                       size_t                                     batch_size,
                       const std::shared_ptr<provider_interface>& prov,
                       bool                                       transpose);
    ~batch_iterator_fbm() { finalize(); }
    fixed_buffer_map* filler() override;

    size_t record_count() const override { return m_batch_size; }
    size_t elements_per_record() const override { return m_element_count; }
    void   initialize() override
    {
        m_input_ptr = nullptr;
        async_manager<fixed_buffer_map, fixed_buffer_map>::initialize();
    }

private:
    size_t            m_batch_size;
    bool              m_transpose;
    size_t            m_element_count;
    fixed_buffer_map* m_input_ptr{nullptr};
    size_t            m_src_index = 0;
    size_t            m_dst_index = 0;
};
