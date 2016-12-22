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
#include "block_manager.hpp"
#include "buffer_batch.hpp"
#include "util.hpp"

/* block_loader_file
 *
 * Loads blocks of files from a Manifest into a BufferPair.
 *
 */

namespace nervana
{
    class batch_iterator;
}

class nervana::batch_iterator
    : public async_manager<encoded_record_list, encoded_record_list>
{
public:
    batch_iterator(block_manager*, size_t batch_size);
    virtual ~batch_iterator() { finalize(); }
    virtual size_t record_count() const override { return m_batch_size; }
    virtual encoded_record_list* filler() override;

    virtual void initialize() override
    {
        async_manager<encoded_record_list, encoded_record_list>::initialize();
        m_input_ptr = nullptr;
    }

private:
    size_t                 m_batch_size;
    size_t                 m_element_count;
    encoded_record_list*   m_input_ptr{nullptr};
};
