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

#include <memory>

#include "buffer_in.hpp"
#include "block_iterator.hpp"

namespace nervana
{
    class batch_iterator;
}

class nervana::batch_iterator
{
public:
    batch_iterator(std::shared_ptr<block_iterator> src_block_iterator, int batch_size);

    void read(nervana::buffer_in_array& dst_buffer_array);
    void reset();
protected:
    void pop_item_from_block(nervana::buffer_in_array& dst_buffer_array);
    void transfer_buffer_item(nervana::buffer_in* dst, nervana::buffer_in* src);

    std::shared_ptr<block_iterator> m_src_block_iterator;
    int m_batch_size;

    std::shared_ptr<nervana::buffer_in_array> m_src_buffer_array_ptr;
    // the index into the m_macrobatch to read next
    int m_i;
};
