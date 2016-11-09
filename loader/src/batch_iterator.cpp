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

#include "batch_iterator.hpp"

using namespace nervana;

batch_iterator::batch_iterator(std::shared_ptr<block_iterator> src_block_iterator,
                               int batch_size) :
    _src_block_iterator(src_block_iterator),
    _batch_size(batch_size),
    _i(0)
{
    // Note that we don't know how many buffer_ins in we will be writing to until this.read()
    // is called.  So we leave our _macrobatch buffer_in_array pointer to null until we get that
    // information

    _src_block_iterator->reset();

}

void batch_iterator::read(buffer_in_array& dst_buffer_array)
{
    if (_src_buffer_array_ptr == nullptr) {
        _src_buffer_array_ptr = std::make_shared<buffer_in_array>(dst_buffer_array.size());
    }
    // read `_batch_size` items from _src_buffer_array_ptr into `dst_buffer_array`
    for(auto i = 0; i < _batch_size; ++i) {
        pop_item_from_block(dst_buffer_array);
    }
}

void batch_iterator::reset()
{
    for (auto m: *_src_buffer_array_ptr) {
        m->reset();
    }

    _src_block_iterator->reset();

    _i = 0;
}

void batch_iterator::transfer_buffer_item(buffer_in* dst, buffer_in* src)
{
    try {
        dst->add_item(src->get_item(_i));
    } catch (std::exception& e) {
        dst->add_exception(std::current_exception());
    }
}

void batch_iterator::pop_item_from_block(buffer_in_array& dst_buffer_array)
{
    // load a new macrobatch if we've already iterated through the previous one
    buffer_in_array &src_buffer_array = *_src_buffer_array_ptr;

    if(_i >= src_buffer_array[0]->get_item_count()) {
        for (auto m: src_buffer_array) {
            m->reset();
        }

        _src_block_iterator->read(src_buffer_array);

        _i = 0;
    }

    // because the _src_buffer_array_ptr Buffers may have been shuffled, and its shuffle
    // reorders the index, we can't just read a large contiguous block of
    // memory out of the _src_buffer_array_ptr.  We must copy out each element one at
    // a time
    for (uint32_t idx=0; idx < src_buffer_array.size(); ++idx) {
        transfer_buffer_item(dst_buffer_array[idx], src_buffer_array[idx]);
    }

    _i += 1;
}
