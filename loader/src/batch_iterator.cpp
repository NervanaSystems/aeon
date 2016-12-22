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
#include "log.hpp"

using namespace nervana;

batch_iterator::batch_iterator(block_manager* blkl, size_t batch_size)
    : async_manager<encoded_record_list, encoded_record_list>(blkl)
    , m_batch_size(batch_size)
{
    m_element_count = element_count();
}

encoded_record_list* batch_iterator::filler()
{
    encoded_record_list* rc = get_pending_buffer();

    rc->clear();

    // This is for the first pass
    if (m_input_ptr == nullptr)
    {
        m_input_ptr = m_source->next();
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
            m_input_ptr = m_source->next();
        }
    }

//    for (size_t item = 0; item < rc->size(); ++item)
//    {
//        const encoded_record& record = rc->record(item);
//        for (size_t element_number=0; element_number<record.size(); element_number++)
//        {
//            std::string element = vector2string(record.element(element_number));
//            INFO << "got element " << element;
//        }
//    }

    return rc;
}
