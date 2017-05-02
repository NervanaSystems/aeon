/*
 Copyright 2017 Nervana Systems Inc.
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

#include "batch_decoder.hpp"
#include "provider_factory.hpp"

using namespace std;
using namespace nervana;

batch_decoder::batch_decoder(batch_iterator*                            b_itor,
                             size_t                                     batch_size,
                             uint32_t                                   thread_count,
                             bool                                       pinned,
                             const std::shared_ptr<provider_interface>& prov)
    : async_manager<encoded_record_list, fixed_buffer_map>(b_itor, "batch_decoder")
    , m_batch_size(batch_size)
    , m_active_count{0}
{
    // Note:  all we need are thread_count, batch_size, pinned + the provider template
    //        can we just use copy constructor instead?
    int nthreads = 1;

    if (thread_count == 0)  // automatically determine number of threads
    {
        int itemsPerThread = (batch_size - 1) / thread::hardware_concurrency() + 1;
        nthreads           = std::min((batch_size - 1) / itemsPerThread + 1, batch_size);
    }
    else
    {
        // don't return more threads than we can get
        nthreads = std::min(thread::hardware_concurrency(), thread_count);

        // don't return more threads than items per batch
        nthreads = std::min((int) batch_size, nthreads);

        // TODO: log info message if nthreads != thread_count
    }

    if (nthreads <= 0)
    {
        throw std::invalid_argument("Number of threads must be > 0");
    }

    m_items_per_thread = (batch_size - 1) / nthreads + 1;

    for (int i = 0; i < nthreads; i++)
    {
        int start = i * m_items_per_thread;
        int record_count =
            i == nthreads - 1 ? (batch_size - i * m_items_per_thread) : m_items_per_thread;
        int end = start + record_count;
        m_decode_thread_info.push_back(make_shared<decode_thread_info>(
            i, start, end, prov, m_work_complete_event, m_active_count));
    }

    auto oshapes         = prov->get_output_shapes();
    m_number_elements_in = prov->get_input_count();

    // Allocate the space in the output buffers
    for (unsigned int k = 0; k < 2; ++k)
    {
        for (auto& sz : oshapes)
        {
            m_containers[k].add_item(sz.first, sz.second, batch_size, pinned);
        }
    }
}

batch_decoder::~batch_decoder()
{
    finalize();
}

fixed_buffer_map* batch_decoder::filler()
{
    m_state                     = async_state::wait_for_buffer;
    fixed_buffer_map* outputs   = get_pending_buffer();
    m_state                     = async_state::fetching_data;
    encoded_record_list* inputs = m_source->next();
    m_state                     = async_state::processing;

    if (inputs == nullptr)
    {
        outputs = nullptr;
    }
    else
    {
        for(const encoded_record& record : *inputs)
        {
            record.rethrow_if_exception();
        }

        m_active_count = m_decode_thread_info.size();
        for (int id = 0; id < m_decode_thread_info.size(); ++id)
        {
            m_decode_thread_info[id]->set_buffers(inputs, outputs);
        }
        m_work_complete_event.wait();
    }

    m_state = async_state::idle;
    return outputs;
}
