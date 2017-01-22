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
                             bool                                       single_thread,
                             bool                                       pinned,
                             const std::shared_ptr<provider_interface>& prov)
    : async_manager<encoded_record_list, fixed_buffer_map>(b_itor, "batch_decoder")
    , m_batch_size(batch_size)
{
    // Note:  all we need are single_thread, batch_size, pinned + the provider template
    //        can we just use copy constructor instead?
    int nthreads = 1;

    if (!single_thread)
    {
        int itemsPerThread = (batch_size - 1) / thread::hardware_concurrency() + 1;
        nthreads           = std::min((batch_size - 1) / itemsPerThread + 1, batch_size);
    }

    m_items_per_thread = (batch_size - 1) / nthreads + 1;

    if (nthreads <= 0)
    {
        throw std::invalid_argument("Number of threads must be > 0");
    }

    for (int i = 0; i < nthreads; i++)
    {
        m_providers.push_back(nervana::provider_factory::clone(prov));
        m_start_inds.push_back(i * m_items_per_thread);
        int record_count =
            i == nthreads - 1 ? (batch_size - i * m_items_per_thread) : m_items_per_thread;
        m_end_inds.push_back(m_start_inds[i] + record_count);
    }

    auto oshapes         = m_providers[0]->get_output_shapes();
    m_number_elements_in = m_providers[0]->get_input_count();

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
    m_state                      = async_state::wait_for_buffer;
    fixed_buffer_map*    outputs = get_pending_buffer();
    m_state                      = async_state::fetching_data;
    encoded_record_list* inputs  = m_source->next();
    m_state                      = async_state::processing;

    if (inputs == nullptr)
    {
        outputs = nullptr;
    }
    else
    {
        std::vector<std::thread> provider_threads;
        try
        {
            for (int id = 0; id < m_providers.size(); ++id)
            {
                provider_threads.emplace_back(&batch_decoder::work, this, id, inputs, outputs);
            }

            for (auto& t : provider_threads)
            {
                t.join();
            }
            // Now perform any potentially necessary whole-batch operation
            m_providers[0]->post_process(*outputs);
        }
        catch (std::exception&)
        {
            outputs = nullptr;
        }
    }

    m_state = async_state::idle;
    return outputs;
}

void batch_decoder::work(int id, encoded_record_list* in_buf, fixed_buffer_map* out_buf)
{
    // Thread function.
    // No locking required because threads write into non-overlapping regions.
    try
    {
        affirm(in_buf->size() != 0, "input buffer pool is empty.");

        for (int item_idx = m_start_inds[id]; item_idx < m_end_inds[id]; item_idx++)
        {
            m_providers[id]->provide(item_idx, *in_buf, *out_buf);
        }
    }
    catch (std::exception& e)
    {
        cout << "decode_thread_pool exception: " << e.what() << endl;
        // m_buffer_pool_decoded->write_exception(std::current_exception());
    }
}
