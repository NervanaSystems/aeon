/*
 Copyright 2017 Intel(R) Nervana(TM)
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
#include "batch_iterator.hpp"

using namespace std;
using namespace nervana;

batch_decoder::batch_decoder(shared_ptr<batch_iterator>                 b_itor,
                             size_t                                     batch_size,
                             uint32_t                                   thread_count,
                             bool                                       pinned,
                             const std::shared_ptr<provider_interface>& prov,
                             uint32_t                                   seed)
    : async_manager<encoded_record_list, fixed_buffer_map>(b_itor, "batch_decoder")
    , m_batch_size(batch_size)
    , m_provider(prov)
    , m_deterministic_mode(seed != 0)
{
    m_thread_pool =
        singleton<thread_pool_queue<batch_decoder, &batch_decoder::process>>::get(thread_count);
    m_number_elements_in = prov->get_input_count();

    // Allocate the space in the output buffers
    for (unsigned int k = 0; k < 2; ++k)
        m_containers[k].add_items(prov->get_output_shapes(), batch_size, pinned);

    if (m_deterministic_mode)
    {
        m_random.resize(batch_size);
        for_each(m_random.begin(), m_random.end(), [&](random_engine_t& eng) { eng.seed(seed++); });
    }
}

batch_decoder::~batch_decoder()
{
    finalize();
}

void batch_decoder::process(const int index)
{
    if (m_deterministic_mode)
        get_thread_local_random_engine() = m_random[index];

    m_provider->provide(index, *m_inputs, *m_outputs);

    if (m_deterministic_mode)
        m_random[index] = get_thread_local_random_engine();
}

fixed_buffer_map* batch_decoder::filler()
{
    m_state                     = async_state::wait_for_buffer;
    fixed_buffer_map* outputs   = get_pending_buffer();
    m_state                     = async_state::fetching_data;
    encoded_record_list* inputs = m_source->next();
    m_state                     = async_state::processing;

    m_iteration_number++;

    if (inputs == nullptr)
    {
        outputs = nullptr;
    }
    else
    {
        for (const encoded_record& record : *inputs)
        {
            record.rethrow_if_exception();
        }
        m_inputs  = inputs;
        m_outputs = outputs;
        m_thread_pool->run(this, m_batch_size);
    }
    m_state = async_state::idle;
    return outputs;
}
