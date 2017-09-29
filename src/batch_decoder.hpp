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

#pragma once

#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "provider_interface.hpp"
#include "provider_factory.hpp"
#include "thread_pool.hpp"

namespace nervana
{
    class batch_decoder;
    class batch_iterator;
}

class nervana::batch_decoder : public async_manager<encoded_record_list, fixed_buffer_map>
{
public:
    batch_decoder(batch_iterator*                            b_itor,
                  size_t                                     batch_size,
                  uint32_t                                   thread_count,
                  bool                                       pinned,
                  const std::shared_ptr<provider_interface>& prov,
                  uint32_t                                   seed = 0);

    virtual ~batch_decoder();

    virtual size_t            record_count() const override { return m_batch_size; }
    virtual size_t            elements_per_record() const override { return m_number_elements_out; }
    virtual fixed_buffer_map* filler() override;

    void register_info_handler(std::function<void(const fixed_buffer_map*)>& f)
    {
        m_info_handler = f;
    }

    void process(const int index);

private:
    size_t                                    m_batch_size;
    size_t                                    m_number_elements_in;
    size_t                                    m_number_elements_out;
    std::shared_ptr<const provider_interface> m_provider;
    encoded_record_list*                      m_inputs{nullptr};
    fixed_buffer_map*                         m_outputs{nullptr};
    std::shared_ptr<thread_pool_queue<batch_decoder, &batch_decoder::process>> m_thread_pool;
    std::function<void(const fixed_buffer_map*)> m_info_handler;
    size_t                                       m_iteration_number{0};
    std::vector<nervana::random_engine_t>        m_random;
    bool                                         m_deterministic_mode;
};
