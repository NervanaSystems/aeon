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
#include "batch_iterator.hpp"
#include "provider_interface.hpp"
#include "provider_factory.hpp"
#include "event.hpp"

namespace nervana
{
    class batch_decoder;
    class decode_thread_info;
}

class nervana::decode_thread_info
{
public:
    decode_thread_info(int                                        id,
                       int                                        start,
                       int                                        end,
                       const std::shared_ptr<provider_interface>& prov,
                       event&                                     done,
                       std::atomic<size_t>&                       active)
        : m_id{id}
        , m_start_index{start}
        , m_end_index{end}
        , m_provider{provider_factory::clone(prov)}
        , m_in_buf{nullptr}
        , m_out_buf{nullptr}
        , m_work_complete_event{done}
        , m_active_count{active}
        , m_thread_active{true}
    {
        m_thread = std::thread(&decode_thread_info::thread_entry, this);
    }

    void set_buffers(encoded_record_list* in, fixed_buffer_map* out)
    {
        m_in_buf  = in;
        m_out_buf = out;
        m_waiting_for_work_event.notify();
    }

    ~decode_thread_info()
    {
        m_thread_active = false;
        m_waiting_for_work_event.notify();
        m_thread.join();
    }

    const int                           m_id;
    const int                           m_start_index;
    const int                           m_end_index;
    std::shared_ptr<provider_interface> m_provider;
    encoded_record_list*                m_in_buf;
    fixed_buffer_map*                   m_out_buf;
    event                               m_waiting_for_work_event;
    event&                              m_work_complete_event;
    std::atomic<size_t>&                m_active_count;
    bool                                m_thread_active;
    std::thread                         m_thread;

private:
    void thread_entry()
    {
        while (m_thread_active)
        {
            m_waiting_for_work_event.wait();
            if (m_thread_active)
            {
                for (int index = m_start_index; index < m_end_index; index++)
                {
                    m_provider->provide(index, *m_in_buf, *m_out_buf);
                }
                size_t count = m_active_count.fetch_sub(1);
                if (count == 1) // last count is 1 means current count is 0
                {
                    m_work_complete_event.notify();
                }
            }
        }
    }
};

class nervana::batch_decoder : public async_manager<encoded_record_list, fixed_buffer_map>
{
public:
    batch_decoder(batch_iterator*                            b_itor,
                  size_t                                     batch_size,
                  uint32_t                                   thread_count,
                  bool                                       pinned,
                  const std::shared_ptr<provider_interface>& prov);

    virtual ~batch_decoder();

    virtual size_t            record_count() const override { return m_batch_size; }
    virtual size_t            elements_per_record() const override { return m_number_elements_out; }
    virtual fixed_buffer_map* filler() override;

    void register_info_handler(std::function<void(const fixed_buffer_map*)>& f)
    {
        m_info_handler = f;
    }

private:
    size_t m_batch_size;
    size_t m_number_elements_in;
    size_t m_number_elements_out;
    int    m_items_per_thread;

    std::function<void(const fixed_buffer_map*)> m_info_handler;

    std::vector<std::shared_ptr<decode_thread_info>> m_decode_thread_info;
    event                                            m_work_complete_event;
    std::atomic<size_t>                              m_active_count;
};
