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
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "python_backend.hpp"
#include "thread_pool.hpp"
#include "buffer_pool_in.hpp"
#include "buffer_pool_out.hpp"
#include "provider_interface.hpp"

namespace nervana
{
    class decode_thread_pool;
}

/* decode_thread_pool
 *
 * decode_thread_pool takes data from the BufferPool `in`, transforms it
 * using `count` threads with a Media::transform built from
 * `mediaParams`.  Each minibatch is transposed by a manager thread and
 * then copied to the `device`.
 *
 */
class nervana::decode_thread_pool : public nervana::thread_pool
{
public:
    decode_thread_pool(int count,
                       const std::shared_ptr<nervana::buffer_pool_in>& in,
                       const std::shared_ptr<nervana::buffer_pool_out>& out,
                       const std::shared_ptr<python_backend>& pbe);

    virtual ~decode_thread_pool();
    virtual void start() override;
    virtual void stop() override;
    void add_provider(std::shared_ptr<nervana::provider_interface> prov);

protected:
    virtual void run(int id) override;
    virtual void work(int id) override;
    void produce();
    void consume();
    void manage();

private:
    decode_thread_pool();
    decode_thread_pool(const decode_thread_pool&);

    std::vector<std::shared_ptr<nervana::provider_interface>> m_providers;
    std::shared_ptr<nervana::buffer_pool_in>                  m_buffer_pool_encoded;
    std::shared_ptr<nervana::buffer_pool_out>                 m_buffer_pool_decoded;
    std::shared_ptr<python_backend>                           m_python_backend;

    int                       m_items_per_thread;
    std::mutex                m_mutex;
    std::condition_variable   m_started;
    std::condition_variable   m_ended;
    int                       m_batch_size;
    std::atomic<int>          m_end_signaled{0};
    std::thread*              m_manager         = 0;
    bool                      m_stop_manager    = false;
    bool                      m_manager_stopped = false;
    int                       m_buffer_index    = 0;
    std::vector<int>          m_start_inds;
    std::vector<int>          m_end_inds;
    std::vector<std::atomic<bool>>          m_start_signaled;

};
