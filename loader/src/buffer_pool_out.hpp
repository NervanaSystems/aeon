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

#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstring>

#include "buffer_pool.hpp"
#include "buffer_out.hpp"

namespace nervana
{
    class buffer_pool_out;
}

// buffer_pool_out acts as our double buffer to hold data before copying to device
class nervana::buffer_pool_out : public nervana::buffer_pool
{
public:
    buffer_pool_out(const std::map<std::string, size_t>& writeSizes, size_t batchSize, bool pinned = false);
    virtual ~buffer_pool_out();
    buffer_out_array& get_write_buffer();
    buffer_out_array& get_read_buffer();

    void        switch_read_buffer();
    void        switch_write_buffer();
    bool        no_read_buffers();
    bool        has_read_buffers();
    bool        no_write_buffers();
    bool        has_write_buffers();

    std::mutex& get_mutex();
    void wait_for_available_read_buffer(std::unique_lock<std::mutex>& lock);
    void wait_for_available_write_buffer(std::unique_lock<std::mutex>& lock);
    void signal_available_read_buffer();
    void signal_available_write_buffer();

protected:
    int                               m_used  = 0;
    std::shared_ptr<buffer_out_array> m_bufs[2];
    std::mutex                        m_mutex;
    std::condition_variable           m_available_read_buffer;
    std::condition_variable           m_available_write_buffer;

};
