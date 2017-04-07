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

#if HAS_GPU
#include <cuda.h>
#endif

#include <random>
#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>

#include "buffer_pool_in.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

buffer_pool_in::buffer_pool_in(unsigned int nbuffers_in)
    : buffer_pool()
{
    m_bufs[0] = make_shared<buffer_in_array>(nbuffers_in);
    m_bufs[1] = make_shared<buffer_in_array>(nbuffers_in);
}

buffer_pool_in::~buffer_pool_in()
{
}

buffer_in_array& buffer_pool_in::get_write_buffer()
{
    buffer_in_array& buf_ary = *m_bufs[m_write_pos];
    for (auto& b : buf_ary)
    {
        b->reset();
    }
    return buf_ary;
}

buffer_in_array& buffer_pool_in::get_read_buffer()
{
    reraise_exception();
    return *m_bufs[m_read_pos];
}

void buffer_pool_in::switch_read_buffer()
{
    m_used--;
    m_read_pos = (m_read_pos == 0) ? 1 : 0;
}

void buffer_pool_in::switch_write_buffer()
{
    m_used++;
    m_write_pos = (m_write_pos == 0) ? 1 : 0;
    clear_exception();
}

bool buffer_pool_in::no_read_buffers()
{
    affirm(m_used >= 0, "buffer_pool_in used < 0");
    return (m_used == 0);
}

bool buffer_pool_in::has_read_buffers()
{
    return no_read_buffers() == false;
}

bool buffer_pool_in::no_write_buffers()
{
    affirm(m_used <= 2, "buffer_pool_in used > count");
    return (m_used == 2);
}

bool buffer_pool_in::has_write_buffers()
{
    return no_write_buffers() == false;
}

std::mutex& buffer_pool_in::get_mutex()
{
    return m_mutex;
}

void buffer_pool_in::wait_for_available_read_buffer(std::unique_lock<std::mutex>& lock)
{
    m_available_read_buffer.wait(lock);
}

void buffer_pool_in::wait_for_available_write_buffer(std::unique_lock<std::mutex>& lock)
{
    m_available_write_buffer.wait(lock);
}

void buffer_pool_in::signal_available_read_buffer()
{
    m_available_read_buffer.notify_all();
}

void buffer_pool_in::signal_available_write_buffer()
{
    m_available_write_buffer.notify_all();
}
