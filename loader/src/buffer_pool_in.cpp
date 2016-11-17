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

buffer_pool_in::buffer_pool_in(unsigned int nbuffers_in) :
    buffer_pool()
{
    for (int i = 0; i < m_count; i++) {
        m_bufs.push_back(make_shared<buffer_in_array>(nbuffers_in));
    }

}

buffer_pool_in::~buffer_pool_in() {}

buffer_in_array& buffer_pool_in::get_for_write()
{
    buffer_in_array& buf_ary = *m_bufs[m_write_pos];
    for (auto &b : buf_ary) {
        b->reset();
    }
    return buf_ary;
}

buffer_in_array& buffer_pool_in::get_for_read()
{
    reraise_exception();
    return *m_bufs[m_read_pos];
}

void buffer_pool_in::advance_read_pos()
{
    m_used--;
    advance(m_read_pos);
}

void buffer_pool_in::advance_write_pos()
{
    m_used++;
    advance(m_write_pos);
    clear_exception();
}

bool buffer_pool_in::empty()
{
    affirm(m_used >= 0, "buffer_pool_in used < 0");
    return (m_used == 0);
}

bool buffer_pool_in::full()
{
    affirm(m_used <= m_count, "buffer_pool_in used > count");
    return (m_used == m_count);
}

std::mutex& buffer_pool_in::get_mutex()
{
    return m_mutex;
}

void buffer_pool_in::wait_for_not_empty(std::unique_lock<std::mutex>& lock)
{
    m_non_empty.wait(lock);
}

void buffer_pool_in::wait_for_non_full(std::unique_lock<std::mutex>& lock)
{
    m_non_full.wait(lock);
}

void buffer_pool_in::signal_not_empty()
{
    m_non_empty.notify_all();
}

void buffer_pool_in::signal_not_full()
{
    m_non_full.notify_all();
}

void buffer_pool_in::advance(int& index)
{
    // increment index and reset to 0 when index hits `m_count`
    if (++index == m_count) {
        index = 0;
    }
}

