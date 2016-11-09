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
    for (int i = 0; i < _count; i++) {
        _bufs.push_back(make_shared<buffer_in_array>(nbuffers_in));
    }

}

buffer_pool_in::~buffer_pool_in() {}

buffer_in_array& buffer_pool_in::get_for_write()
{
    buffer_in_array& buf_ary = *_bufs[_writePos];
    for (auto &b : buf_ary) {
        b->reset();
    }
    return buf_ary;
}

buffer_in_array& buffer_pool_in::get_for_read()
{
    reraise_exception();
    return *_bufs[_readPos];
}

void buffer_pool_in::advance_read_pos()
{
    _used--;
    advance(_readPos);
}

void buffer_pool_in::advance_write_pos()
{
    _used++;
    advance(_writePos);
    clear_exception();
}

bool buffer_pool_in::empty()
{
    affirm(_used >= 0, "buffer_pool_in used < 0");
    return (_used == 0);
}

bool buffer_pool_in::full()
{
    affirm(_used <= _count, "buffer_pool_in used > count");
    return (_used == _count);
}

std::mutex& buffer_pool_in::get_mutex()
{
    return _mutex;
}

void buffer_pool_in::wait_for_not_empty(std::unique_lock<std::mutex>& lock)
{
    _nonEmpty.wait(lock);
}

void buffer_pool_in::wait_for_non_full(std::unique_lock<std::mutex>& lock)
{
    _nonFull.wait(lock);
}

void buffer_pool_in::signal_not_empty()
{
    _nonEmpty.notify_all();
}

void buffer_pool_in::signal_not_full()
{
    _nonFull.notify_all();
}

void buffer_pool_in::advance(int& index)
{
    // increment index and reset to 0 when index hits `_count`
    if (++index == _count) {
        index = 0;
    }
}

