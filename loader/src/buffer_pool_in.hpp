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
#include "buffer_in.hpp"

namespace nervana
{
    class buffer_pool_in;
}

class nervana::buffer_pool_in : public nervana::buffer_pool
{
public:
    buffer_pool_in(unsigned int nbuffers_in);
    virtual ~buffer_pool_in();
    buffer_in_array& get_for_write();
    buffer_in_array& get_for_read();

    void advance_read_pos();
    void advance_write_pos();
    bool empty();
    bool full();
    std::mutex& get_mutex();
    void wait_for_not_empty(std::unique_lock<std::mutex>& lock);
    void wait_for_non_full(std::unique_lock<std::mutex>& lock);
    void signal_not_empty();
    void signal_not_full();

protected:
    void advance(int& index);

protected:
    static constexpr int        _count = 2;
    int                         _used = 0;
    std::vector<std::shared_ptr<buffer_in_array>> _bufs;
    std::mutex                  _mutex;
    std::condition_variable     _nonFull;
    std::condition_variable     _nonEmpty;
};
