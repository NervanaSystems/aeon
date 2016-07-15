/*
 Copyright 2015 Nervana Systems Inc.
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

class buffer_pool_in : public buffer_pool {
public:
    buffer_pool_in(const std::vector<size_t>& initial_sizes);
    virtual ~buffer_pool_in();
    buffer_in_array& getForWrite();
    buffer_in_array& getForRead();
    buffer_in_array& getPair(int bufIdx);  // mark for delete?
    int getCount() { return _bufs.size();}

    void advanceReadPos();
    void advanceWritePos();
    bool empty();
    bool full();
    std::mutex& getMutex();
    void waitForNonEmpty(std::unique_lock<std::mutex>& lock);
    void waitForNonFull(std::unique_lock<std::mutex>& lock);
    void signalNonEmpty();
    void signalNonFull();

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
