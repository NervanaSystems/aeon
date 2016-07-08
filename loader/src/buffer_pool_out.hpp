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

#include "buffer_pool_out.hpp"
#include "buffer_out.hpp"

class buffer_pool_out {
public:
    buffer_pool_out(size_t dataSize, size_t targetSize, size_t batchSize, bool pinned = false);
    virtual ~buffer_pool_out();
    buffer_out_array& getForWrite();
    buffer_out_array& getForRead();
    buffer_out_array& getPair(int bufIdx);
    int getCount() { return _count;}

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
    int                         _count;
    int                         _used;
    std::vector<buffer_out_array> _bufs;
    int                         _readPos;
    int                         _writePos;
    std::mutex                  _mutex;
    std::condition_variable     _nonFull;
    std::condition_variable     _nonEmpty;
};
