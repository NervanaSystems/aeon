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

#include <assert.h>
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

#include "host_buffer_pool.hpp"
#include "buffer.hpp"

using namespace std;

host_buffer_pool::host_buffer_pool(int dataSize, int targetSize, bool pinned, int count)
: _count(count), _used(0), _readPos(0), _writePos(0) {
    for (int i = 0; i < count; i++) {
        Buffer* dataBuffer = new Buffer(dataSize, pinned);
        Buffer* targetBuffer = new Buffer(targetSize, pinned);
        _bufs.push_back(BufferArray{dataBuffer, targetBuffer});
    }
}

host_buffer_pool::~host_buffer_pool() {
    for (auto buf : _bufs) {
        delete buf[0];
        delete buf[1];
    }
}

BufferArray& host_buffer_pool::getForWrite()
{
    _bufs[_writePos][0]->reset();
    _bufs[_writePos][1]->reset();
    return _bufs[_writePos];
}

BufferArray& host_buffer_pool::getForRead() {
    return _bufs[_readPos];
}

BufferArray& host_buffer_pool::getPair(int bufIdx) {
    assert(bufIdx >= 0 && bufIdx < _count);
    return _bufs[bufIdx];
}

void host_buffer_pool::advanceReadPos() {
    _used--;
    advance(_readPos);
}

void host_buffer_pool::advanceWritePos() {
    _used++;
    advance(_writePos);
}

bool host_buffer_pool::empty() {
    assert(_used >= 0);
    return (_used == 0);
}

bool host_buffer_pool::full() {
    assert(_used <= _count);
    return (_used == _count);
}

std::mutex& host_buffer_pool::getMutex() {
    return _mutex;
}

void host_buffer_pool::waitForNonEmpty(std::unique_lock<std::mutex>& lock) {
    _nonEmpty.wait(lock);
}

void host_buffer_pool::waitForNonFull(std::unique_lock<std::mutex>& lock) {
    _nonFull.wait(lock);
}

void host_buffer_pool::signalNonEmpty() {
    _nonEmpty.notify_all();
}

void host_buffer_pool::signalNonFull() {
    _nonFull.notify_all();
}

void host_buffer_pool::advance(int& index) {
    // increment index and reset to 0 when index hits `_count`
    if (++index == _count) {
        index = 0;
    }
}

