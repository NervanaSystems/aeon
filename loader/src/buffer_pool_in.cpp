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

#include "buffer_pool_in.hpp"

using namespace std;

buffer_pool_in::buffer_pool_in(int dataSize, int targetSize)
: _used(0), _readPos(0), _writePos(0) {
    for (int i = 0; i < _count; i++) {
        buffer_in* dataBuffer = new buffer_in(dataSize);
        buffer_in* targetBuffer = new buffer_in(targetSize);
        _bufs.push_back(buffer_in_array{dataBuffer, targetBuffer});
    }
}

buffer_pool_in::~buffer_pool_in() {
    for (auto buf : _bufs) {
        delete buf[0];
        delete buf[1];
    }
}

buffer_in_array& buffer_pool_in::getForWrite()
{
    _bufs[_writePos][0]->reset();
    _bufs[_writePos][1]->reset();
    return _bufs[_writePos];
}

buffer_in_array& buffer_pool_in::getForRead() {
    return _bufs[_readPos];
}

buffer_in_array& buffer_pool_in::getPair(int bufIdx) {
    assert(bufIdx >= 0 && bufIdx < _count);
    return _bufs[bufIdx];
}

void buffer_pool_in::advanceReadPos() {
    _used--;
    advance(_readPos);
}

void buffer_pool_in::advanceWritePos() {
    _used++;
    advance(_writePos);
}

bool buffer_pool_in::empty() {
    assert(_used >= 0);
    return (_used == 0);
}

bool buffer_pool_in::full() {
    assert(_used <= _count);
    return (_used == _count);
}

std::mutex& buffer_pool_in::getMutex() {
    return _mutex;
}

void buffer_pool_in::waitForNonEmpty(std::unique_lock<std::mutex>& lock) {
    _nonEmpty.wait(lock);
}

void buffer_pool_in::waitForNonFull(std::unique_lock<std::mutex>& lock) {
    _nonFull.wait(lock);
}

void buffer_pool_in::signalNonEmpty() {
    _nonEmpty.notify_all();
}

void buffer_pool_in::signalNonFull() {
    _nonFull.notify_all();
}

void buffer_pool_in::advance(int& index) {
    // increment index and reset to 0 when index hits `_count`
    if (++index == _count) {
        index = 0;
    }
}

