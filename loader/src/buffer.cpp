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

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "buffer.hpp"
#include "streams.hpp"

 using namespace std;

BufferPool::BufferPool(int dataSize, int targetSize, bool pinned, int count)
: _count(count), _used(0), _readPos(0), _writePos(0) {
    for (int i = 0; i < count; i++) {
        Buffer* dataBuffer = new Buffer(dataSize, pinned);
        Buffer* targetBuffer = new Buffer(targetSize, pinned);
        _bufs.push_back(make_pair(dataBuffer, targetBuffer));
    }
}

BufferPool::~BufferPool() {
    for (auto buf : _bufs) {
        delete buf.first;
        delete buf.second;
    }
}

BufferPair& BufferPool::getForWrite() {
    _bufs[_writePos].first->reset();
    _bufs[_writePos].second->reset();
    return _bufs[_writePos];
}

BufferPair& BufferPool::getForRead() {
    return _bufs[_readPos];
}

void BufferPool::advanceReadPos() {
    _used--;
    advance(_readPos);
}

void BufferPool::advanceWritePos() {
    _used++;
    advance(_writePos);
}

bool BufferPool::empty() {
    assert(_used >= 0);
    return (_used == 0);
}

bool BufferPool::full() {
    assert(_used <= _count);
    return (_used == _count);
}

std::mutex& BufferPool::getMutex() {
    return _mutex;
}

void BufferPool::waitForNonEmpty(std::unique_lock<std::mutex>& lock) {
    _nonEmpty.wait(lock);
}

void BufferPool::waitForNonFull(std::unique_lock<std::mutex>& lock) {
    _nonFull.wait(lock);
}

void BufferPool::signalNonEmpty() {
    _nonEmpty.notify_all();
}

void BufferPool::signalNonFull() {
    _nonFull.notify_all();
}

void BufferPool::advance(int& index) {
    if (++index == _count) {
        index = 0;
    }
}

