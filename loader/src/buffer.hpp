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

#include <assert.h>
#if HAS_GPU
#include <cuda.h>
#endif

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>

#include "streams.hpp"

class Buffer {
public:
    explicit Buffer(int size, bool pinned = false)
    : _size(size), _idx(0), _alloc(true), _pinned(pinned) {
        _data = alloc();
        _cur = _data;
    }

    Buffer(char* data, int size)
    : _data(data), _size(size), _cur(_data), _idx(0), _alloc(false) {
    }

    virtual ~Buffer() {
        if (_alloc == true) {
            dealloc(_data);
        }
    }

    void reset() {
        _cur = _data;
        _idx = 0;
        _items.clear();
        _lens.clear();
    }

    void dump() {
        uint8_t* data = reinterpret_cast<uint8_t*>(_data);
        int len = _size;
        assert(len % 16 == 0);
        int index = 0;
        while (index < len) {
            printf("%08x", index);
            for (int i = 0; i < 8; i++) {
                printf(" %02x", data[i]);
            }
            printf("  ");
            for (int i = 8; i < 16; i++) {
                printf(" %02x", data[i]);
            }
            printf(" ");
            for (int i = 0; i < 16; i++) {
                printf("%c", (data[i] < 32)? '.' : data[i]);
            }
            printf("\n");
            data += 16;
            index += 16;
        }
    }

    void pushItem(int len) {
        _items.push_back(_idx);
        _lens.push_back(len);
        _cur += len;
        _idx += len;
    }

    char* getItem(int index, int& len) {
        if (index >= (int) _items.size()) {
            return 0;
        }
        len = _lens[index];
        return _data + _items[index];
    }

    int getItemCount() {
        return _items.size();
    }

    char* getCurrent() {
        return _cur;
    }

    uint getSize() {
        return _size;
    }

    uint getLevel() {
        return _idx;
    }

    void read(IfStream& ifs, int size) {
        resizeIfNeeded(size);
        ifs.read(_cur, size);
        pushItem(size);
    }

    void read(char* src, int size) {
        resizeIfNeeded(size);
        memcpy((void *) _cur, (void *) src, size);
        pushItem(size);
    }

private:
    void resizeIfNeeded(int inc) {
        if (getLevel() + inc > getSize()) {
            resize(inc);
        }
    }

    void resize(int inc) {
        assert(_alloc == true);
        _size = getLevel() + inc;
        // Allocate a bit more to minimize reallocations.
        _size += _size / 8;
        char* data = alloc();
        memcpy(data, _data, getLevel());
        dealloc(_data);
        _data = data;
        _cur = _data + _idx;
    }

    char* alloc() {
        char*      data;
        assert(_alloc == true);
        if (_pinned == true) {
#if HAS_GPU
            CUresult status = cuMemAllocHost((void**)&data, _size);
            if (status != CUDA_SUCCESS) {
                throw std::bad_alloc();
            }
#else
            data = new char[_size];
#endif
        } else {
            data = new char[_size];
        }
        return data;
    }

    void dealloc(char* data) {
        if (_pinned == true) {
#if HAS_GPU
            cuMemFreeHost(data);
#else
            delete[] data;
#endif
        } else {
            delete[] data;
        }
    }

public:
    char*                       _data;
    uint                        _size;

protected:
    char*                       _cur;
    int                         _idx;
    std::vector<int>            _items;
    std::vector<int>            _lens;
    bool                        _alloc;
    bool                        _pinned;
};

typedef std::pair<Buffer*, Buffer*>             BufferPair;

class BufferPool {
public:
    BufferPool(int dataSize, int targetSize, bool pinned = false, int count = 2);
    virtual ~BufferPool();
    BufferPair& getForWrite();
    BufferPair& getForRead();
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
    std::vector<BufferPair>     _bufs;
    int                         _readPos;
    int                         _writePos;
    std::mutex                  _mutex;
    std::condition_variable     _nonFull;
    std::condition_variable     _nonEmpty;
};
