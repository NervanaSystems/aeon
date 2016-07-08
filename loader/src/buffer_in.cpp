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

#include "buffer_in.hpp"

using namespace std;

buffer_in::buffer_in(int size, bool pinned)
: _size(size), _idx(0), _alloc(true), _pinned(pinned) {
    _data = alloc();
    _cur = _data;
}

buffer_in::buffer_in(char* data, int size)
: _data(data), _size(size), _cur(_data), _idx(0), _alloc(false), _pinned(false) {
    pushItem(size);
}

buffer_in::~buffer_in() {
    if (_alloc == true) {
        dealloc(_data);
    }
}

void buffer_in::reset() {
    _cur = _data;
    _idx = 0;
    _items.clear();
    _lens.clear();
}

void buffer_in::shuffle(uint seed) {
    // TODO: instead of reseeding the shuffle, store these in a pair
    std::minstd_rand0 rand_items(seed);
    std::shuffle(_items.begin(), _items.end(), rand_items);

    std::minstd_rand0 rand_lens(seed);
    std::shuffle(_lens.begin(), _lens.end(), rand_lens);
}

void buffer_in::dump() {
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

void buffer_in::pushItem(int len) {
    _items.push_back(_idx);
    _lens.push_back(len);
    _cur += len;
    _idx += len;
}

char* buffer_in::getItem(int index, int& len) {
    if (index >= (int) _items.size()) {
        // TODO: why not raise exception here?  Is anyone actually
        // checking the return value of getItem to make sure it is
        // non-0?
        return 0;
    }
    len = _lens[index];
    return _data + _items[index];
}

int buffer_in::getItemCount() {
    return _items.size();
}

char* buffer_in::getCurrent() {
    return _cur;
}

uint buffer_in::getSize() {
    return _size;
}

uint buffer_in::getLevel() {
    return _idx;
}

void buffer_in::read(istream& is, int size) {
    // read `size` bytes out of `ifs` and push into buffer
    resizeIfNeeded(size);
    is.read(_cur, size);
    pushItem(size);
}

void buffer_in::read(const char* src, int size) {
    // read `size` bytes out of `src` and push into buffer
    resizeIfNeeded(size);
    memcpy((void *) _cur, (void *) src, size);
    pushItem(size);
}

void buffer_in::resizeIfNeeded(int inc) {
    if (getLevel() + inc > getSize()) {
        resize(inc);
    }
}

void buffer_in::resize(int inc) {
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

char* buffer_in::alloc() {
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

void buffer_in::dealloc(char* data) {
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
