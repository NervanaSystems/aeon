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

#include "buffer_out.hpp"

using namespace std;

buffer_out::buffer_out(int sizeof_type, int item_count, int minibatch_size, bool pinned) :
    _size(sizeof_type * item_count * minibatch_size),
    _pinned(pinned),
    _stride(sizeof_type * item_count),
    _item_size(sizeof_type * item_count)
{
    _data = alloc();
}

buffer_out::~buffer_out() {
    dealloc(_data);
}

char* buffer_out::getItem(int index, int& len) {
    size_t offset = index * _stride;
//    if (index >= (int) _items.size()) {
//        // TODO: why not raise exception here?  Is anyone actually
//        // checking the return value of getItem to make sure it is
//        // non-0?
//        return 0;
//    }
    len = _item_size;
    return &_data[offset];
}

int buffer_out::getItemCount() {
    return _size / _item_size;
}

uint buffer_out::getSize() {
    return _size;
}

char* buffer_out::alloc() {
    char*      data;
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

void buffer_out::dealloc(char* data) {
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
