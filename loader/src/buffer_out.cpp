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

#include "buffer_out.hpp"

using namespace std;
using namespace nervana;

buffer_out::buffer_out(size_t element_size, size_t minibatch_size, bool pinned)
    : _size(element_size * minibatch_size)
    , _batch_size(minibatch_size)
    , _pinned(pinned)
    , _stride(element_size)
    , _item_size(element_size)
{
    _data = alloc();
}

buffer_out::~buffer_out()
{
    dealloc(_data);
}

char* buffer_out::get_item(size_t index)
{
    size_t offset = index * _stride;
    if (index >= (int)_batch_size)
    {
        // TODO: why not raise exception here?  Is anyone actually
        // checking the return value of get_item to make sure it is
        // non-0?
        return 0;
    }
    return &_data[offset];
}

size_t buffer_out::get_item_count()
{
    return _size / _item_size;
}

size_t buffer_out::size()
{
    return _size;
}

char* buffer_out::alloc()
{
    char* data;
    if (_pinned == true)
    {
#if HAS_GPU
        CUresult status = cuMemAllocHost((void**)&data, _size);
        if (status != CUDA_SUCCESS)
        {
            throw std::bad_alloc();
        }
#else
        data = new char[_size];
#endif
    }
    else
    {
        data = new char[_size];
    }
    return data;
}

void buffer_out::dealloc(char* data)
{
    if (_pinned == true)
    {
#if HAS_GPU
        cuMemFreeHost(data);
#else
        delete[] data;
#endif
    }
    else
    {
        delete[] data;
    }
}
