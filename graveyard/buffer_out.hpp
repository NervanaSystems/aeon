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
#include <cstring>
#include <initializer_list>
#include <iostream>

#include "util.hpp"
#include "typemap.hpp"

#if HAS_GPU
#include <cuda.h>
#endif

namespace nervana
{
    class buffer_out;
    class buffer_out_array;
}

class nervana::buffer_out
{
public:
    explicit buffer_out(const std::string& name, size_t element_size, size_t batch_size, bool pinned = false);
    virtual ~buffer_out();

    const char* get_item(size_t index) const;
    char* get_item(size_t index);
    char*  data() { return m_data; }
    size_t record_count() const;
    size_t size() const;

private:
    buffer_out() = delete;
    char* alloc();
    void dealloc(char* data);

    char*  m_data;
    size_t m_size;
    size_t m_batch_size;
    bool   m_pinned;
    size_t m_stride;
    size_t m_item_size;
};

// in cases with (object, target) pairs, buffer_out is length 2
class nervana::buffer_out_array
{
public:
    buffer_out_array(const std::map<std::string, size_t>& write_sizes, size_t batch_size, bool pinned = false)
    {
        for (auto sz : write_sizes)
        {
            m_data.insert({sz.first, new buffer_out(sz.first, sz.second, batch_size, pinned)});
        }
    }

    buffer_out_array(const std::map<std::string, shape_type>& write_sizes, size_t batch_size, bool pinned = false)
    {
        for (auto sz : write_sizes)
        {
            m_data.insert({sz.first, new buffer_out(sz.first, sz.second.get_byte_size(), batch_size, pinned)});
        }
    }

    ~buffer_out_array()
    {
        for (auto buf : m_data)
        {
            delete buf.second;
        }
    }

    const buffer_out* operator[](const std::string& name) const
    {
        auto it = m_data.find(name);
        return (it == m_data.end() ? nullptr : it->second);
    }

    buffer_out* operator[](const std::string& name)
    {
        auto it = m_data.find(name);
        return (it == m_data.end() ? nullptr : it->second);
    }

    size_t size() const { return m_data.size(); }
private:
    // these must be defined because buffer_out_array[0] is resolved to call the string method
    const buffer_out* operator[](int) const = delete;
    buffer_out* operator[](int)             = delete;

    std::map<std::string, buffer_out*> m_data;
};
