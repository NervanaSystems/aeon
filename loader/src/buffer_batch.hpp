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
#include <utility>
#include <cstring>
#include <iostream>
#include <initializer_list>

#include "typemap.hpp"
#include "util.hpp"
#if HAS_GPU
#include <cuda.h>
#endif

namespace nervana
{
    // class buffer_batch;
    class buffer_variable_size_elements;
    class buffer_fixed_size_elements;
    class fixed_buffer_map;

    typedef std::vector<buffer_variable_size_elements> variable_buffer_array;
    typedef std::pair<std::vector<char>,std::exception_ptr> variable_record_field;
    typedef std::vector<nervana::variable_record_field> variable_record_field_list;

}

class nervana::buffer_variable_size_elements
{
public:
    buffer_variable_size_elements() {}
    virtual ~buffer_variable_size_elements() {}
    void read(std::istream& is, int size);
    void               reset() { m_buffers.clear(); }
    std::vector<char>& get_item (int index);

    void add_item(const std::vector<char>&);
    void add_item(std::vector<char>&&);
    void add_exception(std::exception_ptr);

    void shuffle(uint32_t random_seed);

    size_t size() const { return m_buffers.size(); }
    size_t get_item_count() { return size(); }

    typedef variable_record_field_list::iterator vrfl_iter;

    vrfl_iter begin() { return m_buffers.begin(); }
    vrfl_iter end() { return m_buffers.end(); }

    void append(std::move_iterator<vrfl_iter> first,
                std::move_iterator<vrfl_iter> last)
    {
        m_buffers.insert(m_buffers.end(), first, last);
    }

    void erase(vrfl_iter first, vrfl_iter last)
    {
        m_buffers.erase(first, last);
    }

private:
    nervana::variable_record_field_list m_buffers;
};


class nervana::buffer_fixed_size_elements
{
public:
    explicit buffer_fixed_size_elements() {}
    explicit buffer_fixed_size_elements(size_t element_size, size_t batch_size, bool pinned = false);

    virtual ~buffer_fixed_size_elements();

    virtual void allocate(size_t element_size, size_t batch_size, bool pinned = false);
    const char* get_item(size_t index) const;
    char* get_item(size_t index);
    char*  data() { return m_data; }
    size_t get_item_count() { return m_size / m_item_size; }
    size_t size() { return m_size; }

protected:
    char*  m_data{nullptr};
    size_t m_size{0};
    size_t m_batch_size{0};
    size_t m_stride{0};
    size_t m_item_size{0};
    bool   m_pinned{false};
};


class nervana::fixed_buffer_map
{
public:
    fixed_buffer_map() {}

    void add_item(const std::string &name, size_t element_size, size_t batch_size, bool pinned = false)
    {
        m_data.insert({name, new buffer_fixed_size_elements(element_size, batch_size, pinned)});
    }

    fixed_buffer_map(const std::map<std::string, size_t>& write_sizes, size_t batch_size, bool pinned = false)
    {
        for (auto sz : write_sizes)
        {
            add_item(sz.first, sz.second, batch_size, pinned);
        }
    }

    fixed_buffer_map(const std::map<std::string, shape_type>& write_sizes, size_t batch_size, bool pinned = false)
    {
        for (auto sz : write_sizes)
        {
            add_item(sz.first, sz.second.get_byte_size(), batch_size, pinned);
        }
    }

    ~fixed_buffer_map()
    {
        for (auto buf : m_data)
        {
            delete buf.second;
        }
    }

    const buffer_fixed_size_elements* operator[](const std::string& name) const
    {
        auto it = m_data.find(name);
        return (it == m_data.end() ? nullptr : it->second);
    }

    buffer_fixed_size_elements* operator[](const std::string& name)
    {
        auto it = m_data.find(name);
        return (it == m_data.end() ? nullptr : it->second);
    }

    size_t size() const
    {
        return m_data.size();
    }
private:
    // these must be defined because fixed_buffer_map[0] is resolved to call the string method
    const buffer_fixed_size_elements* operator[](int) const = delete;
    buffer_fixed_size_elements* operator[](int) = delete;

    std::map<std::string, buffer_fixed_size_elements*> m_data;
};
