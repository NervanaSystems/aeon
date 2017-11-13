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
#include <opencv2/core/core.hpp>
#include <tuple>

#include "typemap.hpp"
#include "util.hpp"
#if HAS_GPU
#include <cuda.h>
#endif

namespace nervana
{
    // class buffer_batch;
    class buffer_fixed_size_elements;
    class fixed_buffer_map;
    class encoded_record;
    class encoded_record_list;

    typedef std::vector<char>                           variable_record_field;
    typedef std::vector<nervana::variable_record_field> variable_record_field_list;
}

class nervana::encoded_record
{
    friend class encoded_record_list;

public:
    variable_record_field& element(size_t index);
    const variable_record_field& element(size_t index) const;
    size_t size() const { return m_elements.size(); }
    void add_element(const void* data, size_t size)
    {
        std::vector<char> tmp(size);
        const char*       p = (const char*)data;
        for (size_t i = 0; i < size; i++)
        {
            tmp[i] = p[i];
        }
        m_elements.emplace_back(tmp);
    }

    void add_element(const std::vector<char>& data) { m_elements.emplace_back(data); }
    void add_element(std::vector<char>&& data) { m_elements.emplace_back(std::move(data)); }
    void add_exception(std::exception_ptr e) { m_exception = e; }
    variable_record_field_list::iterator  begin() { return m_elements.begin(); }
    variable_record_field_list::iterator  end() { return m_elements.end(); }
    void                                  rethrow_if_exception() const
    {
        if (m_exception != nullptr)
        {
            std::rethrow_exception(m_exception);
        }
    }

private:
    variable_record_field_list m_elements;
    std::exception_ptr         m_exception;
};

class nervana::encoded_record_list
{
public:
    encoded_record& record(size_t index)
    {
        encoded_record& rc = m_records[index];
        rc.rethrow_if_exception();
        return rc;
    }

    const encoded_record& record(size_t index) const
    {
        const encoded_record& rc = m_records[index];
        rc.rethrow_if_exception();
        return rc;
    }

    void add_record(const encoded_record& buffer)
    {
        verify(buffer);
        m_records.push_back(buffer);
    }

    void add_record(encoded_record&& buffer)
    {
        verify(buffer);
        m_records.push_back(std::move(buffer));
    }

    size_t size() const { return m_records.size(); }
    size_t elements_per_record() const { return m_elements_per_record; }
    void swap(encoded_record_list& other) { m_records.swap(other.m_records); }
    void move_to(encoded_record_list& target, size_t count)
    {
        auto begin = m_records.begin();
        auto end   = begin + count;

        std::move(begin, end, std::back_inserter(target.m_records));
        m_records.erase(begin, end);
    }

    void                                        clear() { m_records.clear(); }
    std::vector<encoded_record>::iterator       begin() { return m_records.begin(); }
    std::vector<encoded_record>::iterator       end() { return m_records.end(); }
    std::vector<encoded_record>::const_iterator begin() const { return m_records.begin(); }
    std::vector<encoded_record>::const_iterator end() const { return m_records.end(); }
    void shuffle(uint32_t random_seed)
    {
        std::minstd_rand0 rand_items(random_seed);
        std::shuffle(m_records.begin(), m_records.end(), rand_items);
    }

private:
    void verify(const encoded_record& buffer)
    {
        if (buffer.m_exception != nullptr)
        {
        }
        else if (m_elements_per_record == -1)
        {
            m_elements_per_record = buffer.size();
        }
        else if (buffer.size() != m_elements_per_record)
        {
            throw std::runtime_error("all records must have the same number of elements");
        }
    }

    std::vector<encoded_record> m_records;
    size_t                      m_elements_per_record = -1;
};

class nervana::buffer_fixed_size_elements
{
public:
    explicit buffer_fixed_size_elements(const shape_type& shp_tp,
                                        size_t            batch_size,
                                        bool              pinned = false);

    virtual ~buffer_fixed_size_elements();

    explicit buffer_fixed_size_elements(const buffer_fixed_size_elements&);

    virtual void allocate();
    const char* get_item(size_t index) const;
    char* get_item(size_t index);
    cv::Mat get_item_as_mat(size_t index, bool channel_major = false) const;
    char*             data() const { return m_data; }
    size_t            get_item_count() const { return m_size / m_stride; }
    size_t            size() const { return m_size; }
    size_t            get_stride() const { return m_stride; }
    const shape_type& get_shape_type() const { return m_shape_type; }
protected:
    buffer_fixed_size_elements() = delete;

    char*      m_data{nullptr};
    shape_type m_shape_type;
    size_t     m_size{0};
    size_t     m_batch_size{0};
    size_t     m_stride{0};
    bool       m_pinned{false};
};

class nervana::fixed_buffer_map
{
public:
    fixed_buffer_map() {}
    fixed_buffer_map(const std::vector<std::pair<std::string, shape_type>>& write_sizes,
                     size_t batch_size,
                     bool   pinned = false)
    {
        add_items(write_sizes, batch_size, pinned);
    }

    void add_items(const std::vector<std::pair<std::string, shape_type>>& write_sizes,
                   size_t batch_size,
                   bool   pinned = false)
    {
        for (auto sz : write_sizes)
        {
            add_item(std::get<0>(sz), std::get<1>(sz), batch_size, pinned);
        }
    }

    void add_item(const std::string& name,
                  const shape_type&  shp_tp,
                  size_t             batch_size,
                  bool               pinned = false)
    {
        m_names.push_back(name);
        m_data.emplace_back(
            std::make_pair(name, new buffer_fixed_size_elements(shp_tp, batch_size, pinned)));
    }

    ~fixed_buffer_map()
    {
        for (auto buf : m_data)
        {
            delete buf.second;
        }
    }

    const std::vector<std::string>& get_names() { return m_names; }
    const buffer_fixed_size_elements* operator[](const std::string& name) const
    {
        auto it = std::find_if(m_data.begin(), m_data.end(), [&](decltype(*m_data.begin())& v) {
            return v.first == name;
        });
        return (it == m_data.end() ? nullptr : it->second);
    }

    buffer_fixed_size_elements* operator[](const std::string& name)
    {
        auto it = std::find_if(m_data.begin(), m_data.end(), [&](decltype(*m_data.begin())& v) {
            return v.first == name;
        });
        return (it == m_data.end() ? nullptr : it->second);
    }

    void copy(fixed_buffer_map& src,
              size_t            src_index,
              size_t            dst_index,
              size_t            count,
              size_t            batch_size,
              bool              transpose);

    size_t size() const { return m_data.size(); }
private:
    fixed_buffer_map(const fixed_buffer_map&) = delete;

    // these must be defined because fixed_buffer_map[0] is resolved to call the string method
    const buffer_fixed_size_elements* operator[](int) const = delete;
    buffer_fixed_size_elements* operator[](int)             = delete;
    std::vector<std::string> m_names;
    std::vector<std::pair<std::string, buffer_fixed_size_elements*>> m_data;
};
