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
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>
#include <map>
#include <future>

#include "manifest.hpp"
#include "provider_factory.hpp"

#include "async_manager.hpp"
#include "buffer_batch.hpp"
#include "batch_iterator_async.hpp"
#include "block_loader_file_async.hpp"
#include "block_manager_async.hpp"
#include "log.hpp"
#include "util.hpp"

namespace nervana
{
    class loader_config;
    class loader;
    class dataset_builder;
    class loader_async;
}

/* decode_thread_pool
 *
 * decode_thread_pool takes data from the BufferPool `in`, transforms it
 * using `count` threads with a Media::transform built from
 * `mediaParams`.  Each minibatch is transposed by a manager thread and
 * then copied to the `device`.
 *
 */

class nervana::loader_config : public nervana::interface::config
{
public:
    std::string manifest_filename;
    std::string manifest_root;
    int         batch_size;

    std::string type;
    std::string cache_directory      = "";
    int         block_size           = 0;
    float       subset_fraction      = 1.0;
    bool        shuffle_every_epoch  = false;
    bool        shuffle_manifest     = false;
    bool        single_thread        = false;
    bool        pinned               = false;
    int         random_seed          = 0;
    std::string iteration_mode       = "ONCE";
    int         iteration_mode_count = 0;

    loader_config(nlohmann::json js);

private:
    loader_config() {}
    std::vector<std::shared_ptr<nervana::interface::config_info_interface>> config_list = {
        ADD_SCALAR(type, mode::REQUIRED),
        ADD_SCALAR(manifest_filename, mode::REQUIRED),
        ADD_SCALAR(manifest_root, mode::OPTIONAL),
        ADD_SCALAR(batch_size, mode::REQUIRED),
        ADD_SCALAR(cache_directory, mode::OPTIONAL),
        ADD_SCALAR(block_size, mode::OPTIONAL),
        ADD_SCALAR(subset_fraction, mode::OPTIONAL, [](decltype(subset_fraction) v) { return v <= 1.0 && v >= 0.0; }),
        ADD_SCALAR(shuffle_every_epoch, mode::OPTIONAL),
        ADD_SCALAR(shuffle_manifest, mode::OPTIONAL),
        ADD_SCALAR(single_thread, mode::OPTIONAL),
        ADD_SCALAR(pinned, mode::OPTIONAL),
        ADD_SCALAR(random_seed, mode::OPTIONAL),
        ADD_SCALAR(iteration_mode, mode::OPTIONAL),
        ADD_SCALAR(iteration_mode_count, mode::OPTIONAL)};

    void validate();
};

class nervana::loader_async : public async_manager<variable_buffer_array, fixed_buffer_map>
{
public:
    loader_async(batch_iterator_async* b_itor, size_t batch_size, bool single_thread, bool pinned,
                 const std::shared_ptr<provider_interface>& prov);

    virtual ~loader_async();

    virtual size_t                     record_count() const override { return m_batch_size; }
    virtual size_t                     element_count() const override { return m_number_elements_out; }
    virtual fixed_buffer_map* filler() override;

private:
    void work(int id, variable_buffer_array* in_buf, fixed_buffer_map* out_buf);

    std::vector<std::shared_ptr<provider_interface>> m_providers;
    size_t                                           m_batch_size;
    size_t                                           m_number_elements_in;
    size_t                                           m_number_elements_out;
    int                                              m_items_per_thread;
    std::vector<int>                                 m_start_inds;
    std::vector<int>                                 m_end_inds;
};

class nervana::loader
{
public:
    enum class BatchMode
    {
        INFINITE,
        ONCE,
        COUNT
    };

    loader(const std::string&);
    loader(nlohmann::json&);

    const std::vector<std::string>& get_buffer_names() const;
    const std::map<std::string, shape_type>& get_names_and_shapes() const;
    const shape_t& get_shape(const std::string& name) const;

    int record_count() { return m_manifest->record_count(); }
    // member typedefs provided through inheriting from std::iterator
    class iterator : public std::iterator<std::input_iterator_tag,     // iterator_category
                                          fixed_buffer_map             // value_type
                                          // long,                     // difference_type
                                          // const fixed_buffer_map*,  // pointer
                                          // fixed_buffer_map          // reference
                                          >
    {
        friend class loader;

    public:
        explicit iterator(loader& ld, bool is_end);
        iterator(const iterator&);
        ~iterator() {}
        iterator& operator++(); // {num = TO >= FROM ? num + 1: num - 1; return *this;}
        iterator& operator++(int);
        bool operator==(const iterator& other) const; // {return num == other.num;}
        bool operator!=(const iterator& other) const; // {return !(*this == other);}
        const fixed_buffer_map& operator*() const;    // {return num;}
        const size_t& position() const { return m_current_loader.m_position; }
        bool positional_end() const;

    private:
        iterator() = delete;

        loader&           m_current_loader;
        const bool        m_is_end;
    };

    // Note that these are returning COPIES
    iterator begin()
    {
        reset();
        return m_current_iter;
    }

    iterator end()
    {
        return m_end_iter;
    }

    // These are returning references
    iterator& get_current_iter() { return m_current_iter; }
    iterator& get_end_iter() { return m_end_iter; }

    void reset()
    {
        m_decoder->reset();
        m_output_buffer_ptr = m_decoder->next();
        m_position = 0;
    }

private:
    loader() = delete;
    void initialize(nlohmann::json& config_json);
    void increment_position();

    friend class nervana::loader::iterator;

    iterator                                 m_current_iter;
    iterator                                 m_end_iter;
    std::shared_ptr<manifest_csv>            m_manifest;
    std::shared_ptr<block_loader_file_async> m_block_loader;
    std::shared_ptr<block_manager_async>     m_block_manager;
    std::shared_ptr<batch_iterator_async>    m_batch_iterator;
    std::shared_ptr<provider_interface>      m_provider;
    std::shared_ptr<loader_async>            m_decoder;
    int                                      m_batch_size;
    BatchMode                                m_batch_mode;
    size_t                                   m_batch_count_value;
    size_t                                   m_position{0};
    fixed_buffer_map*                        m_output_buffer_ptr{nullptr};
};
