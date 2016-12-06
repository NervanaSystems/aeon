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
    std::string cache_directory     = "";
    int         block_size     = 0;
    float       subset_fraction     = 1.0;
    bool        shuffle_every_epoch = false;
    bool        shuffle_manifest    = false;
    bool        single_thread       = false;
    bool        pinned              = false;
    int         random_seed         = 0;

    loader_config(nlohmann::json js)
    {
        if (js.is_null())
        {
            throw std::runtime_error("missing loader config in json config");
        }

        for (auto& info : config_list)
        {
            info->parse(js);
        }
        verify_config("loader", config_list, js);

        if (block_size == 0)
        {
            block_size = 3 * batch_size;
        }

        set_global_random_seed(random_seed);
        validate();
    }

private:
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
    };

    loader_config() {}
    bool validate() { return true; }
};


class nervana::loader_async : public nervana::async_manager<nervana::variable_buffer_array, nervana::fixed_buffer_map>
{
public:
    loader_async(batch_iterator_async* b_itor, const std::string& config_string);

    virtual ~loader_async() { finalize(); }
    virtual size_t object_count() override { return m_batch_size; }
    virtual size_t element_count() override { return m_number_elements_out; }

    virtual nervana::fixed_buffer_map* filler() override;

private:
    void work(int id, nervana::variable_buffer_array* in_buf, nervana::fixed_buffer_map* out_buf);

    std::vector<std::shared_ptr<nervana::provider_interface>> m_providers;
    uint32_t                        m_batch_size;
    size_t                          m_number_elements_in;
    size_t                          m_number_elements_out;
    int                             m_items_per_thread;
    std::vector<int>                m_start_inds;
    std::vector<int>                m_end_inds;
    nlohmann::json                  m_lcfg_json;
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

    const std::vector<std::string>& get_buffer_names() const;
    const shape_t& get_shape(const std::string& name) const;

    void set_iterator_count(BatchMode count);
    void set_iterator_count(size_t count);

    int itemCount() { return m_block_loader->object_count(); }

    // member typedefs provided through inheriting from std::iterator
    class iterator : public std::iterator<std::input_iterator_tag, // iterator_category
                                          fixed_buffer_map         // value_type
                                          // long,                    // difference_type
                                          // const buffer_out*,       // pointer
                                          // buffer_out               // reference
                                          >
    {
        //        long num = FROM;
        friend class loader;

    public:
        explicit iterator(loader& ld, bool m_is_end, size_t num_inputs);
        ~iterator();
        iterator& operator++(); // {num = TO >= FROM ? num + 1: num - 1; return *this;}
        iterator& operator++(int);
        bool operator==(const iterator& other) const; // {return num == other.num;}
        bool operator!=(const iterator& other) const; // {return !(*this == other);}
        const fixed_buffer_map& operator*() const;    // {return num;}
    private:
        iterator() = delete;
        iterator(const iterator&);
        // void read_input();
        // void fill_buffer();

        loader&                          m_current_loader;
        const bool                       m_is_end;
        // size_t                           m_num_inputs;
        // std::future<void>                m_async_result;
        fixed_buffer_map*                m_output_buffer_ptr{nullptr};
        size_t                           m_object_count;
        size_t                           m_position;
        // variable_buffer_array            m_pending_block;
        BatchMode                        m_batch_mode;
    };

    iterator begin();
    iterator end();

private:
    loader() = delete;

    friend class nervana::loader::iterator;

    bool                                m_single_thread_mode = false;
    std::shared_ptr<manifest_csv>       m_manifest;
    std::shared_ptr<block_loader_file_async>       m_block_loader;
    std::shared_ptr<batch_iterator_async>     m_batch_iterator;
    std::shared_ptr<loader_async>       m_decoder;
    std::shared_ptr<provider_interface> m_provider;
    int                                 m_batch_size;
    std::map<std::string, size_t>       m_out_sizes;
    BatchMode                           m_batch_mode;
    size_t                              m_batch_count_value;
};

class nervana::dataset_builder
{
public:
    dataset_builder& config(const std::string&);
    dataset_builder& batch_size(size_t);
    dataset_builder& batch_count(loader::BatchMode);
    dataset_builder& batch_count(size_t);
    loader create();
private:
    loader::BatchMode      m_batch_mode;
    size_t          m_batch_count_value;
    size_t          m_batch_size;
    std::string     m_config;
};
