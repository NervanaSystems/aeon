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

#include "python_backend.hpp"
#include "thread_pool_read.hpp"
#include "thread_pool_decode.hpp"
#include "block_loader.hpp"
#include "block_iterator.hpp"
#include "batch_iterator.hpp"
#include "manifest.hpp"
#include "provider_factory.hpp"
#include "buffer_pool_in.hpp"
#include "buffer_pool_out.hpp"
#include "util.hpp"

namespace nervana
{
    class loader_config;
    class loader;
    class dataset_builder;
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
    int         minibatch_size;

    std::string type;
    std::string cache_directory     = "";
    int         macrobatch_size     = 0;
    float       subset_fraction     = 1.0;
    bool        shuffle_every_epoch = false;
    bool        shuffle_manifest    = false;
    bool        single_thread       = false;
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

        if (macrobatch_size == 0)
        {
            macrobatch_size = minibatch_size;
        }

        set_global_random_seed(random_seed);
        validate();
    }

private:
    std::vector<std::shared_ptr<nervana::interface::config_info_interface>> config_list = {
        ADD_SCALAR(type, mode::REQUIRED),
        ADD_SCALAR(manifest_filename, mode::REQUIRED),
        ADD_SCALAR(manifest_root, mode::OPTIONAL),
        ADD_SCALAR(minibatch_size, mode::REQUIRED),
        ADD_SCALAR(cache_directory, mode::OPTIONAL),
        ADD_SCALAR(macrobatch_size, mode::OPTIONAL),
        ADD_SCALAR(subset_fraction, mode::OPTIONAL, [](decltype(subset_fraction) v) { return v <= 1.0 && v >= 0.0; }),
        ADD_SCALAR(shuffle_every_epoch, mode::OPTIONAL),
        ADD_SCALAR(shuffle_manifest, mode::OPTIONAL),
        ADD_SCALAR(single_thread, mode::OPTIONAL),
        ADD_SCALAR(random_seed, mode::OPTIONAL),
    };

    loader_config() {}
    bool validate() { return true; }
};

/* loader
 *
 * The loader instantiates and then coordinates the effort of loading ingested data, caching
 * blocks of it in contiguous disk (using cpio file format), transforming the data and finally
 * loading the data into device memory
*/

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
                                          buffer_out_array         // value_type
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
        const buffer_out_array& operator*() const;    // {return num;}
    private:
        iterator() = delete;
        iterator(const iterator&);
        void read_input();
        void fill_buffer();

        loader&                          m_current_loader;
        const bool                       m_is_end;
        size_t                           m_num_inputs;
        std::future<void>                m_async_result;
        buffer_out_array                 m_output_buffer;
        size_t                           m_object_count;
        size_t                           m_position;
        buffer_in_array                  m_pending_block;
        BatchMode                        m_batch_mode;
    };

    iterator begin();
    iterator end();

private:
    loader() = delete;

    friend class nervana::loader::iterator;

    bool                                m_single_thread_mode = false;
    std::shared_ptr<block_loader>       m_block_loader;
    std::shared_ptr<batch_iterator>     m_batch_iterator;
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
