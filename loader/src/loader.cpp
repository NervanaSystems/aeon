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

#include <vector>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>
#include <sox.h>
#include <memory>

#include "loader.hpp"
#include "block_loader_cpio_cache.hpp"
#include "block_iterator_sequential.hpp"
#include "block_iterator_shuffled.hpp"
#include "batch_iterator.hpp"
#include "manifest_nds.hpp"
#include "block_loader_nds.hpp"

using namespace std;
using namespace nervana;

loader::loader(const std::string& config_string)
{
    auto config_json = nlohmann::json::parse(config_string);
    std::cout << __FILE__ << " " << __LINE__ << " " << config_json.dump(4) << std::endl;

    loader_config lcfg(config_json);
    m_batch_size                       = lcfg.minibatch_size;
    m_single_thread_mode               = lcfg.single_thread;
    shared_ptr<manifest> base_manifest;
    sox_format_init();

    if (manifest_nds::is_likely_json(lcfg.manifest_filename))
    {
        affirm(lcfg.subset_fraction == 1, "subset_fraction must be 1.0 for nds");

        auto manifest = make_shared<manifest_nds>(lcfg.manifest_filename);

        // TODO: add shard_count/shard_index to cfg
        m_block_loader =
            make_shared<block_loader_nds>(manifest->baseurl, manifest->token, manifest->collection_id, lcfg.macrobatch_size);

        base_manifest = manifest;
    }
    else
    {
        // the manifest defines which data should be included in the dataset
        auto manifest = make_shared<manifest_csv>(lcfg.manifest_filename, lcfg.shuffle_manifest, lcfg.manifest_root);

        // TODO: make the constructor throw this error
        if (manifest->object_count() == 0)
        {
            throw std::runtime_error("manifest file is empty");
        }

        m_block_loader = make_shared<block_loader_file>(manifest, lcfg.subset_fraction, lcfg.macrobatch_size);
        base_manifest  = manifest;
    }

    if (lcfg.cache_directory.length() > 0)
    {
        string cache_id = base_manifest->cache_id() + to_string(m_block_loader->object_count());
        m_block_loader =
            make_shared<block_loader_cpio_cache>(lcfg.cache_directory, cache_id, base_manifest->version(), m_block_loader);
    }

    shared_ptr<block_iterator> block_iter;
    if (lcfg.shuffle_every_epoch)
    {
        block_iter = make_shared<block_iterator_shuffled>(m_block_loader);
    }
    else
    {
        block_iter = make_shared<block_iterator_sequential>(m_block_loader);
    }

    m_batch_iterator = make_shared<batch_iterator>(block_iter, lcfg.minibatch_size);

    auto                      media   = provider_factory::create(config_json);

    for (auto shape : media->get_output_shapes())
    {
        m_out_sizes.insert({shape.first, shape.second.get_byte_size()});
    }

    m_provider = provider_factory::create(config_json);
    std::cout << __FILE__ << " " << __LINE__ << std::endl;
}

const vector<string>& loader::get_buffer_names() const
{
    return m_provider->get_buffer_names();
}

const shape_t& loader::get_shape(const string& name) const
{
    return m_provider->get_output_shape(name).get_shape();
}

void loader::set_iterator_count(BatchMode count)
{
    m_batch_mode = count;
    m_batch_count_value = 0;
}

void loader::set_iterator_count(size_t count)
{
    m_batch_mode = BatchMode::COUNT;
    m_batch_count_value = count;
}

loader::iterator::iterator(loader& ld, bool is_end, size_t num_inputs)
    : m_current_loader(ld)
    , m_is_end{is_end}
    , m_num_inputs{num_inputs}
    , m_output_buffer{ld.m_out_sizes, (size_t)ld.m_batch_size}
    , m_object_count{ld.m_block_loader->object_count()}
    , m_pending_block{(uint32_t)m_num_inputs}
    , m_batch_mode{ld.m_batch_mode}
{
    if (m_is_end)
    {
        if (m_batch_mode == loader::BatchMode::COUNT)
        {
            m_position = ld.m_batch_count_value;
        }
        else
        {
            m_position = m_object_count;
        }
    }
    else
    {
        m_position = 0;
    }

    // variable size buffers for reading encoded data (start off zero and grow as needed)
    if (!m_is_end)
    {
        m_async_result = async(&loader::iterator::read_input, this);
        fill_buffer();
    }
}

loader::iterator::iterator(const iterator& other)
    : m_current_loader{other.m_current_loader}
    , m_is_end{other.m_is_end}
    , m_num_inputs{other.m_num_inputs}
    , m_output_buffer{other.m_output_buffer}
    , m_object_count{other.m_object_count}
    , m_position{other.m_position}
    , m_pending_block{(uint32_t)m_num_inputs}
    , m_batch_mode{other.m_batch_mode}
{
    cout << __FILE__ << " " << __LINE__ << " iterator copy ctor" << endl;
}

loader::iterator::~iterator()
{
}

void loader::iterator::read_input()
{
    m_pending_block.clear();
    m_current_loader.m_batch_iterator->read(m_pending_block);
}

void loader::iterator::fill_buffer()
{
    if (!m_is_end)
    {
        m_async_result.get();
        buffer_in_array& input_buffer = m_pending_block;

        if (m_current_loader.m_provider)
        {
            for (int i = 0; i < m_current_loader.m_batch_size; i++)
            {
                m_current_loader.m_provider->provide(i, input_buffer, m_output_buffer);
            }
        }
        m_async_result = async(&loader::iterator::read_input, this);
    }
}

loader::iterator& loader::iterator::operator++()
{
    fill_buffer();
    m_position++;
    if (m_batch_mode == BatchMode::INFINITE && m_position == m_object_count)
    {
        m_position = 0;
    }
    return *this;
}

loader::iterator& loader::iterator::operator++(int)
{
    iterator& rc = *this;
    ++rc;
    return rc;
}

bool loader::iterator::operator==(const iterator& other) const
{
    return &m_current_loader == &other.m_current_loader && m_position == other.m_position;
}

bool loader::iterator::operator!=(const iterator& other) const
{
    return !(*this == other);
}

const buffer_out_array& loader::iterator::operator*() const
{
    return m_output_buffer;
}

loader::iterator loader::begin()
{
    iterator rc{*this, false, m_provider->get_input_count()};
    return rc;
}

loader::iterator loader::end()
{
    iterator rc{*this, true, m_provider->get_input_count()};
    return rc;
}

nervana::dataset_builder& dataset_builder::config(const std::string& config)
{
    m_config = config;
    return *this;
}

nervana::dataset_builder& dataset_builder::batch_size(size_t size)
{
    m_batch_size = size;
    return *this;
}

nervana::dataset_builder& dataset_builder::batch_count(loader::BatchMode type)
{
    m_batch_mode = type;
    return *this;
}

nervana::dataset_builder& dataset_builder::batch_count(size_t count)
{
    m_batch_mode = loader::BatchMode::COUNT;
    m_batch_count_value = count;
    return *this;
}

nervana::loader dataset_builder::create()
{
    nlohmann::json js = nlohmann::json::parse(m_config);
    js["minibatch_size"] = m_batch_size;

    loader rc(js.dump());
    if (m_batch_mode == loader::BatchMode::COUNT)
    {
        rc.set_iterator_count(m_batch_count_value);
    }
    else
    {
        rc.set_iterator_count(m_batch_mode);
    }
    return rc;
}
