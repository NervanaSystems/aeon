/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <vector>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>
#include <memory>

#include "loader.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

using nlohmann::json;

loader_config::loader_config(json js)
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
        block_size = 2 * batch_size;
    }

    validate();
}

void loader_config::validate()
{
    if (iteration_mode == "ONCE")
    {
    }
    else if (iteration_mode == "INFINITE")
    {
    }
    else if (iteration_mode == "COUNT")
    {
        if (iteration_mode_count <= 0)
        {
            throw invalid_argument("iteration_mode_count must be a positive integer");
        }
    }
    else
    {
        throw invalid_argument("iteration_mode must be one of ONCE, COUNT, or INFINITE");
    }
}

loader_local::loader_local(const std::string& config_string)
    : m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    auto tmp = json::parse(config_string);
    initialize(tmp);
}

loader_local::loader_local(const json& config_json)
    : m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    initialize(config_json);
}

void loader_local::initialize(const json& config_json)
{
    string config_string = config_json.dump();
    m_current_config     = config_json;
    loader_config lcfg(config_json);
    m_batch_size = lcfg.batch_size;

    // shared_ptr<manifest> base_manifest;

    if (lcfg.node_count > 1)
    {
        if (lcfg.node_id >= lcfg.node_count)
            throw std::runtime_error("node_id can't be greater than node_count");

        if (!lcfg.cache_directory.empty())
        {
            WARN<<"File caching for multinode is not implemented yet";
            lcfg.cache_directory.clear();
        }

        if (lcfg.random_seed == 0)
        {
            WARN<<"You have to set non zero random_seed for multi node training. random_seed = 1 is used";
            lcfg.random_seed = 1;
        }
    }

    // the manifest defines which data should be included in the dataset
    m_manifest_file = make_shared<manifest_file>(lcfg.manifest_filename,
                                                    lcfg.shuffle_manifest,
                                                    lcfg.manifest_root,
                                                    lcfg.subset_fraction,
                                                    lcfg.block_size,
                                                    lcfg.random_seed,
                                                    lcfg.node_id,
                                                    lcfg.node_count,
                                                    lcfg.batch_size);

    // TODO: make the constructor throw this error
    if (record_count() == 0)
    {
        throw std::runtime_error("manifest file is empty");
    }
    m_block_loader = make_shared<block_loader_file>(m_manifest_file, lcfg.block_size);

    m_block_manager = make_shared<block_manager>(m_block_loader,
                                                 lcfg.block_size,
                                                 lcfg.cache_directory,
                                                 lcfg.shuffle_enable,
                                                 lcfg.random_seed);

    // Default ceil div to get number of batches
    m_batch_count_value = (record_count() + m_batch_size - 1) / m_batch_size;
    if (lcfg.iteration_mode == "ONCE")
    {
        m_batch_mode = BatchMode::ONCE;
    }
    else if (lcfg.iteration_mode == "INFINITE")
    {
        m_batch_mode = BatchMode::INFINITE;
    }
    else if (lcfg.iteration_mode == "COUNT")
    {
        m_batch_mode        = BatchMode::COUNT;
        m_batch_count_value = lcfg.iteration_mode_count;
    }

    m_provider = provider_factory::create(config_json);

    std::vector<int> thread_affinity_map = nervana::get_thread_affinity_map(
        lcfg.cpu_list, m_max_count_of_free_threads, m_free_threads_ratio);

    // Smallest multiple of batch_size ensuring at least m_input_multiplier images per thread
    int decode_size = lcfg.batch_size *
                      ((thread_affinity_map.size() * m_input_multiplier - 1) / lcfg.batch_size + 1);

    m_batch_iterator = make_shared<batch_iterator>(m_block_manager, decode_size);

    m_decoder = make_shared<batch_decoder>(m_batch_iterator,
                                           lcfg.batch_size,
                                           decode_size,
                                           std::move(thread_affinity_map),
                                           lcfg.pinned,
                                           m_provider,
                                           lcfg.random_seed + lcfg.node_id);

    m_final_stage =
        make_shared<batch_iterator_fbm>(m_decoder, lcfg.batch_size, m_provider, !lcfg.batch_major);

    m_output_buffer_ptr = m_final_stage->next();
}

const vector<string>& loader_local::get_buffer_names() const
{
    return m_provider->get_buffer_names();
}

const vector<pair<string, shape_type>>& loader_local::get_names_and_shapes() const
{
    return m_provider->get_output_shapes();
}

const shape_t& loader_local::get_shape(const string& name) const
{
    return m_provider->get_output_shape(name).get_shape();
}

loader::iterator::iterator(loader& ld, bool is_end)
    : m_current_loader(ld)
    , m_is_end{is_end}
{
}

loader::iterator::iterator(const iterator& other)
    : m_current_loader{other.m_current_loader}
    , m_is_end{other.m_is_end}
{
}

loader::iterator& loader::iterator::operator++()
{
    m_current_loader.increment_position();
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
    bool res = &m_current_loader == &other.m_current_loader;
    res &= (other.m_is_end && positional_end()) || (m_is_end && other.positional_end());
    return res;
}

bool loader::iterator::operator!=(const iterator& other) const
{
    return !(*this == other);
}
// Whether or not this strictly positional iterator has reached the end
bool loader::iterator::positional_end() const
{
    return !m_is_end && (position() >= m_current_loader.batch_count());
}

const fixed_buffer_map& loader::iterator::operator*() const
{
    if (m_current_loader.get_output_buffer())
        return *m_current_loader.get_output_buffer();
    else
        throw std::runtime_error("empty buffer");
}

void loader_local::increment_position()
{
    m_output_buffer_ptr = m_final_stage->next();
    m_position++;

    // Wrap around if this is an infinite iterator
    if (m_batch_mode == BatchMode::INFINITE && m_position == m_batch_count_value)
    {
        m_position = 0;
    }
}

std::unique_ptr<loader> loader_factory::get_loader(const std::string& config)
{
    json parsed_config = json::parse(config);
    return get_loader(parsed_config);
}

std::unique_ptr<loader> loader_factory::get_loader(const json& config)
{
    return unique_ptr<loader_local>(new loader_local(config));
}
