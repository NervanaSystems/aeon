/*
 Copyright 2016 Intel(R) Nervana(TM)
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
#include "log.hpp"
#include "web_app.hpp"
#include "manifest_nds.hpp"

#if defined(ENABLE_AEON_SERVICE)
#include "client/loader_remote.hpp"
#include "client/curl_connector.hpp"
#include "client/remote_config.hpp"
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
#include "client/ofi_connector.hpp"
#endif
#endif

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

loader_local::~loader_local()
{
    if (m_debug_web_app)
    {
        m_debug_web_app->deregister_loader(this);
    }
}

void loader_local::initialize(const json& config_json)
{
    string config_string = config_json.dump();
    m_current_config     = config_json;
    loader_config lcfg(config_json);
    m_batch_size = lcfg.batch_size;

    // shared_ptr<manifest> base_manifest;
    sox_format_init();

    if (nervana::manifest_nds::is_likely_json(lcfg.manifest_filename))
    {
        m_manifest_nds = nervana::manifest_nds_builder()
                             .filename(lcfg.manifest_filename)
                             .block_size(lcfg.block_size)
                             .elements_per_record(2)
                             .shuffle(lcfg.shuffle_manifest)
                             .seed(lcfg.random_seed)
                             .make_shared();

        m_block_loader = std::make_shared<block_loader_nds>(m_manifest_nds, lcfg.block_size);
    }
    else
    {
        // the manifest defines which data should be included in the dataset
        m_manifest_file = make_shared<manifest_file>(lcfg.manifest_filename,
                                                     lcfg.shuffle_manifest,
                                                     lcfg.manifest_root,
                                                     lcfg.subset_fraction,
                                                     lcfg.block_size,
                                                     lcfg.random_seed);

        // TODO: make the constructor throw this error
        if (record_count() == 0)
        {
            throw std::runtime_error("manifest file is empty");
        }
        m_block_loader = make_shared<block_loader_file>(m_manifest_file, lcfg.block_size);
    }

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

    unsigned int threads_num = lcfg.decode_thread_count != 0 ? lcfg.decode_thread_count
                                                             : std::thread::hardware_concurrency();

    const int decode_size =
        lcfg.batch_size * ((threads_num * m_input_multiplier - 1) / lcfg.batch_size + 1);
    m_batch_iterator = make_shared<batch_iterator>(m_block_manager, decode_size);

    m_decoder = make_shared<batch_decoder>(m_batch_iterator,
                                           decode_size,
                                           lcfg.decode_thread_count,
                                           lcfg.pinned,
                                           m_provider,
                                           lcfg.random_seed);

    m_final_stage =
        make_shared<batch_iterator_fbm>(m_decoder, lcfg.batch_size, m_provider, !lcfg.batch_major);

    m_output_buffer_ptr = m_final_stage->next();

    if (lcfg.web_server_port != 0)
    {
        m_debug_web_app = make_shared<web_app>(lcfg.web_server_port);
        m_debug_web_app->register_loader(this);
    }
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
#if defined(ENABLE_AEON_SERVICE)
    if (remote_version(config))
    {
        return create_loader_remote(config);
    }
#endif
    return unique_ptr<loader_local>(new loader_local(config));
}

#if defined(ENABLE_AEON_SERVICE)

bool loader_factory::remote_version(const json& config)
{
    try
    {
        config.at("remote");
    }
    catch (json::out_of_range)
    {
        return false;
    }

    return true;
}

unique_ptr<loader> loader_factory::create_loader_remote(const json& js)
{
    remote_config config(js.at("remote"));

    shared_ptr<http_connector> http_connector_obj =
        make_shared<curl_connector>(config.address, config.port);
#if defined(ENABLE_OPENFABRICS_CONNECTOR)
    if (!config.rdma_address.empty() && config.rdma_port != 0)
    {
        INFO << "Using OFI library to fetch batches via RDMA.";
        auto ofi_ptr = new ofi_connector(config.rdma_address, config.rdma_port, http_connector_obj);
        http_connector_obj = shared_ptr<ofi_connector>(ofi_ptr);
    }
#endif
    shared_ptr<service> service_obj;
    if (config.async)
    {
        auto service_connector_obj    = make_shared<service_connector>(http_connector_obj);
        auto service_async_source_obj = make_shared<service_async_source>(service_connector_obj);
        service_obj                   = make_shared<service_async>(service_async_source_obj);
        INFO << "Using asynchronous batch fetching.";
    }
    else
    {
        service_obj = make_shared<service_connector>(http_connector_obj);
    }

    loader_remote* new_loader = new loader_remote(service_obj, js);
    return unique_ptr<loader_remote>(new_loader);
}
#endif
