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
#include "log.hpp"

using namespace std;
using namespace nervana;

loader_async::loader_async(batch_iterator_async* b_itor, size_t batch_size, bool single_thread, bool pinned,
                           const std::shared_ptr<provider_interface>& prov)
    : async_manager<variable_buffer_array, fixed_buffer_map>(b_itor)
    , m_batch_size(batch_size)
{
    // Note:  all we need are single_thread, batch_size, pinned + the provider template
    //        can we just use copy constructor instead?
    int nthreads = 1;

    if (!single_thread)
    {
        int itemsPerThread = (batch_size - 1) / thread::hardware_concurrency() + 1;
        nthreads           = std::min((batch_size - 1) / itemsPerThread + 1, batch_size);
    }

    m_items_per_thread = (batch_size - 1) / nthreads + 1;

    if (nthreads <= 0)
    {
        throw std::invalid_argument("Number of threads must be > 0");
    }

    for (int i = 0; i < nthreads; i++)
    {
        m_providers.push_back(nervana::provider_factory::clone(prov));
        m_start_inds.push_back(i * m_items_per_thread);
        int record_count = i == nthreads - 1 ? (batch_size - i * m_items_per_thread) : m_items_per_thread;
        m_end_inds.push_back(m_start_inds[i] + record_count);
    }

    auto oshapes         = m_providers[0]->get_output_shapes();
    m_number_elements_in = m_providers[0]->get_input_count();

    // Allocate the space in the output buffers
    for (unsigned int k = 0; k < 2; ++k)
    {
        for (auto& sz : oshapes)
        {
            m_containers[k].add_item(sz.first, sz.second, batch_size, pinned);
        }
    }
}

loader_async::~loader_async()
{
    finalize();
}

fixed_buffer_map* loader_async::filler()
{
    fixed_buffer_map*      outputs = get_pending_buffer();
    variable_buffer_array* inputs  = m_source->next();

    if (inputs == nullptr)
    {
        outputs = nullptr;
    }
    else
    {
        std::vector<std::thread> provider_threads;
        try
        {
            for (int id = 0; id < m_providers.size(); ++id)
            {
                provider_threads.emplace_back(&loader_async::work, this, id, inputs, outputs);
            }

            for (auto& t : provider_threads)
            {
                t.join();
            }
            // Now perform any potentially necessary whole-batch operation
            m_providers[0]->post_process(*outputs);
        }
        catch (std::exception& e)
        {
            outputs = nullptr;
        }
    }

    return outputs;
}

void loader_async::work(int id, variable_buffer_array* in_buf, fixed_buffer_map* out_buf)
{
    // Thread function.
    // No locking required because threads write into non-overlapping regions.
    try
    {
        affirm(in_buf->at(0).get_item_count() != 0, "input buffer pool is empty.");

        for (int item_idx = m_start_inds[id]; item_idx < m_end_inds[id]; item_idx++)
        {
            m_providers[id]->provide(item_idx, *in_buf, *out_buf);
        }
    }
    catch (std::exception& e)
    {
        cout << "decode_thread_pool exception: " << e.what() << endl;
        // m_buffer_pool_decoded->write_exception(std::current_exception());
    }
}

loader_config::loader_config(nlohmann::json js)
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

    set_global_random_seed(random_seed);
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

loader::loader(const std::string& config_string)
    : m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    auto tmp = nlohmann::json::parse(config_string);
    initialize(tmp);
}

loader::loader(nlohmann::json& config_json)
    : m_current_iter(*this, false)
    , m_end_iter(*this, true)
{
    initialize(config_json);
}

void loader::initialize(nlohmann::json& config_json)
{
    string        config_string = config_json.dump();
    loader_config lcfg(config_json);
    m_batch_size = lcfg.batch_size;

    // shared_ptr<manifest> base_manifest;
    sox_format_init();

    // the manifest defines which data should be included in the dataset
    m_manifest = make_shared<manifest_csv>(lcfg.manifest_filename, lcfg.shuffle_manifest, lcfg.manifest_root, lcfg.subset_fraction);

    // TODO: make the constructor throw this error
    if (m_manifest->record_count() == 0)
    {
        throw std::runtime_error("manifest file is empty");
    }

    if (lcfg.iteration_mode == "ONCE")
    {
        m_batch_mode = BatchMode::ONCE;
        m_batch_count_value = m_manifest->record_count();
    }
    else if (lcfg.iteration_mode == "INFINITE")
    {
        m_batch_mode = BatchMode::INFINITE;
        m_batch_count_value = m_manifest->record_count();
    }
    else if (lcfg.iteration_mode == "COUNT")
    {
        m_batch_mode        = BatchMode::COUNT;
        m_batch_count_value = lcfg.iteration_mode_count;
    }

    m_block_loader = make_shared<block_loader_file_async>(m_manifest.get(), lcfg.block_size);

    m_block_manager = make_shared<block_manager_async>(m_block_loader.get(), lcfg.block_size, lcfg.cache_directory, lcfg.shuffle_every_epoch);

    m_batch_iterator = make_shared<batch_iterator_async>(m_block_manager.get(), lcfg.batch_size);

    m_provider = provider_factory::create(config_json);

    m_decoder = make_shared<loader_async>(m_batch_iterator.get(), static_cast<size_t>(lcfg.batch_size), lcfg.single_thread,
                                          lcfg.pinned, m_provider);

    m_output_buffer_ptr = m_decoder->next();

}

const vector<string>& loader::get_buffer_names() const
{
    return m_provider->get_buffer_names();
}

const map<string, shape_type>& loader::get_names_and_shapes() const
{
    return m_provider->get_output_shapes();
}

const shape_t& loader::get_shape(const string& name) const
{
    return m_provider->get_output_shape(name).get_shape();
}

loader::iterator::iterator(loader& ld, bool is_end)
    : m_current_loader(ld)
    , m_is_end{is_end}
{}

loader::iterator::iterator(const iterator& other)
    : m_current_loader{other.m_current_loader}
    , m_is_end{other.m_is_end}
{}

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
    return !m_is_end && (position() >= m_current_loader.m_batch_count_value);
}

const fixed_buffer_map& loader::iterator::operator*() const
{
    return *(m_current_loader.m_output_buffer_ptr);
}

void loader::increment_position()
{
    m_output_buffer_ptr = m_decoder->next();
    m_position++;

    // Wrap around if this is an infinite iterator
    if (m_batch_mode == BatchMode::INFINITE && m_position == m_batch_count_value)
    {
        m_position = 0;
    }
}
