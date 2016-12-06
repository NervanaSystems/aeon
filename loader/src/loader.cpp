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

loader_async::loader_async(batch_iterator_async* b_itor, const std::string& config_string)
    : async_manager<variable_buffer_array, fixed_buffer_map>(b_itor)
{
    // Note:  all we need are single_thread, batch_size, pinned + the provider template
    //        can we just use copy constructor instead?
    m_lcfg_json = nlohmann::json::parse(config_string);
    loader_config lcfg(m_lcfg_json);

    int ncores         = thread::hardware_concurrency();
    int itemsPerThread = (lcfg.batch_size - 1) / ncores + 1;
    int nthreads       = (lcfg.batch_size - 1) / itemsPerThread + 1;
    nthreads           = lcfg.single_thread ? 1 : std::min(nthreads, lcfg.batch_size);

    m_items_per_thread = (lcfg.batch_size - 1) / nthreads + 1;

    if (nthreads <= 0)
    {
        throw std::invalid_argument("Number of threads must be > 0");
    }

    for (int i = 0; i < nthreads; i++)
    {
        m_providers.push_back(nervana::provider_factory::create(m_lcfg_json));
        m_start_inds.push_back(i * m_items_per_thread);
        int item_count = i == nthreads - 1 ? (lcfg.batch_size - i * m_items_per_thread) : m_items_per_thread;
        m_end_inds.push_back(m_start_inds[i] + item_count);
    }

    auto oshapes = m_providers[0]->get_output_shapes();
    m_number_elements_in = m_providers[0]->get_input_count();

    // Allocate the space in the output buffers
    for (unsigned int k = 0; k < 2; ++k)
    {
        for (auto& sz : oshapes)
        {
            m_containers[k].add_item(sz.first, sz.second.get_byte_size(), lcfg.batch_size, lcfg.pinned);
        }
    }
}

fixed_buffer_map* loader_async::filler()
{
    fixed_buffer_map* outputs = get_pending_buffer();
    variable_buffer_array* inputs = m_source->next();

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
{
    auto tmp = nlohmann::json::parse(config_string);
    initialize(tmp);
}

loader::loader(nlohmann::json& config_json)
{
    initialize(config_json);
}

void loader::initialize(nlohmann::json& config_json)
{
    string config_string = config_json.dump();
    loader_config lcfg(config_json);
    m_batch_size                       = lcfg.batch_size;
    int block_size = lcfg.block_size == 0 ? lcfg.batch_size * 2 : lcfg.block_size;

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
        m_batch_mode = BatchMode::COUNT;
        m_batch_count_value = lcfg.iteration_mode_count;
    }

    // shared_ptr<manifest> base_manifest;
    sox_format_init();

    // the manifest defines which data should be included in the dataset
    m_manifest = make_shared<manifest_csv>(
                                    lcfg.manifest_filename,
                                    lcfg.shuffle_manifest,
                                    lcfg.manifest_root,
                                    lcfg.subset_fraction
                            );

    // TODO: make the constructor throw this error
    if (m_manifest->object_count() == 0)
    {
        throw std::runtime_error("manifest file is empty");
    }

    m_block_loader = make_shared<block_loader_file_async>(m_manifest.get(), block_size);

    // base_manifest  = manifest;

    // if (lcfg.cache_directory.length() > 0)
    // {
    //     string cache_id = base_manifest->cache_id() + to_string(m_block_loader->object_count());
    //     m_block_loader =
    //         make_shared<block_loader_cpio_cache>(lcfg.cache_directory, cache_id, base_manifest->version(), m_block_loader);
    // }

    // shared_ptr<block_iterator> block_iter;
    // if (lcfg.shuffle_every_epoch)
    // {
    //     block_iter = make_shared<block_iterator_shuffled>(m_block_loader);
    // }
    // else
    // {
    //     block_iter = make_shared<block_iterator_sequential>(m_block_loader);
    // }

    m_batch_iterator = make_shared<batch_iterator_async>(m_block_loader.get(), lcfg.batch_size);

    m_decoder = make_shared<loader_async>(m_batch_iterator.get(), config_string);

    auto                      media   = provider_factory::create(config_json);

    for (auto shape : media->get_output_shapes())
    {
        m_out_sizes.insert({shape.first, shape.second.get_byte_size()});
    }

    m_provider = provider_factory::create(config_json);
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
    // , m_num_inputs{num_inputs}
    // , m_output_buffer{ld.m_out_sizes, (size_t)ld.m_batch_size}
    , m_object_count{ld.m_manifest->object_count()}
    // , m_pending_block{(uint32_t)m_num_inputs}
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
        m_output_buffer_ptr = m_current_loader.m_decoder->next();
        // m_async_result = async(&loader::iterator::read_input, this);
        // fill_buffer();
    }
}

loader::iterator::iterator(const iterator& other)
    : m_current_loader{other.m_current_loader}
    , m_is_end{other.m_is_end}
    // , m_num_inputs{other.m_num_inputs}
    , m_object_count{other.m_object_count}
    , m_position{other.m_position}
    , m_batch_mode{other.m_batch_mode}
{
    INFO << "iterator copy ctor";
}

loader::iterator::~iterator()
{
}

// void loader::iterator::read_input()
// {
//     m_pending_block.clear();
//     m_current_loader.m_batch_iterator->read(m_pending_block);
// }

// void loader::iterator::fill_buffer()
// {
//     if (!m_is_end)
//     {
//         m_async_result.get();
//         variable_buffer_array& input_buffer = m_pending_block;

//         if (m_current_loader.m_provider)
//         {
//             for (int i = 0; i < m_current_loader.m_batch_size; i++)
//             {
//                 m_current_loader.m_provider->provide(i, input_buffer, m_output_buffer);
//             }
//         }
//         m_async_result = async(&loader::iterator::read_input, this);
//     }
// }

loader::iterator& loader::iterator::operator++()
{
    m_output_buffer_ptr = m_current_loader.m_decoder->next();
    // fill_buffer();
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

const fixed_buffer_map& loader::iterator::operator*() const
{
    return *m_output_buffer_ptr;
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

