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

#include "loader.hpp"
#include "block_loader_cpio_cache.hpp"
#include "block_iterator_sequential.hpp"
#include "block_iterator_shuffled.hpp"
#include "batch_iterator.hpp"
#include "manifest_nds.hpp"
#include "block_loader_nds.hpp"

using namespace std;
using namespace nervana;


loader::loader(const char* cfg_string, PyObject *py_obj_backend)
{
    m_python_backend = make_shared<python_backend>(py_obj_backend);
    m_lcfg_json = nlohmann::json::parse(cfg_string);
    loader_config lcfg(m_lcfg_json);
    m_batch_size = lcfg.minibatch_size;
    m_single_thread_mode = lcfg.single_thread;
    shared_ptr<nervana::manifest> base_manifest = nullptr;
    sox_format_init();

    if(nervana::manifest_nds::is_likely_json(lcfg.manifest_filename)) {
        affirm(lcfg.subset_fraction == 1, "subset_fraction must be 1.0 for nds");

        auto manifest = make_shared<nervana::manifest_nds>(lcfg.manifest_filename);

        // TODO: add shard_count/shard_index to cfg
        m_block_loader = make_shared<block_loader_nds>(manifest->baseurl,
                                                      manifest->token,
                                                      manifest->collection_id,
                                                      lcfg.macrobatch_size);

        base_manifest = manifest;
    } else {
        // the manifest defines which data should be included in the dataset
        auto manifest = make_shared<nervana::manifest_csv>(lcfg.manifest_filename,
                                                           lcfg.shuffle_manifest, lcfg.manifest_root);

        // TODO: make the constructor throw this error
        if(manifest->object_count() == 0) {
            throw std::runtime_error("manifest file is empty");
        }

        m_block_loader = make_shared<block_loader_file>(manifest,
                                                       lcfg.subset_fraction,
                                                       lcfg.macrobatch_size);
        base_manifest = manifest;
    }

    if(lcfg.cache_directory.length() > 0) {
        string cache_id = base_manifest->cache_id() + to_string(m_block_loader->object_count());
        m_block_loader = make_shared<block_loader_cpio_cache>(lcfg.cache_directory,
                                                             cache_id,
                                                             base_manifest->version(),
                                                             m_block_loader);
    }

    shared_ptr<block_iterator> block_iter;
    if (lcfg.shuffle_every_epoch) {
        block_iter = make_shared<block_iterator_shuffled>(m_block_loader);
    } else {
        block_iter = make_shared<block_iterator_sequential>(m_block_loader);
    }

    m_batch_iterator = make_shared<batch_iterator>(block_iter, lcfg.minibatch_size);
}


int loader::start()
{
    m_first = true;
    try {
        int ncores         = thread::hardware_concurrency();
        int itemsPerThread = (m_batch_size - 1) /  ncores + 1;
        int nthreads       = (m_batch_size - 1) / itemsPerThread + 1;
        nthreads           = m_single_thread_mode ? 1 : std::min(nthreads, m_batch_size);

        if (nthreads <= 0)
        {
            throw std::invalid_argument("Number of threads must be > 0");
        }

        vector<shared_ptr<nervana::provider_interface>> providers;
        for (int i=0; i<nthreads; i++) {
            providers.push_back(nervana::provider_factory::create(m_lcfg_json));
        }

        // variable size buffers for reading encoded data (start off zero and grow as needed)
        m_read_buffers = make_shared<buffer_pool_in>(providers[0]->num_inputs);
        m_read_thread_pool = unique_ptr<read_thread_pool>(
                        new read_thread_pool(m_read_buffers, m_batch_iterator));

        // fixed size buffers for writing out decoded data
        const vector<nervana::shape_type>& oshapes = providers[0]->get_oshapes();
        vector<size_t> write_sizes;
        for (auto& o: oshapes)
        {
            write_sizes.push_back(o.get_byte_size());
        }

        // Bind the python backend here
        m_python_backend->setup_buffers(oshapes, m_batch_size);
        // These are fixed size output buffers (need batchSize for stride)
        m_decode_buffers = make_shared<buffer_pool_out>(write_sizes,
                                                       (size_t)m_batch_size,
                                                       m_python_backend->use_pinned_memory());

        m_decode_thread_pool = unique_ptr<decode_thread_pool>(
                new decode_thread_pool(nthreads, m_read_buffers, m_decode_buffers, m_python_backend));

        for (auto& p: providers)
        {
            m_decode_thread_pool->add_provider(p);
        }

    } catch(std::bad_alloc&) {
        return -1;
    }
    m_decode_thread_pool->start();
    m_read_thread_pool->start();

    return 0;
}

void loader::stop()
{
    m_read_thread_pool->stop();
    while (m_read_thread_pool->stopped() == false)
    {
        std::this_thread::yield();
        drain();
    }
    while ((m_decode_buffers->empty() == false) ||
           (m_read_buffers->empty() == false))
    {
        drain();
    }
    m_decode_thread_pool->stop();

    m_read_thread_pool   = nullptr;
    m_decode_buffers     = nullptr;
    m_decode_thread_pool = nullptr;
    m_python_backend->clear_buffers();
}

int loader::reset()
{
    stop();
    m_batch_iterator->reset();
    return start();
}

PyObject* loader::next(int bufIdx)
{
    unique_lock<mutex> lock(m_decode_buffers->get_mutex());
    if (m_first == true) {
        m_first = false;
    } else {
        // Unlock the buffer used for the previous minibatch.
        m_decode_buffers->advance_read_pos();
        m_decode_buffers->signal_not_full();
    }

    while (m_decode_buffers->empty()) {
        m_decode_buffers->wait_for_not_empty(lock);
    }

    m_decode_buffers->reraise_exception();
    return m_python_backend->get_host_tuple(bufIdx);
}

PyObject* loader::shapes()
{
    return m_python_backend->get_shapes();
}

void loader::drain()
{
    {
        unique_lock<mutex> lock(m_decode_buffers->get_mutex());
        if (m_decode_buffers->empty() == true) {
            return;
        }
        m_decode_buffers->advance_read_pos();
    }
    m_decode_buffers->signal_not_full();
}
