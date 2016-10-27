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

#include "python_backend.hpp"
#include "thread_pool.hpp"
#include "block_loader.hpp"
#include "block_iterator.hpp"
#include "batch_iterator.hpp"
#include "manifest.hpp"
#include "provider_factory.hpp"
#include "buffer_pool_in.hpp"
#include "buffer_pool_out.hpp"
#include "util.hpp"

namespace nervana {
    class decode_thread_pool;
    class loader_config;
    class read_thread_pool;
    class loader;
}

/* decode_thread_pool
 *
 * decode_thread_pool takes data from the BufferPool `in`, transforms it
 * using `count` threads with a Media::transform built from
 * `mediaParams`.  Each minibatch is transposed by a manager thread and
 * then copied to the `device`.
 *
 */
class nervana::decode_thread_pool : public nervana::thread_pool {
public:
    decode_thread_pool(int count,
                       const std::shared_ptr<nervana::buffer_pool_in>& in,
                       const std::shared_ptr<nervana::buffer_pool_out>& out,
                       const std::shared_ptr<python_backend>& pbe);

    virtual ~decode_thread_pool();
    virtual void start() override;
    virtual void stop() override;
    void add_provider(std::shared_ptr<nervana::provider_interface> prov);

protected:
    virtual void run(int id) override;
    virtual void work(int id) override;
    void produce();
    void consume();
    void manage();

private:
    decode_thread_pool();
    decode_thread_pool(const decode_thread_pool&);

    int                         _itemsPerThread;
    std::shared_ptr<nervana::buffer_pool_in> _in;
    std::shared_ptr<nervana::buffer_pool_out> _out;
    std::shared_ptr<python_backend> _python_backend;
    std::mutex                  _mutex;
    std::condition_variable     _started;
    std::condition_variable     _ended;
    int                         _batchSize;
    int                         _endSignaled    = 0;
    std::thread*                _manager        = 0;
    bool                        _stopManager    = false;
    bool                        _managerStopped = false;
    nervana::buffer_in_array*   _inputBuf       = 0;
    int                         _bufferIndex    = 0;

    std::vector<std::shared_ptr<nervana::provider_interface>> _providers;

    std::vector<int>            _startSignaled;
    std::vector<int>            _startInds;
    std::vector<int>            _endInds;
};

class nervana::loader_config : public nervana::interface::config {
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
        if(js.is_null()) {
            throw std::runtime_error("missing loader config in json config");
        }

        for(auto& info : config_list) {
            info->parse(js);
        }
        verify_config("loader", config_list, js);

        if(macrobatch_size == 0) {
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
        ADD_SCALAR(subset_fraction, mode::OPTIONAL, [](decltype(subset_fraction) v){ return v <= 1.0 && v >= 0.0; }),
        ADD_SCALAR(shuffle_every_epoch, mode::OPTIONAL),
        ADD_SCALAR(shuffle_manifest, mode::OPTIONAL),
        ADD_SCALAR(single_thread, mode::OPTIONAL),
        ADD_SCALAR(random_seed, mode::OPTIONAL),
    };

    loader_config() {}
    bool validate() { return true; }
};

/*
 * The read_thread_pool wraps BatchIterator in a thread an coordinates work
 * with other threads via locks on the output BufferPool `out`
 *
 */

class nervana::read_thread_pool: public thread_pool {
public:
    read_thread_pool(const std::shared_ptr<nervana::buffer_pool_in>& out,
                     const std::shared_ptr<nervana::batch_iterator>& batch_iterator);

protected:
    virtual void work(int id) override;

private:
    read_thread_pool();
    read_thread_pool(const read_thread_pool&);
    std::shared_ptr<nervana::buffer_pool_in> _out;
    std::shared_ptr<nervana::batch_iterator> _batch_iterator;
};


/* loader
 *
 * The loader instantiates and then coordinates the effort of loading ingested data, caching
 * blocks of it in contiguous disk (using cpio file format), transforming the data and finally
 * loading the data into device memory
*/

class nervana::loader {
public:
    loader(const char*, PyObject *);

    virtual ~loader() {}
    int start();
    void stop();
    int reset();
    PyObject* shapes();
    PyObject* next(int bufIdx);

    int itemCount() { return _block_loader->objectCount(); }

private:
    void drain();

private:
    loader();
    loader(const loader&);

    bool                                        _first = true;
    bool                                        _single_thread_mode = false;

    std::shared_ptr<nervana::buffer_pool_in>    _read_buffers = nullptr;
    std::shared_ptr<nervana::buffer_pool_out>   _decode_buffers = nullptr;
    std::unique_ptr<nervana::read_thread_pool>  _read_thread_pool = nullptr;
    std::unique_ptr<decode_thread_pool>         _decode_thread_pool = nullptr;
    std::shared_ptr<nervana::block_loader>      _block_loader = nullptr;
    std::shared_ptr<nervana::batch_iterator>    _batch_iterator = nullptr;

    int                                         _batchSize;
    nlohmann::json                              _lcfg_json;
    std::shared_ptr<python_backend>             _python_backend;
};
