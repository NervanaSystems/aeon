/*
 Copyright 2015 Nervana Systems Inc.
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

#include <assert.h>

#include <vector>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>

#include "pyBackendWrapper.hpp"
#include "threadpool.hpp"
#include "batch_loader.hpp"
#include "batch_iterator.hpp"
#include "manifest.hpp"
#include "provider_factory.hpp"
#include "buffer_pool_in.hpp"
#include "buffer_pool_out.hpp"

/* DecodeThreadPool
 *
 * DecodeThreadPool takes data from the BufferPool `in`, transforms it
 * using `count` threads with a Media::transform built from
 * `mediaParams`.  Each minibatch is transposed by a manager thread and
 * then copied to the `device`.
 *
 */
class pyDecodeThreadPool : public ThreadPool {
public:
    pyDecodeThreadPool(int count,
                       const std::shared_ptr<buffer_pool_in>& in,
                       const std::shared_ptr<buffer_pool_out>& out,
                       const std::shared_ptr<pyBackendWrapper>& pbe);

    virtual ~pyDecodeThreadPool();
    virtual void start();
    virtual void stop();
    void add_provider(std::shared_ptr<nervana::provider_interface> prov);

protected:
    virtual void run(int id);
    virtual void work(int id);
    void produce();
    void consume();
    void manage();

private:
    pyDecodeThreadPool();
    pyDecodeThreadPool(const pyDecodeThreadPool&);

    int                         _itemsPerThread;
    std::shared_ptr<buffer_pool_in> _in;
    std::shared_ptr<buffer_pool_out> _out;
    std::shared_ptr<pyBackendWrapper> _pbe;
    std::mutex                  _mutex;
    std::condition_variable     _started;
    std::condition_variable     _ended;
    int                         _endSignaled    = 0;
    std::thread*                _manager        = 0;
    bool                        _stopManager    = false;
    bool                        _managerStopped = false;
    buffer_in_array*            _inputBuf       = 0;
    int                         _bufferIndex    = 0;
    int                         _batchSize;

    std::vector<std::shared_ptr<nervana::provider_interface>> _providers;

    std::vector<int>            _startSignaled;
    std::vector<int>            _startInds;
    std::vector<int>            _endInds;
};

class pyLoaderConfig : public nervana::json_config_parser {
public:
    std::string manifest_filename;
    int minibatch_size;

    std::string cache_directory = "";
    int macrobatch_size      = 0;
    bool shuffle_every_epoch = false;
    bool shuffle_manifest    = false;
    bool single_thread       = false;
    int subset_percent        = 100;
    int random_seed           = 0;

    pyLoaderConfig(nlohmann::json js)
    {
        parse_value(manifest_filename, "manifest_filename", js, mode::REQUIRED);
        parse_value(minibatch_size,    "minibatch_size", js, mode::REQUIRED);

        parse_value(single_thread,       "single_thread", js);
        parse_value(cache_directory,     "cache_directory", js);
        parse_value(macrobatch_size,     "macrobatch_size", js);
        parse_value(shuffle_every_epoch, "shuffle_every_epoch", js);
        parse_value(shuffle_manifest,    "shuffle_manifest", js);
        parse_value(subset_percent,      "subset_percent", js);
        parse_value(random_seed,         "random_seed", js);

        if(macrobatch_size == 0) {
            macrobatch_size = minibatch_size;
        }

        validate();
    }

private:
    pyLoaderConfig() = delete;
    bool validate() { return true; }
};

/*
 * The ReadThread wraps BatchIterator in a thread an coordinates work
 * with other threads via locks on the output BufferPool `out`
 *
 */

class ReadThread: public ThreadPool {
public:
    ReadThread(const std::shared_ptr<buffer_pool_in>& out,
               const std::shared_ptr<BatchIterator>& batch_iterator);

protected:
    virtual void work(int id);

private:
    ReadThread();
    ReadThread(const ReadThread&);
    std::shared_ptr<buffer_pool_in> _out;
    std::shared_ptr<BatchIterator> _batch_iterator;
};


/* PyLoader
 *
 * The PyLoader instantiates and then coordinates the effort of loading ingested data, caching
 * blocks of it in contiguous disk (using cpio file format), transforming the data and finally
 * loading the data into device memory
*/

class PyLoader {
public:
    PyLoader(const char* PyloaderConfigString, PyObject *pbe);

    virtual ~PyLoader() {}
    int start();
    void stop();
    int reset();
    PyObject* shapes();
    PyObject* next(int bufIdx);

    std::shared_ptr<BatchIterator> getBatchIterator() { return _batch_iterator; }
    int itemCount() { return _manifest->getSize(); }

private:
    void drain();

private:
    PyLoader();
    PyLoader(const PyLoader&);

    bool                                _first = true;

    std::shared_ptr<buffer_pool_in>     _readBufs = nullptr;
    std::shared_ptr<buffer_pool_out>    _decodeBufs = nullptr;
    std::unique_ptr<ReadThread>         _readThread = nullptr;
    std::unique_ptr<pyDecodeThreadPool> _decodeThreads = nullptr;
    std::shared_ptr<BatchLoader>        _batchLoader = nullptr;
    std::shared_ptr<BatchIterator>      _batch_iterator = nullptr;
    std::shared_ptr<Manifest>           _manifest = nullptr;
    std::shared_ptr<pyLoaderConfig>     _lcfg = nullptr;

    std::vector<std::shared_ptr<nervana::interface::config>> _provider_configs;

    int                                 _batchSize;
    nlohmann::json                      _lcfg_json;
    PyObject*                           _pbe;
    std::shared_ptr<pyBackendWrapper>   _pyBackend;
};
