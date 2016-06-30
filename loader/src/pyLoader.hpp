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
#include "batch_iterator.hpp"
#include "manifest.hpp"
#include "provider_factory.hpp"


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
                       const std::shared_ptr<BufferPool>& in,
                       const std::shared_ptr<BufferPool>& out,
                       const std::shared_ptr<pyBackendWrapper>& pbe);

    virtual ~pyDecodeThreadPool();
    virtual void start();
    virtual void stop();
    void add_provider(std::shared_ptr<nervana::train_base> prov);

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
    std::shared_ptr<BufferPool> _in;
    std::shared_ptr<BufferPool> _out;
    std::shared_ptr<pyBackendWrapper> _pbe;
    std::mutex                  _mutex;
    std::condition_variable     _started;
    std::condition_variable     _ended;
    int                         _endSignaled    = 0;
    std::thread*                _manager        = 0;
    bool                        _stopManager    = false;
    bool                        _managerStopped = false;
    BufferPair*                 _inputBuf       = 0;
    int                         _bufferIndex    = 0;
    int                         _batchSize;

    std::vector<std::shared_ptr<nervana::train_base>> _providers;

    std::vector<int>            _startSignaled;
    std::vector<int>            _startInds;
    std::vector<int>            _endInds;
    std::vector<int>            _dataOffsets;
    std::vector<int>            _targetOffsets;

    int                         _datumLen;
    int                         _targetLen;
};

class pyLoaderConfig : public nervana::json_config_parser {
public:
    std::string manifest_filename;
    std::string cache_directory;
    int macrobatch_size;
    int minibatch_size;

    bool shuffle_every_epoch = false;
    bool shuffle_manifest    = false;
    int subset_percent        = 100;
    int random_seed           = 0;

    bool set_config(nlohmann::json js) override
    {
        parse_req(manifest_filename, "manifest_filename", js);
        parse_req(cache_directory,   "cache_directory", js);
        parse_req(macrobatch_size,   "macrobatch_size", js);
        parse_req(minibatch_size,    "minibatch_size", js);

        parse_opt(shuffle_every_epoch, "shuffle_every_epoch", js);
        parse_opt(shuffle_manifest,    "shuffle_manifest", js);
        parse_opt(subset_percent,      "subset_percent", js);
        parse_opt(random_seed,         "random_seed", js);

        return validate();
    }

private:
    bool validate() { return true; }
};

/*
 * The ReadThread wraps BatchIterator in a thread an coordinates work
 * with other threads via locks on the output BufferPool `out`
 *
 */

class ReadThread: public ThreadPool {
public:
    ReadThread(const std::shared_ptr<BufferPool>& out,
               const std::shared_ptr<BatchIterator>& batch_iterator);

protected:
    virtual void work(int id);

private:
    ReadThread();
    ReadThread(const ReadThread&);
    std::shared_ptr<BufferPool> _out;
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
    // void next();
    PyObject* next(int bufIdx);

    std::shared_ptr<BatchIterator> getBatchIterator() { return _batch_iterator; }
    int itemCount() { return _manifest->getSize(); }

private:
    void drain();

private:
    PyLoader();
    PyLoader(const PyLoader&);

    bool                                _first = true;

    std::shared_ptr<BufferPool>         _readBufs = nullptr;
    std::shared_ptr<BufferPool>         _decodeBufs = nullptr;
    std::unique_ptr<ReadThread>         _readThread = nullptr;
    std::unique_ptr<pyDecodeThreadPool> _decodeThreads = nullptr;
    std::shared_ptr<BatchIterator>      _batch_iterator = nullptr;
    std::shared_ptr<Manifest>           _manifest = nullptr;

    nervana::count_size_type            _dtmInfo;
    nervana::count_size_type            _tgtInfo;
    int                                 _batchSize;
    std::shared_ptr<pyLoaderConfig>     _lcfg;
    nlohmann::json                      _lcfg_json;
    PyObject*                           _pbe;
    std::shared_ptr<pyBackendWrapper>   _pyBackend;
};
