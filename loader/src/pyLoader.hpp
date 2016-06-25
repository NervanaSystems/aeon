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
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "threadpool.hpp"
#include "loader.hpp"
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
                       int batchSize, int datumLen, int targetLen);
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

class pyLoaderConfig : public LoaderConfig {
public:
    int minibatch_size;
    bool set_config(nlohmann::json js) override
    {
        parse_req(minibatch_size,   "minibatch_size", js);
        return validate();
    }

private:
    bool validate() { return true; }
};

/* PyLoader
 *
 * The PyLoader instantiates and then coordinates the effort of loading ingested data, caching
 * blocks of it in contiguous disk (using cpio file format), transforming the data and finally
 * loading the data into device memory
*/

class PyLoader {
public:
    PyLoader(PyObject *pBackend, const char* PyloaderConfigString);

    virtual ~PyLoader() {}
    int start();
    void stop();
    int reset();
    void next();

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
    PyObject*                           _pBackend;
};
