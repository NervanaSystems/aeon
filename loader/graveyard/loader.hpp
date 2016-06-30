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

#include "threadpool.hpp"
#include "matrix.hpp"
#include "device.hpp"
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
class DecodeThreadPool : public ThreadPool {
public:
    DecodeThreadPool(int count, int batchSize, nlohmann::json config, DeviceParams *dp);
    virtual ~DecodeThreadPool();
    virtual void start();
    virtual void stop();
    int get_dtm_len() { return _datumLen; }
    int get_tgt_len() { return _targetLen; }
    void set_io_buffers(const std::shared_ptr<BufferPool>& in,
                        const std::shared_ptr<Device>& device,
                        const std::shared_ptr<BufferPool>& out);

protected:
    virtual void run(int id);
    virtual void work(int id);
    void produce();
    void consume();
    void manage();

private:
    DecodeThreadPool();
    DecodeThreadPool(const DecodeThreadPool&);

    int                         _itemsPerThread;
    std::shared_ptr<BufferPool> _in;
    std::shared_ptr<BufferPool> _out;
    std::mutex                  _mutex;
    std::condition_variable     _started;
    std::condition_variable     _ended;
    std::vector<int>            _startSignaled;
    int                         _endSignaled;
    std::thread*                _manager;
    bool                        _stopManager;
    bool                        _managerStopped;
    BufferPair*                 _inputBuf;
    int                         _bufferIndex;
    int                         _batchSize;
    std::vector<int>            _startInds;
    std::vector<int>            _endInds;
    std::vector<int>            _dataOffsets;
    std::vector<int>            _targetOffsets;

    int                         _datumLen;
    int                         _targetLen;

    std::shared_ptr<Device>     _device;
    DeviceParams*               _deviceParams;
    std::vector<std::shared_ptr<nervana::train_base>> _providers;
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


class LoaderConfig : public nervana::json_config_parser {
public:
    std::string manifest_filename;
    std::string cache_directory;
    int macrobatch_size;

    bool shuffle_every_epoch = false;
    bool shuffle_manifest    = false;
    int subset_percent        = 100;
    int random_seed           = 0;

    bool set_config(nlohmann::json js) override
    {
        parse_req(manifest_filename, "manifest_filename", js);
        parse_req(cache_directory,   "cache_directory", js);
        parse_req(macrobatch_size,   "macrobatch_size", js);

        parse_opt(shuffle_every_epoch, "shuffle_every_epoch", js);
        parse_opt(shuffle_manifest,    "shuffle_manifest", js);
        parse_opt(subset_percent,      "subset_percent", js);
        parse_opt(random_seed,         "random_seed", js);

        return validate();
    }

private:
    bool validate() { return true; }
};

/* Loader
 *
 * The Loader instantiates and then coordinates the effort of
 * loading ingested data, caching blocks of it in contiguous
 * disk (using cpio file format), transforming the data and
 * finally loading the data into device memory
 *
 */
class Loader {
public:
    Loader(int miniBatchSize, const char* loaderConfigString, DeviceParams *deviceParams);

    virtual ~Loader() {}
    int start();
    void stop();
    int reset();
    void next();

    std::shared_ptr<BatchIterator> getBatchIterator() { return _batch_iterator; }
    std::shared_ptr<Device> getDevice() { return _device; }
    int itemCount() { return _manifest->getSize(); }

private:
    void drain();

private:
    Loader();
    Loader(const Loader&);
    bool                                _first;
    int                                 _miniBatchSize;
    int                                 _datumSize;
    int                                 _datumTypeSize;
    int                                 _targetSize;
    int                                 _targetTypeSize;
    DeviceParams*                       _deviceParams;
    std::shared_ptr<BufferPool>         _readBufs;
    std::shared_ptr<BufferPool>         _decodeBufs;
    std::unique_ptr<ReadThread>         _readThread;
    std::unique_ptr<DecodeThreadPool>   _decodeThreads;
    std::shared_ptr<Device>             _device;
    std::shared_ptr<BatchIterator>      _batch_iterator;
    std::shared_ptr<Manifest>           _manifest;
    nlohmann::json                      _loaderConfigJson;
};
