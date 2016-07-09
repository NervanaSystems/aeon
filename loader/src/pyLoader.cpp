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

#include <assert.h>

#include <vector>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <utility>
#include <algorithm>

#include "matrix.hpp"
#include "pyLoader.hpp"
#include "batch_loader_cpio_cache.hpp"
#include "sequential_batch_iterator.hpp"
#include "shuffled_batch_iterator.hpp"
#include "minibatch_iterator.hpp"

using namespace std;

pyDecodeThreadPool::pyDecodeThreadPool(int count,
                                       const shared_ptr<host_buffer_pool>& in,
                                       const shared_ptr<host_buffer_pool>& out,
                                       const shared_ptr<pyBackendWrapper>& pbe)
: ThreadPool(count), _in(in), _out(out), _pbe(pbe)
{
    _batchSize = _pbe->_batchSize;
    _itemsPerThread = (_batchSize - 1) / _count + 1;
    _datumLen = _pbe->_dtm_config->get_size_bytes();
    _targetLen = _pbe->_tgt_config->get_size_bytes();

    assert(_itemsPerThread * count >= _batchSize);
    assert(_itemsPerThread * (count - 1) < _batchSize);
}


void pyDecodeThreadPool::add_provider(std::shared_ptr<nervana::train_base> prov)
{
    _providers.push_back(prov);
    _startSignaled.push_back(0);

    _startInds.push_back(0);
    _endInds.push_back(0);
    _dataOffsets.push_back(0);
    _targetOffsets.push_back(0);
}

pyDecodeThreadPool::~pyDecodeThreadPool()
{
    if (_manager != 0) {
        _manager->join();
        delete _manager;
    }
    // The other thread objects are freed in the destructor
    // of the parent class.
}

void pyDecodeThreadPool::start()
{
    for (int i = 0; i < _count; i++) {
        _threads.push_back(new thread(&pyDecodeThreadPool::run, this, i));
    }
    _manager = new thread(&pyDecodeThreadPool::manage, this);
}

void pyDecodeThreadPool::stop()
{
    ThreadPool::stop();
    while (stopped() == false) {
        std::this_thread::yield();
        _in->advanceWritePos();
        _in->signalNonEmpty();
    }

    _stopManager = true;
    while (_managerStopped == false) {
        std::this_thread::yield();
        _in->advanceWritePos();
        _in->signalNonEmpty();
        _endSignaled++;
        _ended.notify_one();
    }
}

void pyDecodeThreadPool::run(int id)
{
    // Initialize worker threads by computing memory offsets for the
    // data this thread should work on
    assert(id < _count);
    _startInds[id] = id * _itemsPerThread;
    int itemCount = _itemsPerThread;
    if (id == _count - 1) {
        itemCount = _batchSize - id * _itemsPerThread;
    }

    _endInds[id] = _startInds[id] + itemCount;
    _dataOffsets[id] = _startInds[id] * _datumLen;
    _targetOffsets[id] = _startInds[id] * _targetLen;
    while (_done == false) {
        work(id);
    }

    _stopped[id] = true;
}

void pyDecodeThreadPool::work(int id)
{
    // Thread function.
    {
        unique_lock<mutex> lock(_mutex);
        while (_startSignaled[id] == 0) {
            _started.wait(lock);
            if (_done == true) {
                return;
            }
        }
        _startSignaled[id]--;
        assert(_startSignaled[id] == 0);
    }

    // No locking required because threads write into non-overlapping regions.
    BufferArray& outBuf = _out->getForWrite();
    char* dataBuf      = outBuf[0]->_data + _dataOffsets[id];
    char* targetBuf    = outBuf[1]->_data + _targetOffsets[id];

    for (int i = _startInds[id]; i < _endInds[id]; i++) {
        _providers[id]->provide_pair(i, _inputBuf, dataBuf, targetBuf);
        dataBuf   += _datumLen;
        targetBuf += _targetLen;
    }

    {
        lock_guard<mutex> lock(_mutex);
        _endSignaled++;
        assert(_endSignaled <= _count);
    }
    _ended.notify_one();
}

void pyDecodeThreadPool::produce()
{
    // Produce a minibatch.
    {
        unique_lock<mutex> lock(_out->getMutex());
        while (_out->full() == true) {
            _out->waitForNonFull(lock);
        }
        {
            lock_guard<mutex> lock(_mutex);
            for (unsigned int i = 0; i < _startSignaled.size(); i++) {
                _startSignaled[i] = 1;
            }
        }
        _started.notify_all();
        {
            unique_lock<mutex> lock(_mutex);
            while (_endSignaled < _count) {
                _ended.wait(lock);
            }
            _endSignaled = 0;
        }
        // At this point, we have decoded data for the whole minibatch.
        BufferArray& outBuf = _out->getForWrite();

        // Copy to device.
        _pbe->call_backend_transfer(outBuf, _bufferIndex);

        _bufferIndex = (_bufferIndex == 0) ? 1 : 0;
        _out->advanceWritePos();
    }
    _out->signalNonEmpty();
}

void pyDecodeThreadPool::consume()
{
    // Consume an input buffer.
    {
        unique_lock<mutex> lock(_in->getMutex());
        while (_in->empty() == true) {
            _in->waitForNonEmpty(lock);
            if (_stopManager == true) {
                return;
            }
        }
        _inputBuf = &_in->getForRead();
        produce();
        _in->advanceReadPos();
    }
    _in->signalNonFull();
}

void pyDecodeThreadPool::manage()
{
    // Thread function.
    // int result = _device->init();
    int result = 0;
    if (result != 0) {
        _stopManager = true;
    }
    while (_stopManager == false) {
        consume();
    }
    _managerStopped = true;
}


ReadThread::ReadThread(const shared_ptr<host_buffer_pool>& out,
                       const shared_ptr<BatchIterator>& batch_iterator)
: ThreadPool(1), _out(out), _batch_iterator(batch_iterator)
{
    assert(_count == 1);
}

void ReadThread::work(int id)
{
    // Fill input buffers.
    {
        unique_lock<mutex> lock(_out->getMutex());
        while (_out->full() == true) {
            _out->waitForNonFull(lock);
        }
        _batch_iterator->read(_out->getForWrite());
        _out->advanceWritePos();
    }
    _out->signalNonEmpty();
}


PyLoader::PyLoader(const char* pyloaderConfigString, PyObject *pbe)
: _pbe(pbe)
{
    _lcfg_json = nlohmann::json::parse(pyloaderConfigString);

    _lcfg = make_shared<pyLoaderConfig>();
    _lcfg->set_config(_lcfg_json);

    _batchSize = _lcfg->minibatch_size;

    // the manifest defines which data should be included in the dataset
    _manifest = make_shared<Manifest>(_lcfg->manifest_filename,
                                      _lcfg->shuffle_manifest,
                                      _lcfg->random_seed);

    if(_manifest->getSize() == 0) {
        throw std::runtime_error("manifest file is empty");
    }

    shared_ptr<BatchLoader> batchLoader = make_shared<BatchFileLoader>(
        _manifest, _lcfg->subset_percent
    );

    if(_lcfg->cache_directory.length() > 0) {
        batchLoader = make_shared<BatchLoaderCPIOCache>(_lcfg->cache_directory,
                                                        _manifest->hash(),
                                                        _manifest->version(),
                                                        batchLoader);
    }

    if (_lcfg->shuffle_every_epoch) {
        _batch_iterator = make_shared<ShuffledBatchIterator>(batchLoader,
                                                             _lcfg->macrobatch_size,
                                                             _lcfg->random_seed);
    } else {
        _batch_iterator = make_shared<SequentialBatchIterator>(batchLoader,
                                                               _lcfg->macrobatch_size);
    }

    _batch_iterator = make_shared<MinibatchIterator>(_batch_iterator, _lcfg->minibatch_size);
}

int PyLoader::start()
{
    _first = true;
    try {
        int ncores         = thread::hardware_concurrency();
        int itemsPerThread = (_batchSize - 1) /  ncores + 1;
        int nthreads       = (_batchSize - 1) / itemsPerThread + 1;
        nthreads           = std::min(nthreads, _batchSize);

        if(_lcfg_json["data_config"] == nullptr) {
            throw std::runtime_error("missing PyLoader config parameter data_config");
        }

        if(_lcfg_json["target_config"] == nullptr) {
            throw std::runtime_error("missing PyLoader config parameter target_config");
        }

        _dtm_config = nervana::config_factory::create(_lcfg_json["data_config"]);
        _tgt_config = nervana::config_factory::create(_lcfg_json["target_config"]);

        // Bind the python backend here
        _pyBackend = make_shared<pyBackendWrapper>(_pbe, _dtm_config, _tgt_config, _batchSize);

        // Start the read buffers off with a reasonable size. They will get resized as needed.
        _readBufs = make_shared<host_buffer_pool>(_dtm_config->get_size_bytes() * _batchSize / 8,
                                            _tgt_config->get_size_bytes() * _batchSize);
        _readThread = unique_ptr<ReadThread>(new ReadThread(_readBufs, _batch_iterator));

        _decodeBufs = make_shared<host_buffer_pool>(_dtm_config->get_size_bytes() * _batchSize,
                                              _tgt_config->get_size_bytes() * _batchSize,
                                              _pyBackend->use_pinned_memory());

        _decodeThreads = unique_ptr<pyDecodeThreadPool>(
                            new pyDecodeThreadPool(nthreads, _readBufs, _decodeBufs, _pyBackend));

        // Now add providers
        for (int i=0; i<nthreads; i++) {
            _decodeThreads->add_provider(nervana::train_provider_factory::create(_lcfg_json));
        }

    } catch(std::bad_alloc&) {
        return -1;
    }
    _decodeThreads->start();
    _readThread->start();
    return 0;
}

void PyLoader::stop()
{
    _readThread->stop();
    while (_readThread->stopped() == false) {
        std::this_thread::yield();
        drain();
    }
    while ((_decodeBufs->empty() == false) ||
           (_readBufs->empty() == false)) {
        drain();
    }
    _decodeThreads->stop();

    _readBufs      = nullptr;
    _readThread    = nullptr;
    _decodeBufs    = nullptr;
    _decodeThreads = nullptr;
    _pyBackend     = nullptr;
}

int PyLoader::reset()
{
    stop();
    _batch_iterator->reset();
    start();
    return 0;
}

PyObject* PyLoader::next(int bufIdx)
{
    unique_lock<mutex> lock(_decodeBufs->getMutex());
    if (_first == true) {
        _first = false;
    } else {
        // Unlock the buffer used for the previous minibatch.
        _decodeBufs->advanceReadPos();
        _decodeBufs->signalNonFull();
    }
    while (_decodeBufs->empty()) {
        _decodeBufs->waitForNonEmpty(lock);
    }
    return _pyBackend->get_dtm_tgt_pair(bufIdx);
}

void PyLoader::drain()
{
    {
        unique_lock<mutex> lock(_decodeBufs->getMutex());
        if (_decodeBufs->empty() == true) {
            return;
        }
        _decodeBufs->advanceReadPos();
    }
    _decodeBufs->signalNonFull();
}
