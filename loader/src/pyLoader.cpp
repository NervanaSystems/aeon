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
#include "nds_manifest.hpp"
#include "nds_batch_loader.hpp"

using namespace std;

pyDecodeThreadPool::pyDecodeThreadPool(int count,
                                       const shared_ptr<buffer_pool_in>& in,
                                       const shared_ptr<buffer_pool_out>& out,
                                       const shared_ptr<pyBackendWrapper>& pbe)
: ThreadPool(count), _in(in), _out(out), _pbe(pbe)
{
    _batchSize = _pbe->_batchSize;
    _itemsPerThread = (_batchSize - 1) / _count + 1;

    assert(_itemsPerThread * count >= _batchSize);
    assert(_itemsPerThread * (count - 1) < _batchSize);
}


void pyDecodeThreadPool::add_provider(std::shared_ptr<nervana::provider_interface> prov)
{
    _providers.push_back(prov);
    _startSignaled.push_back(0);

    _startInds.push_back(0);
    _endInds.push_back(0);
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
    try {
        assert(id < _count);
        _startInds[id] = id * _itemsPerThread;
        int itemCount = _itemsPerThread;
        if (id == _count - 1) {
            itemCount = _batchSize - id * _itemsPerThread;
        }

        _endInds[id] = _startInds[id] + itemCount;

        while (_done == false) {
            work(id);
        }

        _stopped[id] = true;
    } catch (std::exception& e) {
        cerr << "fatal exception in DecodeThreadPool::run: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
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
    try {
        for (int i = _startInds[id]; i < _endInds[id]; i++) {
            _providers[id]->provide(i, *_inputBuf, _out->getForWrite());
        }
    } catch (std::exception& e) {
        _out->writeException(std::current_exception());
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
    // lock on output buffers and copy to device
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
        buffer_out_array& outBuf = _out->getForWrite();

        // Do any messy cross datum stuff you may need to do that requires minibatch consistency
        _providers[0]->post_process(outBuf);

        // Copy to device.
        _pbe->call_backend_transfer(outBuf, _bufferIndex);

        _bufferIndex = (_bufferIndex == 0) ? 1 : 0;
        _out->advanceWritePos();
    }
    _out->signalNonEmpty();
}

void pyDecodeThreadPool::consume()
{
    // lock on input buffers and call produce
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
    try {
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
    } catch (std::exception& e) {
        cerr << "exception in DecodeThreadPool::manage: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}


ReadThread::ReadThread(const shared_ptr<buffer_pool_in>& out,
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

        try {
            _batch_iterator->read(_out->getForWrite());
        } catch(std::exception& e) {
            _out->writeException(std::current_exception());
        }

        _out->advanceWritePos();
    }
    _out->signalNonEmpty();
}


PyLoader::PyLoader(const char* pyloaderConfigString, PyObject *pbe)
: _pbe(pbe)
{
    _lcfg_json = nlohmann::json::parse(pyloaderConfigString);
    _lcfg = make_shared<pyLoaderConfig>(_lcfg_json);
    _batchSize = _lcfg->minibatch_size;

    shared_ptr<Manifest> base_manifest = nullptr;

    if(NDSManifest::isLikelyJSON(_lcfg->manifest_filename)) {
        auto manifest = make_shared<NDSManifest>(_lcfg->manifest_filename);

        // TODO: add shard_count/shard_index to cfg
        _batchLoader = make_shared<NDSBatchLoader>(manifest->baseurl,
                                                   manifest->token,
                                                   manifest->collection_id,
                                                   _lcfg->macrobatch_size);

        base_manifest = manifest;
    } else {
        // the manifest defines which data should be included in the dataset
        auto manifest = make_shared<CSVManifest>(_lcfg->manifest_filename,
                                                 _lcfg->shuffle_manifest);

        if(manifest->objectCount() == 0) {
            throw std::runtime_error("manifest file is empty");
        }

        _batchLoader = make_shared<BatchFileLoader>(
            manifest, _lcfg->subset_fraction, _lcfg->macrobatch_size
        );

        base_manifest = manifest;
    }

    if(_lcfg->cache_directory.length() > 0) {
        _batchLoader = make_shared<BatchLoaderCPIOCache>(_lcfg->cache_directory,
                                                         base_manifest->hash(),
                                                         base_manifest->version(),
                                                         _batchLoader);
    }

    if (_lcfg->shuffle_every_epoch) {
        _batch_iterator = make_shared<ShuffledBatchIterator>(_batchLoader,
                                                             _lcfg->random_seed);
    } else {
        _batch_iterator = make_shared<SequentialBatchIterator>(_batchLoader);
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
        if (_lcfg->single_thread)
        {
            nthreads = 1;
        }
        if (nthreads <= 0)
        {
            throw std::invalid_argument("Number of threads must be > 0");
        }

        vector<shared_ptr<nervana::provider_interface>> providers;
        for (int i=0; i<nthreads; i++) {
            try {
                providers.push_back(nervana::train_provider_factory::create(_lcfg_json));
            } catch (const std::invalid_argument e) {
                stringstream ss;
                ss << "exception while parsing provider_factory: ";
                ss << e.what();
                throw std::runtime_error(ss.str());
            }
        }

        // variable size buffers for reading encoded data (start off zero and grow as needed)
        const uint32_t nbuffers_in = providers[0]->num_inputs;
        vector<size_t> read_sizes;
        for (uint i=0; i<nbuffers_in; i++)
        {
            read_sizes.push_back(0);
        }
        _readBufs = make_shared<buffer_pool_in>(read_sizes);
        _readThread = unique_ptr<ReadThread>(new ReadThread(_readBufs, _batch_iterator));

        // fixed size buffers for writing out decoded data
        const vector<nervana::shape_type>& oshapes = providers[0]->get_oshapes();
        vector<size_t> write_sizes;
        for (auto& o: oshapes)
        {
            write_sizes.push_back(o.get_byte_size());
        }

        // Bind the python backend here
        _pyBackend = make_shared<pyBackendWrapper>(_pbe, oshapes, _batchSize);

        // These are fixed size output buffers (need batchSize for stride)
        _decodeBufs = make_shared<buffer_pool_out>(write_sizes, (size_t)_batchSize,
                                                   _pyBackend->use_pinned_memory());
        _decodeThreads = unique_ptr<pyDecodeThreadPool>(
                            new pyDecodeThreadPool(nthreads, _readBufs, _decodeBufs, _pyBackend));
        for (auto& p: providers)
        {
            _decodeThreads->add_provider(p);
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

    _readThread    = nullptr;
    _decodeBufs    = nullptr;
    _decodeThreads = nullptr;
    _pyBackend     = nullptr;
}

int PyLoader::reset()
{
    stop();
    _batch_iterator->reset();
    return start();
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
    // TODO: should this actually be somewhere above the various locks/signals?
    _decodeBufs->reraiseException();
    return _pyBackend->get_host_tuple(bufIdx);
}

PyObject* PyLoader::shapes() {
    return _pyBackend->get_shapes();
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
