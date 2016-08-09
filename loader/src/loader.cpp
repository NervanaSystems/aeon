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
#include "loader.hpp"
#include "block_loader_cpio_cache.hpp"
#include "block_iterator_sequential.hpp"
#include "block_iterator_shuffled.hpp"
#include "batch_iterator.hpp"
#include "nds_manifest.hpp"
#include "block_loader_nds.hpp"

using namespace std;

decode_thread_pool::decode_thread_pool(int count,
                                       const shared_ptr<buffer_pool_in>& in,
                                       const shared_ptr<buffer_pool_out>& out,
                                       const shared_ptr<python_backend>& pbe)
: thread_pool(count), _in(in), _out(out),
  _python_backend(pbe), _batchSize(_python_backend->_batchSize)
{
    _itemsPerThread = (_batchSize - 1) / _count + 1;
    assert(_itemsPerThread * count >= _batchSize);
    assert(_itemsPerThread * (count - 1) < _batchSize);
}


void decode_thread_pool::add_provider(std::shared_ptr<nervana::provider_interface> prov)
{
    _providers.push_back(prov);
    _startSignaled.push_back(0);

    _startInds.push_back(0);
    _endInds.push_back(0);
}

decode_thread_pool::~decode_thread_pool()
{
    if (_manager != 0) {
        _manager->join();
        delete _manager;
    }
    // The other thread objects are freed in the destructor
    // of the parent class.
}

void decode_thread_pool::start()
{
    for (int i = 0; i < _count; i++) {
        _threads.push_back(new thread(&decode_thread_pool::run, this, i));
    }
    _manager = new thread(&decode_thread_pool::manage, this);
}

void decode_thread_pool::stop()
{
    thread_pool::stop();
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

void decode_thread_pool::run(int id)
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
        cerr << "fatal exception in decode_thread_pool::run: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}

void decode_thread_pool::work(int id)
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

void decode_thread_pool::produce()
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
        _python_backend->call_backend_transfer(outBuf, _bufferIndex);

        _bufferIndex = (_bufferIndex == 0) ? 1 : 0;
        _out->advanceWritePos();
    }
    _out->signalNonEmpty();
}

void decode_thread_pool::consume()
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

void decode_thread_pool::manage()
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
        cerr << "exception in decode_thread_pool::manage: " << e.what() << endl;
        // TODO: fail gracefully, not seg fault
    }
}


read_thread_pool::read_thread_pool(const shared_ptr<buffer_pool_in>& out,
                       const shared_ptr<batch_iterator>& b_it)
: thread_pool(1), _out(out), _batch_iterator(b_it)
{
    assert(_count == 1);
}

void read_thread_pool::work(int id)
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


loader::loader(const char* cfg_string, PyObject *py_obj_backend)
: _py_obj_backend(py_obj_backend)
{
    _lcfg_json = nlohmann::json::parse(cfg_string);
    loader_config lcfg(_lcfg_json);

    _batchSize = lcfg.minibatch_size;
    _single_thread_mode = lcfg.single_thread;
    shared_ptr<Manifest> base_manifest = nullptr;

    if(NDSManifest::isLikelyJSON(lcfg.manifest_filename)) {
        auto manifest = make_shared<NDSManifest>(lcfg.manifest_filename);

        // TODO: add shard_count/shard_index to cfg
        _block_loader = make_shared<block_loader_nds>(manifest->baseurl,
                                                      manifest->token,
                                                      manifest->collection_id,
                                                      lcfg.macrobatch_size);

        base_manifest = manifest;
    } else {
        // the manifest defines which data should be included in the dataset
        auto manifest = make_shared<CSVManifest>(lcfg.manifest_filename,
                                                 lcfg.shuffle_manifest);

        // TODO: make the constructor throw this error
        if(manifest->objectCount() == 0) {
            throw std::runtime_error("manifest file is empty");
        }

        _block_loader = make_shared<block_loader_file>(manifest,
                                                       lcfg.subset_fraction,
                                                       lcfg.macrobatch_size);
        base_manifest = manifest;
    }

    if(lcfg.cache_directory.length() > 0) {
        _block_loader = make_shared<block_loader_cpio_cache>(lcfg.cache_directory,
                                                             base_manifest->hash(),
                                                             base_manifest->version(),
                                                             _block_loader);
    }

    shared_ptr<block_iterator> block_iter;
    if (lcfg.shuffle_every_epoch) {
        block_iter = make_shared<block_iterator_shuffled>(_block_loader, lcfg.random_seed);
    } else {
        block_iter = make_shared<block_iterator_sequential>(_block_loader);
    }

    _batch_iterator = make_shared<batch_iterator>(block_iter, lcfg.minibatch_size);
}

int loader::start()
{
    _first = true;
    try {
        int ncores         = thread::hardware_concurrency();
        int itemsPerThread = (_batchSize - 1) /  ncores + 1;
        int nthreads       = (_batchSize - 1) / itemsPerThread + 1;
        nthreads           = _single_thread_mode ? 1 : std::min(nthreads, _batchSize);

        if (nthreads <= 0)
        {
            throw std::invalid_argument("Number of threads must be > 0");
        }

        vector<shared_ptr<nervana::provider_interface>> providers;
        for (int i=0; i<nthreads; i++) {
            providers.push_back(nervana::train_provider_factory::create(_lcfg_json));
        }

        // variable size buffers for reading encoded data (start off zero and grow as needed)
        _read_buffers = make_shared<buffer_pool_in>(providers[0]->num_inputs);
        _read_thread_pool = unique_ptr<read_thread_pool>(
                        new read_thread_pool(_read_buffers, _batch_iterator));

        // fixed size buffers for writing out decoded data
        const vector<nervana::shape_type>& oshapes = providers[0]->get_oshapes();
        vector<size_t> write_sizes;
        for (auto& o: oshapes)
        {
            write_sizes.push_back(o.get_byte_size());
        }

        // Bind the python backend here
        _python_backend = make_shared<python_backend>(_py_obj_backend, oshapes, _batchSize);

        // These are fixed size output buffers (need batchSize for stride)
        _decode_buffers = make_shared<buffer_pool_out>(write_sizes,
                                                       (size_t)_batchSize,
                                                       _python_backend->use_pinned_memory());

        _decode_thread_pool = unique_ptr<decode_thread_pool>(
                new decode_thread_pool(nthreads, _read_buffers, _decode_buffers, _python_backend));

        for (auto& p: providers)
        {
            _decode_thread_pool->add_provider(p);
        }

    } catch(std::bad_alloc&) {
        return -1;
    }
    _decode_thread_pool->start();
    _read_thread_pool->start();
    return 0;
}

void loader::stop()
{
    _read_thread_pool->stop();
    while (_read_thread_pool->stopped() == false)
    {
        std::this_thread::yield();
        drain();
    }
    while ((_decode_buffers->empty() == false) ||
           (_read_buffers->empty() == false))
    {
        drain();
    }
    _decode_thread_pool->stop();

    _read_thread_pool   = nullptr;
    _decode_buffers     = nullptr;
    _decode_thread_pool = nullptr;
    _python_backend         = nullptr;
}

int loader::reset()
{
    stop();
    _batch_iterator->reset();
    return start();
}

PyObject* loader::next(int bufIdx)
{
    unique_lock<mutex> lock(_decode_buffers->getMutex());
    if (_first == true) {
        _first = false;
    } else {
        // Unlock the buffer used for the previous minibatch.
        _decode_buffers->advanceReadPos();
        _decode_buffers->signalNonFull();
    }
    while (_decode_buffers->empty()) {
        _decode_buffers->waitForNonEmpty(lock);
    }
    // TODO: should this actually be somewhere above the various locks/signals?
    _decode_buffers->reraiseException();
    return _python_backend->get_host_tuple(bufIdx);
}

PyObject* loader::shapes()
{
    return _python_backend->get_shapes();
}

void loader::drain()
{
    {
        unique_lock<mutex> lock(_decode_buffers->getMutex());
        if (_decode_buffers->empty() == true) {
            return;
        }
        _decode_buffers->advanceReadPos();
    }
    _decode_buffers->signalNonFull();
}
