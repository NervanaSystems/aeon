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

// #include "media.hpp"
#include "matrix.hpp"
#include "device.hpp"
#include "loader.hpp"
#include "batch_loader_cpio_cache.hpp"
#include "sequential_batch_iterator.hpp"
#include "shuffled_batch_iterator.hpp"

using namespace std;

DecodeThreadPool::DecodeThreadPool(int count, int batchSize,
                 int datumSize, int datumTypeSize,
                 int targetSize, int targetTypeSize,
                 const std::shared_ptr<BufferPool>& in, const std::shared_ptr<BufferPool>& out,
                 const std::shared_ptr<Device>& device,
                 std::string configString)
                 // MediaParams* mediaParams)
: ThreadPool(count),
  _itemsPerThread((batchSize - 1) / count + 1),
  _in(in), _out(out), _endSignaled(0),
  _manager(0), _stopManager(false), _managerStopped(false), _inputBuf(0),
  _bufferIndex(0), _batchSize(batchSize),
  _datumSize(datumSize), _datumTypeSize(datumTypeSize),
  _targetSize(targetSize), _targetTypeSize(targetTypeSize),
  _datumLen(datumSize * datumTypeSize),
  _targetLen(targetSize * targetTypeSize),
  _device(device) {
    assert(_itemsPerThread * count >= _batchSize);
    assert(_itemsPerThread * (count - 1) < _batchSize);
    for (int i = 0; i < count; i++) {
        // _media.push_back(Media::create(mediaParams, 0, i));
        auto prov = make_shared<nervana::image_decoder>(configString, _datumLen, _targetLen);
        _providers.push_back(prov);
        _startSignaled.push_back(0);
        _startInds.push_back(0);
        _endInds.push_back(0);
        _dataOffsets.push_back(0);
        _targetOffsets.push_back(0);
    }
}

DecodeThreadPool::~DecodeThreadPool() {
    if (_manager != 0) {
        _manager->join();
        delete _manager;
    }
    // The other thread objects are freed in the destructor
    // of the parent class.
}

void DecodeThreadPool::start() {
    for (int i = 0; i < _count; i++) {
        _threads.push_back(new thread(&DecodeThreadPool::run, this, i));
    }
    _manager = new thread(&DecodeThreadPool::manage, this);
}

void DecodeThreadPool::stop() {
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

void DecodeThreadPool::run(int id) {
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

void DecodeThreadPool::work(int id) {
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

    int start = _startInds[id];
    int end = _endInds[id];
    // No locking required because threads
    // write into non-overlapping regions.
    BufferPair& outBuf = _out->getForWrite();
    char* dataBuf = outBuf.first->_data + _dataOffsets[id];
    char* targetBuf = outBuf.second->_data + _targetOffsets[id];
    for (int i = start; i < end; i++) {
        _providers[id]->provide_pair(i, _inputBuf, dataBuf, targetBuf);
        dataBuf += _datumLen;
        targetBuf += _targetLen;

        // // Handle the data.
        // int itemSize = 0;
        // char* item = _inputBuf->first->getItem(i, itemSize);
        // if (item == 0) {
        //     return;
        // }
        // _media[id]->transform(item, itemSize, dataBuf, _datumLen);
        // dataBuf += _datumLen;

        // // Handle the targets.
        // int targetLen = 0;
        // char* target = _inputBuf->second->getItem(i, targetLen);
        // memcpy(targetBuf, target, targetLen);
        // if (_targetLen > targetLen) {
        //     // Pad the rest of the buffer with zeros.
        //     memset(targetBuf + targetLen, 0, _targetLen - targetLen);
        // }
        // targetBuf += _targetLen;
    }

    {
        lock_guard<mutex> lock(_mutex);
        _endSignaled++;
        assert(_endSignaled <= _count);
    }
    _ended.notify_one();
}

void DecodeThreadPool::produce() {
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
        BufferPair& outBuf = _out->getForWrite();
        Matrix::transpose(outBuf.first->_data, _batchSize,
                          _datumSize, _datumTypeSize);
        Matrix::transpose(outBuf.second->_data, _batchSize,
                          _targetSize, _targetTypeSize);
        // Copy to device.
        _device->copyData(_bufferIndex, outBuf.first->_data,
                          outBuf.first->_size);
        _device->copyLabels(_bufferIndex, outBuf.second->_data,
                            outBuf.second->_size);
        _bufferIndex = (_bufferIndex == 0) ? 1 : 0;
        _out->advanceWritePos();
    }
    _out->signalNonEmpty();
}

void DecodeThreadPool::consume() {
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

void DecodeThreadPool::manage() {
    // Thread function.
    int result = _device->init();
    if (result != 0) {
        _stopManager = true;
    }
    while (_stopManager == false) {
        consume();
    }
    _managerStopped = true;
}

ReadThread::ReadThread(const shared_ptr<BufferPool>& out, const shared_ptr<BatchIterator>& batch_iterator)
: ThreadPool(1), _out(out), _batch_iterator(batch_iterator) {
    assert(_count == 1);
}

void ReadThread::work(int id) {
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

Loader::Loader(int miniBatchSize,
       bool shuffleManifest, bool shuffleEveryEpoch,
       int datumSize, int datumTypeSize,
       int targetSize, int targetTypeSize,
       int subsetPercent,
       const char* mediaConfigString,
       // MediaParams* mediaParams,
       DeviceParams* deviceParams,
       const char* manifestFilename,
       int macroBatchSize,
       const char* rootCacheDir,
       uint randomSeed)
: _first(true),
  _miniBatchSize(miniBatchSize),
  _datumSize(datumSize), _datumTypeSize(datumTypeSize),
  _targetSize(targetSize), _targetTypeSize(targetTypeSize),
  _readBufs(nullptr), _decodeBufs(nullptr), _readThread(nullptr), _decodeThreads(nullptr),
  _device(nullptr), _batch_iterator(nullptr), _mediaConfigString{mediaConfigString}
  {

    _device = Device::create(deviceParams);

    // the manifest defines which data should be included in the dataset
    _manifest = make_shared<Manifest>(manifestFilename, shuffleManifest, randomSeed);

    // batch loader provdes random access to blocks of data in the manifest
    auto batchLoader = make_shared<BatchLoaderCPIOCache>(
        rootCacheDir, _manifest->hash(), _manifest->version(),
        make_shared<BatchFileLoader>(_manifest, subsetPercent)
    );

    // _batch_iterator provides an unending iterator (shuffled or not) over
    // the batchLoader
    if(shuffleEveryEpoch) {
        _batch_iterator = make_shared<ShuffledBatchIterator>(
             batchLoader, macroBatchSize, randomSeed
        );
    } else {
        _batch_iterator = make_shared<SequentialBatchIterator>(
             batchLoader, macroBatchSize
        );
    }
}

Loader::~Loader() {
}

int Loader::start() {
    _first = true;
    try {
        int dataLen = _miniBatchSize * _datumSize * _datumTypeSize;
        int targetLen = _miniBatchSize * _targetSize * _targetTypeSize;
        // Start the read buffers off with a reasonable size. They will
        // get resized as needed.
        _readBufs = make_shared<BufferPool>(dataLen / 8, targetLen);
        _readThread = unique_ptr<ReadThread>(new ReadThread(_readBufs, _batch_iterator));
        bool pinned = (_device->_type != CPU);
        _decodeBufs = make_shared<BufferPool>(dataLen, targetLen, pinned);
        int numCores = thread::hardware_concurrency();
        int itemsPerThread = (_miniBatchSize - 1) /  numCores + 1;
        int threadCount =  (_miniBatchSize - 1) / itemsPerThread + 1;
        threadCount = std::min(threadCount, _miniBatchSize);
        _decodeThreads = unique_ptr<DecodeThreadPool>(new DecodeThreadPool(threadCount, _miniBatchSize,
                _datumSize, _datumTypeSize,
                _targetSize, _targetTypeSize,
                _readBufs, _decodeBufs, _device, _mediaConfigString));
    } catch(std::bad_alloc&) {
        return -1;
    }
    _decodeThreads->start();
    _readThread->start();
    return 0;
}

void Loader::stop() {
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

    _readBufs = nullptr;
    _readThread = nullptr;
    _decodeBufs = nullptr;
    _decodeThreads = nullptr;
}

int Loader::reset() {
    stop();
    _batch_iterator->reset();
    start();
    return 0;
}

void Loader::next(Buffer* dataBuf, Buffer* targetsBuf) {
    // Copy minibatch data into the buffers passed in.
    // Only used for testing purposes.
    {
        unique_lock<mutex> lock(_decodeBufs->getMutex());
        while (_decodeBufs->empty()) {
            _decodeBufs->waitForNonEmpty(lock);
        }
        Buffer* data = _decodeBufs->getForRead().first;
        memcpy(dataBuf->_data, data->_data, dataBuf->_size);
        Buffer* targets = _decodeBufs->getForRead().second;
        memcpy(targetsBuf->_data, targets->_data, targetsBuf->_size);
        _decodeBufs->advanceReadPos();
    }
    _decodeBufs->signalNonFull();
}

void Loader::next() {
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
}

std::shared_ptr<BatchIterator> Loader::getBatchIterator() {
    return _batch_iterator;
}

std::shared_ptr<Device> Loader::getDevice() {
    return _device;
}

int Loader::itemCount() {
    return _manifest->getSize();
}

void Loader::drain() {
    {
        unique_lock<mutex> lock(_decodeBufs->getMutex());
        if (_decodeBufs->empty() == true) {
            return;
        }
        _decodeBufs->advanceReadPos();
    }
    _decodeBufs->signalNonFull();
}
