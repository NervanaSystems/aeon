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

#include "archive.hpp"
#include "media.hpp"
#include "matrix.hpp"
#include "device.hpp"

class DecodeThreadPool : public ThreadPool {
public:
    DecodeThreadPool(int count, int batchSize,
                     int datumSize, int datumTypeSize,
                     int targetSize, int targetTypeSize,
                     BufferPool& in, BufferPool& out,
                     Device* device,
                     MediaParams* mediaParams);
    virtual ~DecodeThreadPool();
    virtual void start();
    virtual void stop();

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
    BufferPool&                 _in;
    BufferPool&                 _out;
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
    int                         _datumSize;
    int                         _datumTypeSize;
    int                         _targetSize;
    int                         _targetTypeSize;
    // Datum length in bytes.
    int                         _datumLen;
    // Target length in bytes.
    int                         _targetLen;
    Device*                     _device;
    Media**                     _media;
};

class ReadThread: public ThreadPool {
public:
    ReadThread(BufferPool& out, Reader* reader);

protected:
    virtual void work(int id);
    void produce();

private:
    ReadThread();
    ReadThread(const ReadThread&);
    BufferPool&                 _out;
    Reader*                     _reader;
};

class Loader {
public:
    Loader(int* itemCount, int batchSize,
           const char* repoDir, const char* archiveDir,
           const char* indexFile, const char* archivePrefix,
           bool shuffle, bool reshuffle,
           int startFileIdx,
           int datumSize, int datumTypeSize,
           int targetSize, int targetTypeSize,
           int targetConversion, int subsetPercent,
           MediaParams* mediaParams,
           DeviceParams* deviceParams,
           MediaParams* ingestParams);

    virtual ~Loader();
    int start();
    void stop();
    int reset();
    void next(Buffer<char>* dataBuf, Buffer<char>* targetsBuf);
    void next();
;    Reader* getReader();
;    Device* getDevice();

private:
    void drain();
;
private:
    Loader();
    Loader(const Loader&);
    bool                        _first;
    int                         _batchSize;
    int                         _datumSize;
    int                         _datumTypeSize;
    int                         _targetSize;
    int                         _targetTypeSize;
    BufferPool*                 _readBufs;
    BufferPool*                 _decodeBufs;
    ReadThread*                 _readThread;
    DecodeThreadPool*           _decodeThreads;
    Device*                     _device;
    Reader*                     _reader;
    MediaParams*                _mediaParams;
};
