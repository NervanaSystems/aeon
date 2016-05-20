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

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <deque>
#include <random>
#include <chrono>
#include <dirent.h>
#include <math.h>
#include <algorithm>

#include "reader.hpp"
#include "threadpool.hpp"
#include "batchfile.hpp"
#include "media.hpp"
#include "event.hpp"

#define ARCHIVE_ITEM_COUNT  4096

typedef std::pair<std::unique_ptr<ByteVect>,std::unique_ptr<ByteVect>> DataPair;

class Writer {
public:
    virtual int write() = 0;
};

class WriteThread : public ThreadPool {
public:
    WriteThread(Writer* writer);

protected:
    virtual void work(int id);

private:
    Writer*                     _writer;
};

class ArchiveWriter : public Writer {
public:
    ArchiveWriter(int batchSize, const char* repoDir, const char* archiveDir,
                  const char* indexFile, const char* archivePrefix,
                  bool shuffle,
                  MediaParams* params, MediaParams* ingestParams,
                  int targetTypeSize, int targetConversion);

    virtual ~ArchiveWriter();

    void waitFor(std::string& name);

    int write();

private:
    void start();

private:
    int                         _batchSize;
    std::string                 _repoDir;
    std::string                 _archiveDir;
    std::string                 _indexFile;
    std::string                 _archivePrefix;
    // Index of current archive file.
    int                         _fileIdx;
    // Total number of items in this dataset.
    int                         _itemCount;
    bool                        _started;
    std::mutex                  _mutex;
    std::condition_variable     _write;
    WriteThread*                _writeThread;
    FileReader*                 _reader;
    char*                       _dataBuf;
    char*                       _targetBuf;
    int                         _dataBufLen;
    int                         _targetBufLen;
    Media*                      _media;
};

class ArchiveReader : public Reader {
public:
    ArchiveReader(int* itemCount, int batchSize,
                  const char* repoDir, const char* archiveDir,
                  const char* indexFile,
                  const char* archivePrefix,
                  bool shuffle, bool reshuffle,
                  int startFileIdx,
                  int subsetPercent,
                  MediaParams* params,
                  MediaParams* ingestParams,
                  int targetTypeSize,
                  int targetConversion);

    virtual ~ArchiveReader();

    int read(BufferPair& buffers);

    int reset();

    int itemCount();

    int maxDatumSize();

    int maxTargetSize();

    int totalDataSize();

    int totalTargetsSize();

    void addFiles( const std::vector<std::string>& files );

private:
    int getCount();

    int read(BufferPair& buffers, int count);

    static void readThreadEntry( ArchiveReader* ar );
    void killReadThread();

    void readThread();

    void getFileList();

    bool testFileName( const std::string& s );

    // This method is called by mutiple threads so make sure you lock the call with _fileListMutex
    void shuffleFileList();

    void logCurrentFile( const std::string& file );

    void logSeed( unsigned int seed );

private:
    std::string                 _archiveDir;
    std::string                 _indexFile;
    std::string                 _archivePrefix;
    int                         _startFileIdx;
    // Index of current archive file.
    int                         _fileIdx;
    // Index of current item.
    int                         _itemIdx;
    // Number of items left in the current archive.
    int                         _itemsLeft;
    BatchFile                   _batchFile;
    std::deque<DataPair>        _shuffleQueue;
    ArchiveWriter*              _archiveWriter;

    bool                        _active;
    bool                        _shuffle;
    std::deque<DataPair>        _readQueue;
    std::mutex                  _readQueueMutex;
    std::mutex                  _readDataReadyMutex;
    std::mutex                  _fileListMutex;
    std::condition_variable     _readDataRequest;
    std::condition_variable     _readDataReady;
    std::mutex                  _readMutex;
    size_t                      _fileListIndex;
    std::vector<std::string>    _fileList;
    std::thread*                _readThread;
    size_t                      _readAheadSize;
    std::ofstream               _logFile;
};
