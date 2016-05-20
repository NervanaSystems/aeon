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

using std::string;
using std::ifstream;
using std::ofstream;
using std::ios;
using std::stringstream;
using std::vector;
using std::map;

#define ARCHIVE_ITEM_COUNT  4096

typedef std::pair<std::unique_ptr<ByteVect>,std::unique_ptr<ByteVect>> DataPair;

class Writer {
public:
    virtual int write() = 0;
};

class WriteThread : public ThreadPool {
public:
    WriteThread(Writer* writer)
    : ThreadPool(1), _writer(writer) {
    }

protected:
    virtual void work(int id) {
        int result = _writer->write();
        if (result != 0) {
            stop();
        }
    }

private:
    Writer*                     _writer;
};

class ArchiveWriter : public Writer {
public:
    ArchiveWriter(int batchSize, const char* repoDir, const char* archiveDir,
                  const char* indexFile, const char* archivePrefix,
                  bool shuffle,
                  MediaParams* params, MediaParams* ingestParams,
                  int targetTypeSize, int targetConversion)
    : _batchSize(batchSize),
      _repoDir(repoDir), _archiveDir(archiveDir),
      _indexFile(indexFile),
      _archivePrefix(archivePrefix),
      _fileIdx(0), _itemCount(0), _started(false),
      _dataBuf(0), _targetBuf(0), _dataBufLen(0), _targetBufLen(0) {
        _media = Media::create(params, ingestParams, 0);
        _writeThread = new WriteThread(this);
        _reader = new FileReader(&_itemCount, 1, repoDir, indexFile, shuffle,
                                 targetTypeSize, targetConversion);
        if (Reader::exists(_archiveDir) == true) {
            return;
        }
        int result = mkdir(_archiveDir.c_str(), 0755);
        if (result != 0) {
            stringstream ss;
            ss << "Could not create " <<  _archiveDir;
            throw std::ios_base::failure(ss.str());
        }

    }

    virtual ~ArchiveWriter() {
        _writeThread->stop();
        delete _writeThread;
        delete _reader;
        delete[] _targetBuf;
        delete[] _dataBuf;
        delete _media;
    }

    void waitFor(string& name) {
        if (_started == false) {
            start();
        }

        unique_lock<mutex> lock(_mutex);
        while (_reader->exists(name) == false) {
            _write.wait(lock);
        }
    }

    int write() {
        if (_reader->eos() == true) {
            return 1;
        }
        stringstream    fileName;
        fileName << _archiveDir << '/' << _archivePrefix
                 << _fileIdx++ << ".cpio";

        if (Reader::exists(fileName.str()) == true) {
            return 0;
        }
        BatchFile       batchFile(fileName.str(), "");
        for (int i = 0; i < _batchSize; i++) {
            int dataLen = 0;
            int targetLen = 0;
            int result = _reader->next(&_dataBuf, &_targetBuf,
                                       &_dataBufLen, &_targetBufLen,
                                       &dataLen, &targetLen);
            if (result != 0) {
                break;
            }
            // TODO: make this multithreaded.
            _media->ingest(&_dataBuf, &_dataBufLen, &dataLen);
            batchFile.writeItem(_dataBuf, _targetBuf,
                                dataLen, targetLen);
        }

        {
            unique_lock<mutex> lock(_mutex);
            batchFile.close();
        }
        _write.notify_one();
        return 0;
    }

private:
    void start() {
        _writeThread->start();
        _started = true;
    }

private:
    int                         _batchSize;
    string                      _repoDir;
    string                      _archiveDir;
    string                      _indexFile;
    string                      _archivePrefix;
    // Index of current archive file.
    int                         _fileIdx;
    // Total number of items in this dataset.
    int                         _itemCount;
    bool                        _started;
    mutex                       _mutex;
    condition_variable          _write;
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
                  int targetConversion)
    : Reader(batchSize, repoDir, indexFile, shuffle, reshuffle, subsetPercent),
      _archiveDir(archiveDir), _indexFile(indexFile),
      _archivePrefix(archivePrefix),
      _startFileIdx(startFileIdx),
      _fileIdx(startFileIdx), _itemIdx(0), _itemsLeft(0), _archiveWriter(nullptr),
      _active(true), _shuffle(shuffle), _readQueue(), _readQueueMutex(), _readDataReadyMutex(),
      _readDataRequest(), _readDataReady(), _readMutex(), _fileListIndex(-1),
      _readThread(nullptr), _readAheadSize(1000) {
        getFileList();
        if (*itemCount == 0) {
            *itemCount = getCount();
            // Create a writer just in case. It will only be used if archive
            // files are missing or damaged.
            // _archiveWriter = new ArchiveWriter(ARCHIVE_ITEM_COUNT,
            //         repoDir, archiveDir, indexFile, archivePrefix,
            //         shuffle, params, ingestParams,
            //         targetTypeSize, targetConversion);
        }

        _itemCount = *itemCount;
        assert(_itemCount != 0);

        _logFile.open("test.log");

        _readThread = new std::thread(readThreadEntry,this);
    }

    virtual ~ArchiveReader() {
        killReadThread();
        delete _readThread;
        delete _archiveWriter;
        _logFile.close();
    }

    int read(BufferPair& buffers) {
        int offset = 0;
        while (offset < _batchSize) {
            int count = _batchSize - offset;
            int result;
            result = read(buffers, count);
            if (result == -1) {
                return -1;
            }
            offset += result;
        }

        assert(offset == _batchSize);
        assert(_itemIdx <= _itemCount);
        return 0;
    }

    int reset() {
        _fileIdx = _startFileIdx;
        _itemIdx = 0;

        _readQueueMutex.lock();
        _readQueue.clear();
        _readQueueMutex.unlock();
        _fileListIndex = 0;

        return 0;
    }

    int itemCount() {
        return _batchFile.itemCount();
    }

    int maxDatumSize() {
        return _batchFile.maxDatumSize();
    }

    int maxTargetSize() {
        return _batchFile.maxTargetSize();
    }

    int totalDataSize() {
        return _batchFile.totalDataSize();
    }

    int totalTargetsSize() {
        return _batchFile.totalTargetsSize();
    }

    void addFiles( const std::vector<string>& files ) {
        std::lock_guard<mutex> lk(_fileListMutex);
        _fileList.insert( _fileList.end(), files.begin(), files.end() );
        shuffleFileList();
    }

private:
    int getCount() {
        // ifstream ifs(_indexFile);
        // if (!ifs) {
        //     stringstream ss;
        //     ss << "Could not open " << _indexFile;
        //     throw std::ios_base::failure(ss.str());
        // }

        // string  line;
        // int     count = 0;
        // std::getline(ifs, line);
        // while (std::getline(ifs, line)) {
        //     if (line[0] == '#') {
        //         continue;
        //     }
        //     count++;
        // }

        int count = 0;
        for( const std::string& f : _fileList ) {
            BatchFile b;
            b.openForRead(f);
            count += b.itemCount();
        }

        if (_subsetPercent != 100) {
            count  = (count * _subsetPercent) / 100;
        }
        if (count == 0) {
            stringstream ss;
            ss << "Index file is empty: " << _indexFile;
            throw std::runtime_error(ss.str());
        }

        return count;
    }

    int read(BufferPair& buffers, int count) {
        while((int)_readQueue.size() < count) {
            //std::cout << "data starvation" << std::endl;
            std::unique_lock<std::mutex> waitLock(_readDataReadyMutex);
            _readDataRequest.notify_one();
            _readDataReady.wait(waitLock);
        }
        for (int i=0; i<count; ++i) {
            const DataPair& d = _readQueue[0];
            buffers.first->read(d.first->data(), d.first->size());
            buffers.second->read(d.second->data(), d.second->size());
            _readQueue.pop_front();
        }
        _readDataRequest.notify_all();
        return count;
    }

    static void readThreadEntry( ArchiveReader* ar ) {
        ar->readThread();
    }

    void killReadThread() {
        std::unique_lock<std::mutex> lk(_readMutex);
        _active = false;
        lk.unlock();
        _readDataRequest.notify_all();
        _readThread->join();
    }

    void readThread() {
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        logSeed(seed);
        std::minstd_rand0 rand(seed);
        while(_active) {
            if( _readQueue.size() < _readAheadSize ) {
                _fileListMutex.lock();
                if( _fileListIndex >= _fileList.size() ) {
                    _fileListIndex = 0;
                    shuffleFileList();
                }
                string fileName = _fileList[ _fileListIndex++ ];
                _fileListMutex.unlock();
                _fileIdx++;
                if ((Reader::exists(fileName) == false) && (_archiveWriter != 0)) {
                    _archiveWriter->waitFor(fileName);
                }

                BatchFile b;
                b.openForRead(fileName);
                logCurrentFile(fileName);

                // Something larger than 1 to force reading a second macroblock
                _readAheadSize = std::max<size_t>(_readAheadSize, b.itemCount() * 1.5);

                vector<DataPair> tmpBuffer(b.itemCount());
                for( int i=0; i<b.itemCount(); i++ ) {
                    tmpBuffer[i] = b.readItem();
                }
                b.close();
                if(_shuffle) shuffle(tmpBuffer.begin(), tmpBuffer.end(), rand);
                _readQueueMutex.lock();
                for( size_t i=0; i<tmpBuffer.size(); i++ ) {
                    _readQueue.push_back( std::move(tmpBuffer[i]) );
                }
                _readQueueMutex.unlock();
                _readDataReady.notify_all();
            }
            std::unique_lock<std::mutex> lk(_readMutex);
            if(!_active) {
                std::cout << "not active before wait" << std::endl;
                break;
            }
            _readDataRequest.wait(lk);
       }
    }

    void getFileList() {
        DIR *dir;
        struct dirent *ent;
        _fileList.clear();
        if ((dir = opendir (_archiveDir.c_str())) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                if( testFileName(ent->d_name) ) {
                    string path = _archiveDir + "/" + ent->d_name;
                    _fileList.push_back( path );
                }
            }
            closedir (dir);
        }
        else {
            perror ("error getting file list");
        }
    }

    bool testFileName( const string& s ) {
        bool rc = false;
        // sigh, gcc still does not support c++11 regex
        if( s.length() > 5 && s.length() > _archivePrefix.length() &&
            s.substr(s.length()-5,5) == ".cpio" &&
            s.substr(0,_archivePrefix.length()) == _archivePrefix ) {
            rc = true;
        }
        return rc;
    }

    // This method is called by mutiple threads so make sure you lock the call with _fileListMutex
    void shuffleFileList() {
        static auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        static std::minstd_rand0 rand(seed);
        std::shuffle( _fileList.begin(), _fileList.end(), rand );
    }

    void logCurrentFile( const std::string& file ) {
        _logFile << file << "\n";
    }

    void logSeed( unsigned int seed ) {
        _logFile << seed << "\n";
    }

private:
    string                      _archiveDir;
    string                      _indexFile;
    string                      _archivePrefix;
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
    std::vector<string>         _fileList;
    std::thread*                _readThread;
    size_t                      _readAheadSize;
    std::ofstream               _logFile;
};
