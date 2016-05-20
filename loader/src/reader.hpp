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

#include "buffer.hpp"

enum ConversionType {
    NO_CONVERSION = 0,
    ASCII_TO_BINARY = 1
};

class IndexElement {
public:
    IndexElement() {
    }

public:
    std::string                 _fileName;
    std::vector<std::string>    _targets;
};

class Index {
public:
    Index();

    virtual ~Index();

    void addElement(std::string& line);

    IndexElement* operator[] (int idx);

    uint size();

    void shuffle();

public:
    std::vector<IndexElement*>       _elements;
};

class Reader {
public:
    Reader(int batchSize, const char* repoDir, const char* indexFile,
           bool shuffle, bool reshuffle, int subsetPercent);

    virtual ~Reader();
    virtual int read(BufferPair& buffers) = 0;
    virtual int reset() = 0;

    virtual int totalDataSize();

    virtual int totalTargetsSize();

    static bool exists(const std::string& fileName);

protected:
    // Number of items to read at a time.
    int                         _batchSize;
    std::string                 _repoDir;
    std::string                 _indexFile;
    bool                        _shuffle;
    bool                        _reshuffle;
    int                         _subsetPercent;
    // Total number of items.
    int                         _itemCount;
};

class FileReader : public Reader {
public:
    FileReader(int* itemCount, int batchSize,
               const char* repoDir, const char* indexFile,
               bool shuffle, int targetTypeSize, int targetConversion);

    int read(BufferPair& buffers);

    int next(char** dataBuf, char** targetBuf,
             int* dataBufLen, int* targetBufLen,
             int* dataLen, int* targetLen);

    bool eos();

    int reset();

private:
    void resize(char** buf, int* len, int newLen);

    void loadIndex();

private:
    Index                       _index;
    int                         _itemIdx;
    std::ifstream               _ifs;
    int                         _targetTypeSize;
    int                         _targetConversion;
};
