/*
 Copyright 2016 Nervana Systems Inc.
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

#include <vector>
#include <cstdio>
#include <stdexcept>
#include <cstring>
#include <sstream>

using std::vector;

class RawMedia {
public:
    RawMedia() : _bufSize(0), _dataSize(0), _bytesPerSample(0) {
    }

    virtual ~RawMedia() {
        for (uint i = 0; i < _bufs.size(); i++) {
            delete[] _bufs[i];
        }
    }

    void reset() {
        _dataSize = 0;
    }

    void addBufs(int count, int size) {
        for (int i = 0; i < count; i++) {
            _bufs.push_back(new char[size]);
       }
        _bufSize = size;
    }

    void fillBufs(char** frames, int frameSize) {
        // `frames` should contain one frame per channel of audio
        for (uint i = 0; i < _bufs.size(); i++) {
            memcpy(_bufs[i] + _dataSize, frames[i], frameSize);
        }
        _dataSize += frameSize;
    }

    void growBufs(int grow) {
        for (uint i = 0; i < _bufs.size(); i++) {
            char* buf = new char[_bufSize + grow];
            memcpy(buf, _bufs[i], _dataSize);
            delete[] _bufs[i];
            _bufs[i] = buf;
        }
        _bufSize += grow;
    }

    void setBytesPerSample(int bytesPerSample) {
        _bytesPerSample = bytesPerSample;
    }

    int size() {
        return _bufs.size();
    }

    char* getBuf(int idx) {
        return _bufs[idx];
    }

    int bufSize() {
        return _bufSize;
    }

    int dataSize() {
        return _dataSize;
    }

    int bytesPerSample() {
        return _bytesPerSample;
    }

    int numSamples() {
        return dataSize() / bytesPerSample();
    }

private:
    vector<char*>               _bufs;
    int                         _bufSize;
    int                         _dataSize;
    int                         _bytesPerSample;
};
