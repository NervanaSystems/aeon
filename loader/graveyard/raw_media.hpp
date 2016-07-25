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

/*
 * RawMedia is a multichannel audio buffer.  It is used as a resting place
 * between decode from Codec and conversion to frequency domain with
 * Specgram.
 *
 * Codec re-uses RawMedia for multiple files.  reset() is available so that
 * the memory can be reused without de/re-allocation.
 *
 * _dataSize is used to keep track of how big the data in the buffer
 * actually is since the buffer may be bigger than necessary from storing
 * a large file previously.  Buffer size only ever decreases is setChannels
 * is called with decreasing number of channels.
 */

class RawMedia {
public:
    RawMedia() : _dataSize(0), _bytesPerSample(0) {
    }

    virtual ~RawMedia() {
    }

    void reset() {
        // keep memory allocated, but start future data at index 0.
        _dataSize = 0;
    }

    void appendFrames(char** frames, int frameSize) {
        for (uint i = 0; i < channels(); i++) {
            _bufs[i].insert(_bufs[i].begin() + _dataSize, frames[i], frames[i] + frameSize);
        }

        _dataSize += frameSize;
    }

    char* getBuf(int idx) {
        return _bufs[idx].data();
    }

    void setBytesPerSample(int bytesPerSample) {
        _bytesPerSample = bytesPerSample;
    }

    int bytesPerSample() {
        return _bytesPerSample;
    }

    void setChannels(int channels) {
        // set the number of channels
        _bufs.resize(channels);
    }

    int channels() {
        return _bufs.size();
    }

    int numSamples() {
        return _dataSize / bytesPerSample();
    }

private:
    // one buf per channel.  All bufs are of the same length.
    vector<vector<char>>        _bufs;
    int                         _dataSize;
    int                         _bytesPerSample;
};
