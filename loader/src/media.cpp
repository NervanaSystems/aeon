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
#include <cstdio>

#include "loader.hpp"

#if HAS_IMGLIB
#include "image.hpp"
#endif

#if HAS_VIDLIB
#include "video.hpp"
#endif

#if HAS_AUDLIB
#include "audio.hpp"
#endif

#include "api.hpp"

using namespace std;

shared_ptr<nervana::train_base> Media::create(const string& configString) {
    return create(nlohmann::json::parse(configString));
}

shared_ptr<nervana::train_base> Media::create(nlohmann::json configJs) {
    shared_ptr<nervana::train_base> rc;
    string mediaType = configJs["media"];
    cout << "media type " << mediaType << endl;
    if( mediaType == "image" ) {
        rc = make_shared<nervana::image_decoder>(configJs, 100, 100); // wtf: replace constants
    } else {
        rc = nullptr;
    }
    return rc;
}



//std::shared_ptr<Media> Media::create(MediaParams* params, MediaParams* ingestParams, int id) {
//    switch (params->_mtype) {
//    case IMAGE:
//#if HAS_IMGLIB
//        return std::make_shared<Image>(reinterpret_cast<ImageParams*>(params),
//                         reinterpret_cast<ImageIngestParams*>(ingestParams),
//                         id);
//#else
//        {
//            std::string message = "OpenCV " UNSUPPORTED_MEDIA_MESSAGE;
//            throw std::runtime_error(message);
//        }
//#endif
//    case VIDEO:
//#if HAS_VIDLIB
//        return std::make_shared<Video>(reinterpret_cast<VideoParams*>(params), id);
//#else
//        {
//            std::string message = "Video " UNSUPPORTED_MEDIA_MESSAGE;
//            throw std::runtime_error(message);
//        }
//#endif
//    case AUDIO:
//#if HAS_AUDLIB
//        return std::make_shared<Audio>(reinterpret_cast<AudioParams*>(params), id);
//#else
//        {
//            std::string message = "Audio " UNSUPPORTED_MEDIA_MESSAGE;
//            throw std::runtime_error(message);
//        }
//#endif
//    default:
//        throw std::runtime_error("Unknown media type");
//    }
//    return 0;
//}

RawMedia::RawMedia() : _bufSize(0), _dataSize(0), _sampleSize(0) {
}

RawMedia::~RawMedia() {
    for (uint i = 0; i < _bufs.size(); i++) {
        delete[] _bufs[i];
    }
}

void RawMedia::reset() {
    _dataSize = 0;
}

void RawMedia::addBufs(int count, int size) {
    for (int i = 0; i < count; i++) {
        _bufs.push_back(new char[size]);
    }
    _bufSize = size;
}

void RawMedia::fillBufs(char** frames, int frameSize) {
    for (uint i = 0; i < _bufs.size(); i++) {
        memcpy(_bufs[i] + _dataSize, frames[i], frameSize);
    }
    _dataSize += frameSize;
}

void RawMedia::growBufs(int grow) {
    for (uint i = 0; i < _bufs.size(); i++) {
        char* buf = new char[_bufSize + grow];
        memcpy(buf, _bufs[i], _dataSize);
        delete[] _bufs[i];
        _bufs[i] = buf;
    }
    _bufSize += grow;
}

void RawMedia::setSampleSize(int sampleSize) {
    _sampleSize = sampleSize;
}

int RawMedia::size() {
    return _bufs.size();
}

char* RawMedia::getBuf(int idx) {
    return _bufs[idx];
}

int RawMedia::bufSize() {
    return _bufSize;
}

int RawMedia::dataSize() {
    return _dataSize;
}

int RawMedia::sampleSize() {
    return _sampleSize;
}

void RawMedia::copyData(char* buf, int bufSize) {
    if (_dataSize * (int) _bufs.size() > bufSize) {
        std::stringstream ss;
        ss << "Buffer too small to copy decoded data. Buffer size " <<
               bufSize << " Data size " << _dataSize * _bufs.size();
        throw std::runtime_error(ss.str());
    }

    for (uint i = 0; i < _bufs.size(); i++) {
        memcpy(buf, _bufs[i], _dataSize);
        buf += _dataSize;
    }
}
