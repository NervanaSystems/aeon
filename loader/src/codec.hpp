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

#include <sstream>
#include <mutex>

#include "media.hpp"
#include "etl_audio.hpp"

#ifdef __cplusplus
extern "C"
{
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/common.h>
    #include <libavutil/error.h>
}
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 28, 1)
#define av_frame_alloc  avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif

namespace nervana {
    namespace audio {
        class config;
    }
}

typedef uint8_t uchar;
using std::mutex;

void raise_averror(const char* prefix, int errnum);
int lockmgr(void **p, enum AVLockOp op);

class Codec {
public:
    Codec(std::shared_ptr<const nervana::audio::config> params);
    std::shared_ptr<RawMedia> decode(const char* item, int itemSize);

private:
    void decodeFrame(AVPacket* packet, int stream, int itemSize);

private:
    std::shared_ptr<RawMedia>   _raw;
    AVMediaType                 _mediaType;
    AVFormatContext*            _format;
    AVCodecContext*             _codec;
    mutex                       _mutex;
    static int                  _init;
};
