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

typedef uint8_t uchar;
using std::mutex;

void raise_averror(const char* prefix, int errnum);
int lockmgr(void **p, enum AVLockOp op);

class Codec {
public:
    Codec(MediaParams* params) : _format(0), _codec(0) {
        if (params->_mtype == MediaType::VIDEO) {
            _mediaType = AVMEDIA_TYPE_VIDEO;
        } else if (params->_mtype == MediaType::AUDIO) {
            _mediaType = AVMEDIA_TYPE_AUDIO;
        }

        std::lock_guard<mutex> lock(_mutex);
        if (_init == 0) {
            av_register_all();
            av_lockmgr_register(lockmgr);
            av_log_set_level(AV_LOG_FATAL);
            _init = 1;
        }
    }

    RawMedia* decode(const char* item, int itemSize) {
        int errnum;

        _format = avformat_alloc_context();
        if (_format == 0) {
            throw std::runtime_error("Could not get context for decoding");
        }
        uchar* itemCopy = (uchar*) av_malloc(itemSize);
        if (itemCopy == 0) {
            throw std::runtime_error("Could not allocate memory");
        }

        memcpy(itemCopy, item, itemSize);
        _format->pb = avio_alloc_context(itemCopy, itemSize, 0, 0, 0, 0, 0);

        if ((errnum = avformat_open_input(&_format , "", 0, 0)) < 0) {
            raise_averror("Could not open input for decoding", errnum);
        }

        if ((errnum = avformat_find_stream_info(_format, 0)) < 0) {
            raise_averror("Could not find media information", errnum);
        }

        _codec = _format->streams[0]->codec;
        int stream = av_find_best_stream(_format, _mediaType, -1, -1, 0, 0);
        if (stream < 0) {
            raise_averror("Could not find media stream in input", stream);
        }

        errnum = avcodec_open2(_codec, avcodec_find_decoder(_codec->codec_id), 0);
        if (errnum < 0) {
            raise_averror("Could not open decoder", errnum);
        }

        if (_raw.size() == 0) {
            _raw.addBufs(_codec->channels, itemSize);
        } else {
            _raw.reset();
        }

        _raw.setSampleSize(av_get_bytes_per_sample(_codec->sample_fmt));
        assert(_raw.sampleSize() >= 0);
        AVPacket packet;
        while (av_read_frame(_format, &packet) >= 0) {
            decodeFrame(&packet, stream, itemSize);
        }

        avcodec_close(_codec);
        av_free(_format->pb->buffer);
        av_free(_format->pb);
        avformat_close_input(&_format);
        return &_raw;
    }

private:
    void decodeFrame(AVPacket* packet, int stream, int itemSize) {
        int frameFinished;
        if (packet->stream_index == stream) {
            AVFrame* frame = av_frame_alloc();
            int result = 0;
            if (_mediaType == AVMEDIA_TYPE_AUDIO) {
                result = avcodec_decode_audio4(_codec, frame,
                                               &frameFinished, packet);
            } else {
                throw std::runtime_error("Unsupported media");
            }

            if (result < 0) {
                throw std::runtime_error("Could not decode media stream");
            }

            if (frameFinished == true) {
                int frameSize = frame->nb_samples * _raw.sampleSize();
                if (_raw.bufSize() < _raw.dataSize() + frameSize) {
                    _raw.growBufs(itemSize);
                }
                _raw.fillBufs((char**) frame->data, frameSize);
            }
            av_frame_free(&frame);
        }
        av_free_packet(packet);
    }

private:
    RawMedia                    _raw;
    AVMediaType                 _mediaType;
    AVFormatContext*            _format;
    AVCodecContext*             _codec;
    mutex                       _mutex;
    static int                  _init;
};
