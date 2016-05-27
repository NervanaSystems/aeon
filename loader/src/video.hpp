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

#include "media.hpp"
#include "image.hpp"
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavutil/imgutils.h>
    #include <libswscale/swscale.h>
}

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 28, 1)
#define av_frame_alloc  avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif

class VideoParams : public MediaParams {
public:
    void dump();

public:
    ImageParams                 _frameParams;
    int                         _framesPerClip;
};


class Video : public Media {
public:
   Video(VideoParams *params, int id);
    virtual ~Video();

public:
    void transform(char* item, int itemSize, char* buf, int bufSize);
    void ingest(char** dataBuf, int* dataBufLen, int* dataLen);

private:
    void decode(char* item, int itemSize, char* buf);
    int findVideoStream(AVCodecContext* &codecCtx, AVFormatContext* formatCtx);
    void convertFrameFormat(AVCodecContext* codecCtx, AVPixelFormat pFormat,
                            AVFrame* &pFrame, AVFrame* &pFrameRGB);
    void writeFrameToBuf(cv::Mat frame, char* buf, int frameIdx, int channelSize);

private:
    VideoParams*                _params;
    Image*                      _imgDecoder;
    int                         _imgSize;
    int                         _decodedSize;
};
