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

#include "media.hpp"
#include "image.hpp"
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "video.hpp"

void VideoParams::dump() {
    _frameParams.dump();
    printf("frames per clip %d\n", _framesPerClip);
}

Video::Video(VideoParams *params, int id)
: _params(params) {
    assert(params->_mtype == VIDEO);
    assert(params->_frameParams._mtype == IMAGE);
    _imgDecoder = new Image(&(_params->_frameParams), 0, id);
    _imgSize = params->_frameParams.getSize().area();
    _decodedSize = _imgSize * params->_frameParams._channelCount ;
    av_register_all();
}

Video::~Video() {
    delete _imgDecoder;
}

void Video::transform(char* item, int itemSize, char* buf, int bufSize) {
    AVFormatContext* formatCtx = avformat_alloc_context();
    uchar* itemCopy = (unsigned char *) malloc(itemSize);
    memcpy(itemCopy, item, itemSize);
    formatCtx->pb = avio_alloc_context(itemCopy, itemSize, 0, itemCopy,
                                       NULL, NULL, NULL);

    avformat_open_input(&formatCtx , "", NULL, NULL);
    avformat_find_stream_info(formatCtx, NULL);

    AVCodecContext* codecCtx = NULL;
    int videoStream = findVideoStream(codecCtx, formatCtx);

    AVCodec* pCodec = avcodec_find_decoder(codecCtx->codec_id);
    avcodec_open2(codecCtx, pCodec, NULL);

    AVFrame* pFrameRGB = av_frame_alloc();
    AVPixelFormat pFormat = AV_PIX_FMT_BGR24;

    int numBytes = avpicture_get_size(pFormat, codecCtx->width, codecCtx->height);
    uint8_t* buffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));
    avpicture_fill((AVPicture*) pFrameRGB, buffer, pFormat,
                   codecCtx->width, codecCtx->height);
    int numFrames = formatCtx->streams[videoStream]->nb_frames;
    int channelSize = numFrames * _imgSize;

    int frameFinished;
    AVPacket packet;
    int frameIdx = 0;
    while (av_read_frame(formatCtx, &packet) >= 0) {

        if (packet.stream_index == videoStream) {

            AVFrame* pFrame = av_frame_alloc();
            avcodec_decode_video2(codecCtx, pFrame, &frameFinished, &packet);
            if (frameFinished) {
                convertFrameFormat(codecCtx, pFormat, pFrame, pFrameRGB);
                cv::Mat frame(pFrame->height, pFrame->width,
                          CV_8UC3, pFrameRGB->data[0]);
                writeFrameToBuf(frame, buf, frameIdx, channelSize);
                frameIdx++;
            }
            av_frame_free(&pFrame);
        }
        av_free_packet(&packet);
    }

    free(buffer);
    avcodec_close(codecCtx);
    av_frame_free(&pFrameRGB);
    av_free(formatCtx->pb->buffer);
    av_free(formatCtx->pb);
    avformat_close_input(&formatCtx);

}

void Video::ingest(char** dataBuf, int* dataBufLen, int* dataLen) {
    assert(_params != 0);
}

void Video::decode(char* item, int itemSize, char* buf) {

}

int Video::findVideoStream(AVCodecContext* &codecCtx, AVFormatContext* formatCtx) {
    for (int streamIdx = 0; streamIdx < (int) formatCtx->nb_streams; streamIdx++) {
        codecCtx = formatCtx->streams[streamIdx]->codec;
        if (codecCtx->coder_type == AVMEDIA_TYPE_VIDEO) {
            return streamIdx;
        }
    }
    return -1;
}

void Video::convertFrameFormat(AVCodecContext* codecCtx, AVPixelFormat pFormat,
                        AVFrame* &pFrame, AVFrame* &pFrameRGB) {

    struct SwsContext* imgConvertCtx = sws_getContext(
        codecCtx->width,
        codecCtx->height,
        codecCtx->pix_fmt,
        codecCtx->width,
        codecCtx->height,
        pFormat,
        SWS_BICUBIC,
        NULL,
        NULL,
        NULL
    );

    sws_scale(
        imgConvertCtx,
        pFrame->data,
        pFrame->linesize,
        0,
        codecCtx->height,
        pFrameRGB->data,
        pFrameRGB->linesize
    );
    sws_freeContext(imgConvertCtx);
}

void Video::writeFrameToBuf(cv::Mat frame, char* buf, int frameIdx, int channelSize) {
    // transform frame with _imgDecoder and then copy it into buf
    if (frameIdx == 0) {
        _imgDecoder->createRandomAugParams(frame.size());
    }
    // transform frame into imageBuf
    char* imageBuf = new char[_decodedSize];
    _imgDecoder->transformDecodedImage(frame, imageBuf, _decodedSize);
    cv::Mat decodedBuf = cv::Mat(1, _decodedSize, CV_8U, imageBuf);

    for(int c = 0; c < _params->_frameParams._channelCount; c++) {
        cv::Mat channel = decodedBuf.colRange(c * _imgSize, (c + 1) * _imgSize);
        std::copy(channel.data, channel.data + _imgSize,
                  buf + c * channelSize + frameIdx * _imgSize);
    }
    delete[] imageBuf;
}
