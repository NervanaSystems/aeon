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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_video.hpp"

using namespace std;
using namespace nervana;


video::extractor::extractor(const video::config&)
 : _pFormat(AV_PIX_FMT_BGR24)
{
    // AV_PIX_FMT_BGR24 in ffmpeg corresponds to CV_8UC3 in opencv
    _pFrameRGB = av_frame_alloc();
    _pFrame = av_frame_alloc();

    // TODO: pull necessary config from here
    // it looks like the only config we may want here (from image) is
    // color vs black and white.  This will involve changing _pFormat
    // https://ffmpeg.org/doxygen/2.1/pixfmt_8h.html#a9a8e335cf3be472042bc9f0cf80cd4c5
}

video::extractor::~extractor()
{
    av_frame_free(&_pFrameRGB);
    av_frame_free(&_pFrame);
}

std::shared_ptr<image::decoded> video::extractor::extract(const char* item, int itemSize)
{
    // copy-pasted from video.cpp
    _out = make_shared<image::decoded>();

    // copy item for some unknown reason
    AVFormatContext* formatCtx = avformat_alloc_context();
    uchar* itemCopy = (unsigned char *) malloc(itemSize);
    memcpy(itemCopy, item, itemSize);
    formatCtx->pb = avio_alloc_context(itemCopy, itemSize, 0, itemCopy,
                                       NULL, NULL, NULL);

    // set up av boilerplate
    avformat_open_input(&formatCtx , "", NULL, NULL);
    avformat_find_stream_info(formatCtx, NULL);

    AVCodecContext* codecCtx = NULL;
    int videoStream = findVideoStream(codecCtx, formatCtx);

    AVCodec* pCodec = avcodec_find_decoder(codecCtx->codec_id);
    avcodec_open2(codecCtx, pCodec, NULL);

    int numBytes = avpicture_get_size(_pFormat, codecCtx->width, codecCtx->height);
    uint8_t* buffer = (uint8_t *) av_malloc(numBytes * sizeof(uint8_t));
    avpicture_fill((AVPicture*) _pFrameRGB, buffer, _pFormat,
                   codecCtx->width, codecCtx->height);

    // parse data stream
    AVPacket packet;
    while (av_read_frame(formatCtx, &packet) >= 0) {
        if (packet.stream_index == videoStream) {
            decode_video_frame(codecCtx, packet);
        }
        av_free_packet(&packet);
    }

    // some codecs, such as MPEG, transmit the I and P frame with a
    // latency of one frame. You must do the following to have a
    // chance to get the last frame of the video
    packet.data = NULL;
    packet.size = 0;
    decode_video_frame(codecCtx, packet);

    // cleanup
    av_free(buffer);
    avcodec_close(codecCtx);
    av_free(formatCtx->pb->buffer);
    av_free(formatCtx->pb);
    avformat_close_input(&formatCtx);

    return _out;
}

void video::extractor::decode_video_frame(AVCodecContext* codecCtx, AVPacket& packet)
{
    int frameFinished;

    avcodec_decode_video2(codecCtx, _pFrame, &frameFinished, &packet);
    if (frameFinished) {
        convertFrameFormat(codecCtx, _pFormat, _pFrame);

        cv::Mat frame(_pFrame->height, _pFrame->width, CV_8UC3, _pFrameRGB->data[0]);

        _out->add(frame);
    }
}

int video::extractor::findVideoStream(AVCodecContext* &codecCtx, AVFormatContext* formatCtx)
{
    for (int streamIdx = 0; streamIdx < (int) formatCtx->nb_streams; streamIdx++) {
        codecCtx = formatCtx->streams[streamIdx]->codec;
        if (codecCtx->coder_type == AVMEDIA_TYPE_VIDEO) {
            return streamIdx;
        }
    }
    return -1;
}

void video::extractor::convertFrameFormat(AVCodecContext* codecCtx, AVPixelFormat pFormat,
                                          AVFrame* &pFrame)
{
    // supposedly, some codecs allow the raw parameters like the frame size
    // to be changed at any frame, so we cant reuse this imgConvertCtx

    struct SwsContext* imgConvertCtx = sws_getContext(
        codecCtx->width, codecCtx->height, codecCtx->pix_fmt,
        codecCtx->width, codecCtx->height, pFormat,
        SWS_BICUBIC, NULL, NULL, NULL
    );

    sws_scale(
        imgConvertCtx, pFrame->data, pFrame->linesize, 0,
        codecCtx->height, _pFrameRGB->data, _pFrameRGB->linesize
    );

    sws_freeContext(imgConvertCtx);
}

video::transformer::transformer(const video::config& config)
 : frame_transformer(config.frame),
   max_frame_count(config.max_frame_count)
{}

std::shared_ptr<image::decoded> video::transformer::transform(
    std::shared_ptr<image::params> img_xform,
    std::shared_ptr<image::decoded> img)
{
    auto out_img = frame_transformer.transform(img_xform, img);

    // Now pad out if necessary
    cv::Mat pad_frame = cv::Mat::zeros(img_xform->output_size, img->get_image(0).type());
    while (out_img->get_image_count() < max_frame_count) {
        if (out_img->add(pad_frame) == false) {
            out_img = nullptr;
        }
    }

    return out_img;
}

void video::loader::load(const vector<void*>& buflist, shared_ptr<image::decoded> input)
{
    char* outbuf = (char*)buflist[0];
    // loads in channel x depth(frame) x height x width

    int num_channels = input->get_image_channels();
    int channel_size = input->get_image_count() * input->get_image_size().area();
    cv::Size2i image_size = input->get_image_size();

    for (int i=0; i < input->get_image_count(); i++) {
        auto img = input->get_image(i);

        auto image_offset = image_size.area() * i;

        if (num_channels == 1) {
            memcpy(outbuf + image_offset, img.data, image_size.area());
        } else {
            // create views into outbuf for the 3 channels to be copied into
            cv::Mat b(image_size, CV_8U, outbuf + 0 * channel_size + image_offset);
            cv::Mat g(image_size, CV_8U, outbuf + 1 * channel_size + image_offset);
            cv::Mat r(image_size, CV_8U, outbuf + 2 * channel_size + image_offset);

            cv::Mat channels[3] = {b, g, r};
            // cv::split will split img into component channels and copy
            // them into the addresses at b, g, r
            cv::split(img, channels);
        }
    }
}
