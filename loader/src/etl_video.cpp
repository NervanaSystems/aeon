#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_video.hpp"

using namespace std;
using namespace nervana;

video::params::params(std::shared_ptr<image::params> imageParams)
    : _imageParams(imageParams) {
}

void video::params::dump(ostream & ostr) {
    ostr << "FrameParams: ";
    _imageParams->dump(ostr);

    ostr << "Frames Per Clip: " << _framesPerClip << " ";
}

video::extractor::extractor(std::shared_ptr<const video::config>)
    : _pFormat(AV_PIX_FMT_BGR24) {
    _pFrameRGB = av_frame_alloc();
    _pFrame = av_frame_alloc();

    // TODO: pull necessary config from here
    // it looks like the only config we may want here (from image) is
    // color vs black and white.  This will involve changing _pFormat
    // https://ffmpeg.org/doxygen/2.1/pixfmt_8h.html#a9a8e335cf3be472042bc9f0cf80cd4c5
}

video::extractor::~extractor() {
    av_frame_free(&_pFrameRGB);
    av_frame_free(&_pFrame);
}

std::shared_ptr<video::decoded> video::extractor::extract(const char* item, int itemSize) {
    // copy-pasted from video.cpp
    _out = make_shared<video::decoded>();

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
    free(buffer);
    avcodec_close(codecCtx);
    av_free(formatCtx->pb->buffer);
    av_free(formatCtx->pb);
    avformat_close_input(&formatCtx);

    return _out;
}

void video::extractor::decode_video_frame(AVCodecContext* codecCtx, AVPacket& packet) {
    int frameFinished;

    avcodec_decode_video2(codecCtx, _pFrame, &frameFinished, &packet);
    if (frameFinished) {
        convertFrameFormat(codecCtx, _pFormat, _pFrame);

        cv::Mat frame(_pFrame->height, _pFrame->width, CV_8UC3, _pFrameRGB->data[0]);

        _out->add(frame);
    }
}

int video::extractor::findVideoStream(AVCodecContext* &codecCtx, AVFormatContext* formatCtx) {
    for (int streamIdx = 0; streamIdx < (int) formatCtx->nb_streams; streamIdx++) {
        codecCtx = formatCtx->streams[streamIdx]->codec;
        if (codecCtx->coder_type == AVMEDIA_TYPE_VIDEO) {
            return streamIdx;
        }
    }
    return -1;
}

void video::extractor::convertFrameFormat(AVCodecContext* codecCtx, AVPixelFormat pFormat,
                                          AVFrame* &pFrame) {
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

video::transformer::transformer(std::shared_ptr<const video::config> config)
    : _imageTransformer(config) {
}

std::shared_ptr<video::decoded> video::transformer::transform(
    std::shared_ptr<video::params> params,
    std::shared_ptr<video::decoded> decoded) {
    // simple wrapper around image::transformer for now
    return std::static_pointer_cast<video::decoded>(_imageTransformer.transform(
        params->_imageParams, decoded
    ));
}
