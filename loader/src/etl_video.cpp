#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "etl_video.hpp"

using namespace std;
using namespace nervana;

void video::params::dump(ostream & ostr) {
    ostr << "FrameParams: ";

    ostr << "Frames Per Clip: " << _framesPerClip << " ";
}


shared_ptr<video::params>
video::param_factory::make_params(shared_ptr<const image::decoded> input)
{
    auto imgstgs = shared_ptr<video::params>(new video::params());

    imgstgs->output_size = cv::Size2i(_cfg.width, _cfg.height);

    imgstgs->angle = _cfg.angle(_dre);
    imgstgs->flip  = _cfg.flip_distribution(_dre);

    cv::Size2f in_size = input->get_image_size();

    float scale = _cfg.scale(_dre);
    float aspect_ratio = _cfg.aspect_ratio(_dre);
    scale_cropbox(in_size, imgstgs->cropbox, aspect_ratio, scale);

    float c_off_x = _cfg.crop_offset(_dre);
    float c_off_y = _cfg.crop_offset(_dre);
    image::shift_cropbox(in_size, imgstgs->cropbox, c_off_x, c_off_y);

    if (_cfg.lighting.stddev() != 0) {
        for( int i=0; i<3; i++ ) {
            imgstgs->lighting.push_back(_cfg.lighting(_dre));
        }
        imgstgs->color_noise_std = _cfg.lighting.stddev();
    }
    if (_cfg.photometric.a()!=_cfg.photometric.b()) {
        for( int i=0; i<3; i++ ) {
            imgstgs->photometric.push_back(_cfg.photometric(_dre));
        }
    }
    return imgstgs;
}

void video::param_factory::scale_cropbox(
                            const cv::Size2f &in_size,
                            cv::Rect &crop_box,
                            float tgt_aspect_ratio,
                            float tgt_scale )
{

    float out_a_r = static_cast<float>(_cfg.width) / _cfg.height;
    float in_a_r  = in_size.width / in_size.height;

    float crop_a_r = out_a_r * tgt_aspect_ratio;

    if (_cfg.do_area_scale) {
        // Area scaling -- use pctge of original area subject to aspect ratio constraints
        float max_scale = in_a_r > crop_a_r ? crop_a_r /  in_a_r : in_a_r / crop_a_r;
        float tgt_area  = std::min(tgt_scale, max_scale) * in_size.area();

        crop_box.height = sqrt(tgt_area / crop_a_r);
        crop_box.width  = crop_box.height * crop_a_r;
    } else {
        // Linear scaling -- make the long crop box side  the scale pct of the short orig side
        float short_side = std::min(in_size.width, in_size.height);

        if (crop_a_r < 1) { // long side is height
            crop_box.height = tgt_scale * short_side;
            crop_box.width  = crop_box.height * crop_a_r;
        } else {
            crop_box.width  = tgt_scale * short_side;
            crop_box.height = crop_box.width / crop_a_r;
        }
    }
}


video::extractor::extractor(const video::config&)
    : _pFormat(AV_PIX_FMT_BGR24) {
    // AV_PIX_FMT_BGR24 in ffmpeg corresponds to CV_8UC3 in opencv
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
    av_free(buffer);
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

video::transformer::transformer(const video::config& config) {
}

std::shared_ptr<video::decoded> video::transformer::transform(
    std::shared_ptr<video::params> img_xform,
    std::shared_ptr<video::decoded> img)
{
    vector<cv::Mat> finalImageList;
    for(int i=0; i<img->get_image_count(); i++) {
        cv::Mat rotatedImage;
        image::rotate(img->get_image(i), rotatedImage, img_xform->angle);

        cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

        cv::Mat resizedImage;
        image::resize(croppedImage, resizedImage, img_xform->output_size);
        photo.cbsjitter(resizedImage, img_xform->photometric);
        photo.lighting(resizedImage, img_xform->lighting, img_xform->color_noise_std);

        cv::Mat *finalImage = &resizedImage;
        cv::Mat flippedImage;
        if (img_xform->flip) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        }
        finalImageList.push_back(*finalImage);
    }

    auto rc = make_shared<video::decoded>();
    if(rc->add(finalImageList) == false) {
        rc = nullptr;
    }
    return rc;
}

void video::loader::load(const vector<void*>& buflist, shared_ptr<video::decoded> input)
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
