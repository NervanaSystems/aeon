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

#include "etl_image.hpp"

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

namespace nervana {
    namespace video {
        class config;
        class params;
        class decoded;

        // goes from config -> params
        class param_factory;

        class extractor;
        class transformer;
        class loader;
    }

    class video::config : public nervana::image::config {
    public:
        uint32_t num_frames;

        bool set_config(nlohmann::json js) override
        {
            image::config::set_config(js);
            parse_req(num_frames, "num_frames", js);
            shape.insert(shape.begin() + 1, num_frames); // This is for the depth after channels

            return validate();
        }
    };

    class video::params : public nervana::params {
    public:
        params(std::shared_ptr<image::params>);
        void dump(std::ostream & = std::cout);

        std::shared_ptr<image::params> _imageParams;
        int _framesPerClip;
    };

    class video::decoded : public image::decoded {
    public:
        virtual MediaType get_type() override { return MediaType::VIDEO; }
    };

    class video::extractor : public interface::extractor<video::decoded> {
    public:
        extractor(std::shared_ptr<const video::config>);
        ~extractor();

        virtual std::shared_ptr<video::decoded> extract(const char* item, int itemSize) override;

    protected:
        void decode_video_frame(AVCodecContext* codecCtx, AVPacket& packet);
        int findVideoStream(AVCodecContext* &codecCtx, AVFormatContext* formatCtx);
        void convertFrameFormat(AVCodecContext* codecCtx, AVPixelFormat pFormat,
                                AVFrame* &pFrame);

        std::shared_ptr<video::decoded> _out;
        AVPixelFormat _pFormat;
        AVFrame* _pFrameRGB;
        AVFrame* _pFrame;
    };

    // simple wrapper around image::transformer for now
    class video::transformer : public interface::transformer<video::decoded, video::params> {
    public:
        transformer(std::shared_ptr<const video::config>);
        ~transformer() {}
        virtual std::shared_ptr<video::decoded> transform(
                                                std::shared_ptr<video::params>,
                                                std::shared_ptr<video::decoded>) override;
    protected:
        image::transformer _imageTransformer;
    };

    class video::loader : public interface::loader<video::decoded> {
    public:
        loader(std::shared_ptr<const video::config>) {}
        ~loader() {}
        virtual void load(char*, std::shared_ptr<video::decoded>) override;

    private:
        void split(cv::Mat&, char*);
    };
}
