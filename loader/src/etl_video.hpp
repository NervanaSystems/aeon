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

    class video::config : public interface::config {
    public:
        uint32_t                              frame_count;
        uint32_t                              height;
        uint32_t                              width;
        int32_t                               seed = 0; // Default is to seed deterministically
        std::string                           type_string{"uint8_t"};
        bool                                  do_area_scale = false;
        uint32_t                              channels = 3;

        std::uniform_real_distribution<float> scale{1.0f, 1.0f};
        std::uniform_int_distribution<int>    angle{0, 0};
        std::normal_distribution<float>       lighting{0.0f, 0.0f};
        std::uniform_real_distribution<float> aspect_ratio{1.0f, 1.0f};
        std::uniform_real_distribution<float> photometric{0.0f, 0.0f};
        std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};
        std::bernoulli_distribution           flip_distribution{0};
        bool                                  flip_enable = false;

        config(nlohmann::json js)
        {
            if(js.is_null()) {
                throw std::runtime_error("missing multicrop config in json config");
            }

            for(auto& info : config_list) {
                info->parse(js);
            }
            verify_config("video", config_list, js);

            if(flip_enable) {
                flip_distribution = std::bernoulli_distribution{0.5};
            }
            // channel major only
            add_shape_type({channels, frame_count, height, width}, type_string);
        }

    private:
        config() {}
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(height, mode::REQUIRED),
            ADD_SCALAR(width, mode::REQUIRED),
            ADD_SCALAR(seed, mode::OPTIONAL),
            ADD_DISTRIBUTION(scale, mode::OPTIONAL),
            ADD_DISTRIBUTION(angle, mode::OPTIONAL),
            ADD_DISTRIBUTION(lighting, mode::OPTIONAL),
            ADD_DISTRIBUTION(aspect_ratio, mode::OPTIONAL),
            ADD_DISTRIBUTION(photometric, mode::OPTIONAL),
            ADD_DISTRIBUTION(crop_offset, mode::OPTIONAL),
            ADD_SCALAR(flip_enable, mode::OPTIONAL),
            ADD_SCALAR(type_string, mode::OPTIONAL),
            ADD_SCALAR(do_area_scale, mode::OPTIONAL),
            ADD_SCALAR(channels, mode::OPTIONAL),

            ADD_SCALAR(frame_count, mode::REQUIRED)
        };
    };

    class video::params : public interface::params {
        friend class video::param_factory;
    public:
        void dump(std::ostream & = std::cout);

        cv::Rect            cropbox;
        cv::Size2i          output_size;
        int                 angle = 0;
        bool                flip = false;
        std::vector<float>  lighting;  // pixelwise random values
        float               color_noise_std = 0;
        std::vector<float>  photometric;  // contrast, brightness, saturation
        int                 _framesPerClip;

    private:
        params() {}
    };

    class video::param_factory : public interface::param_factory<image::decoded, video::params> {
    public:
        param_factory(video::config& cfg) : _cfg{cfg}, _dre{0}
        {
            // A positive provided seed means to run deterministic with that seed
            if (_cfg.seed >= 0) {
                _dre.seed((uint32_t) _cfg.seed);
            } else {
                _dre.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }
        virtual ~param_factory() {}

        std::shared_ptr<video::params> make_params(std::shared_ptr<const image::decoded> input);
    private:
        void scale_cropbox(const cv::Size2f&, cv::Rect&, float, float);

        video::config& _cfg;
        std::default_random_engine _dre;
    };

    class video::decoded : public image::decoded {
    public:
        virtual ~decoded(){}
    };

    class video::extractor : public interface::extractor<video::decoded> {
    public:
        extractor(const video::config&);
        virtual ~extractor();

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

    private:
        extractor() = delete;
    };

    // simple wrapper around image::transformer for now
    class video::transformer : public interface::transformer<video::decoded, video::params> {
    public:
        transformer(const video::config&);
        virtual ~transformer() {}
        virtual std::shared_ptr<video::decoded> transform(
                                                std::shared_ptr<video::params>,
                                                std::shared_ptr<video::decoded>) override;
    protected:
        transformer() = delete;

        image::photometric  photo;
    };

    class video::loader : public interface::loader<video::decoded> {
    public:
        loader(const video::config&) {}
        virtual ~loader() {}
        virtual void load(const std::vector<void*>&, std::shared_ptr<video::decoded>) override;

    private:
        loader() = delete;
        void split(cv::Mat&, char*);
    };
}
