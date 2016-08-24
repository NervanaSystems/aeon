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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "interface.hpp"


namespace nervana {
    namespace image_var {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;
    }

    class image_var::params : public nervana::interface::params {
        friend class image_var::param_factory;
    public:
        void dump(std::ostream & = std::cout);

        cv::Rect            cropbox;
        cv::Size2i          output_size;
        bool                flip                = false;
        std::vector<float>  lighting;  // pixelwise random values
        float               color_noise_std = 0;
        std::vector<float>  photometric;  // contrast, brightness, saturation

        bool                debug_deterministic = false;
    private:
        params() {}
    };

    class image_var::config : public interface::config {
    public:
        size_t                                min_size;
        size_t                                max_size;
        bool                                  channel_major = true;
        size_t                                channels = 3;
        int32_t                               seed = 0; // Default is to seed deterministically
        std::string                           type_string{"uint8_t"};
        bool                                  do_area_scale = false;

        /** Scale the image (width, height) */
        std::uniform_real_distribution<float> scale{1.0f, 1.0f};

        /** Adjust lighting */
        std::normal_distribution<float>       lighting{0.0f, 0.0f};

        /** Adjust aspect ratio */
        std::uniform_real_distribution<float> horizontal_distortion{1.0f, 1.0f};

        /** Not sure what this guy does */
        std::uniform_real_distribution<float> photometric{0.0f, 0.0f};

        /** Not sure what this guy does */
        std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};

        /** Flip the image left to right */
        std::bernoulli_distribution           flip_distribution{0};

        config(nlohmann::json js);

        virtual int num_crops() const { return 1; }

    private:
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(min_size, mode::REQUIRED),
            ADD_SCALAR(max_size, mode::REQUIRED),
            ADD_SCALAR(seed, mode::OPTIONAL),
            ADD_DISTRIBUTION(scale, mode::OPTIONAL, [](const std::uniform_real_distribution<float>& v){
                return v.a() >= 0 && v.a() <= 1 && v.b() >= 0 && v.b() <= 1 && v.a() <= v.b();
            }),
            ADD_DISTRIBUTION(lighting, mode::OPTIONAL),
            ADD_DISTRIBUTION(horizontal_distortion, mode::OPTIONAL, [](decltype(horizontal_distortion) v){ return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(photometric, mode::OPTIONAL, [](decltype(photometric) v){ return v.a() <= v.b(); }),
            ADD_SCALAR(flip_enable, mode::OPTIONAL),
            ADD_SCALAR(center, mode::OPTIONAL),
            ADD_SCALAR(type_string, mode::OPTIONAL, [](const std::string& v){ return output_type::is_valid_type(v); }),
            ADD_SCALAR(do_area_scale, mode::OPTIONAL),
            ADD_SCALAR(channel_major, mode::OPTIONAL),
            ADD_SCALAR(channels, mode::OPTIONAL, [](uint32_t v){ return v==1 || v==3; })
        };

        config() {}
        void validate();

        bool                                  flip_enable = false;
        bool                                  center = true;
    };

    class image_var::param_factory : public interface::param_factory<image_var::decoded, image_var::params> {
    public:
        param_factory(image_var::config& cfg) :
            config{cfg},
            generator{0}
        {
            // A positive provided seed means to run deterministic with that seed
            if (config.seed >= 0) {
                generator.seed((uint32_t) config.seed);
            } else {
                generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }

        virtual ~param_factory() {}

        std::shared_ptr<image_var::params> make_params(std::shared_ptr<const decoded>);
    private:

        image_var::config&         config;
        std::default_random_engine generator;
    };

// ===============================================================================================
// Decoded
// ===============================================================================================

    class image_var::decoded : public interface::decoded_media {
    public:
        decoded() {}
        decoded(cv::Mat img) : image{img} {}
        virtual ~decoded() override {}

        cv::Mat& get_image() { return image; }
        cv::Size2i get_image_size() const {return image.size(); }
        int get_image_channels() const { return image.channels(); }

    protected:
        cv::Mat image;
    };

    class image_var::extractor : public interface::extractor<image_var::decoded> {
    public:
        extractor(const image_var::config&);
        virtual ~extractor() {}
        virtual std::shared_ptr<image_var::decoded> extract(const char*, int) override;

        const int get_channel_count() {return color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
        void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
    private:
        int pixel_type;
        int color_mode;
    };

    class image_var::transformer : public interface::transformer<image_var::decoded, image_var::params> {
    public:
        transformer(const image_var::config&);
        ~transformer() {}
        virtual std::shared_ptr<image_var::decoded> transform(
                                                std::shared_ptr<image_var::params>,
                                                std::shared_ptr<image_var::decoded>) override;

    private:
        size_t min_size;
        size_t max_size;
    };

    class image_var::loader : public interface::loader<image_var::decoded> {
    public:
        loader(const image_var::config&);
        virtual ~loader() {}
        virtual void load(const std::vector<void*>&, std::shared_ptr<image_var::decoded>) override;

    private:
        bool                channel_major;
        const shape_type&   stype;
    };
}
