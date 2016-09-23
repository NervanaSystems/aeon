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
#include "image.hpp"
#include "util.hpp"

namespace nervana {
    namespace image {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;
    }
    namespace video {
        class config;     // Forward decl for friending
    }

    namespace multicrop {
        class config;     // Forward decl for friending
    }

    class image::params : public nervana::interface::params {
        friend class image::param_factory;
    public:

        void dump(std::ostream & = std::cout);

        cv::Rect            cropbox;
        cv::Size2i          output_size;
        int                 angle = 0;
        bool                flip = false;
        std::vector<float>  lighting;  // pixelwise random values
        float               color_noise_std = 0;
        float               contrast = 1.0;
        float               brightness = 1.0;
        float               saturation = 1.0;
        int                 hue = 0;
        bool                debug_deterministic = false;
    private:
        params() {}
    };

    /**
     * \brief Configuration for image ETL
     *
     * An instantiation of this class controls the ETL of image data into the
     * target memory buffers from the source CPIO archives.
     */
    class image::config : public interface::config {
        friend class video::config;
        friend class multicrop::config;
    public:
        uint32_t                              height;
        uint32_t                              width;
        std::string                           output_type{"uint8_t"};

        bool                                  do_area_scale = false;
        bool                                  channel_major = true;
        bool                                  crop_enable = true;
        bool                                  fixed_aspect_ratio = false;
        uint32_t                              channels = 3;
        float                                 fixed_scaling_factor = -1;

        /** Scale the crop box (width, height) */
        std::uniform_real_distribution<float> scale{1.0f, 1.0f};

        /** Rotate the image (rho, phi) */
        std::uniform_int_distribution<int>    angle{0, 0};

        /** Adjust lighting */
        std::normal_distribution<float>       lighting{0.0f, 0.0f};

        /** Adjust aspect ratio */
        std::uniform_real_distribution<float> horizontal_distortion{1.0f, 1.0f};

        /** Adjust contrast */
        std::uniform_real_distribution<float> contrast{1.0f, 1.0f};

        /** Adjust brightness */
        std::uniform_real_distribution<float> brightness{1.0f, 1.0f};

        /** Adjust saturation */
        std::uniform_real_distribution<float> saturation{1.0f, 1.0f};

        /** Rotate hue in degrees. Valid values are [0-360] */
        std::uniform_int_distribution<int>    hue{0, 0};

        /** Offset from center for the crop */
        std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};

        /** Flip the image left to right */
        std::bernoulli_distribution           flip_distribution{0};

        config(nlohmann::json js);

    private:
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(height, mode::REQUIRED),
            ADD_SCALAR(width, mode::REQUIRED),
            ADD_DISTRIBUTION(scale, mode::OPTIONAL, [](const std::uniform_real_distribution<float>& v){
                return v.a() >= 0 && v.a() <= 1 && v.b() >= 0 && v.b() <= 1 && v.a() <= v.b();
            }),
            ADD_DISTRIBUTION(angle, mode::OPTIONAL, [](decltype(angle) v){ return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(lighting, mode::OPTIONAL),
            ADD_DISTRIBUTION(horizontal_distortion, mode::OPTIONAL, [](decltype(horizontal_distortion) v){ return v.a() <= v.b(); }),
            ADD_SCALAR(flip_enable, mode::OPTIONAL),
            ADD_SCALAR(center, mode::OPTIONAL),
            ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v){ return output_type::is_valid_type(v); }),
            ADD_SCALAR(do_area_scale, mode::OPTIONAL),
            ADD_SCALAR(channel_major, mode::OPTIONAL),
            ADD_SCALAR(channels, mode::OPTIONAL, [](uint32_t v){ return v==1 || v==3; }),
            ADD_SCALAR(crop_enable, mode::OPTIONAL),
            ADD_SCALAR(fixed_aspect_ratio, mode::OPTIONAL),
            ADD_SCALAR(fixed_scaling_factor, mode::OPTIONAL),
            ADD_DISTRIBUTION(contrast, mode::OPTIONAL, [](decltype(contrast) v){ return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(brightness, mode::OPTIONAL, [](decltype(brightness) v){ return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(saturation, mode::OPTIONAL, [](decltype(saturation) v){ return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(hue, mode::OPTIONAL, [](decltype(hue) v){ return v.a() <= v.b(); })
        };

        config() {}
        void validate();

        bool                                  flip_enable = false;
        bool                                  center = true;
    };

    class image::param_factory : public interface::param_factory<image::decoded, image::params> {
    public:
        param_factory(image::config& cfg) : _cfg{cfg}, _dre{0}
        {
            _dre.seed(get_global_random_seed());
        }
        virtual ~param_factory() {}

        std::shared_ptr<image::params> make_params(std::shared_ptr<const image::decoded> input);
    private:

        image::config& _cfg;
        std::default_random_engine _dre;
    };

// ===============================================================================================
// Decoded
// ===============================================================================================

    class image::decoded : public interface::decoded_media {
    public:
        decoded() {}
        decoded(cv::Mat img) { _images.push_back(img); }
        bool add(cv::Mat img) {
            _images.push_back(img);
            return all_images_are_same_size();
        }
        bool add(const std::vector<cv::Mat>& images) {
            for( auto mat : images ) {
                _images.push_back(mat);
            }
            return all_images_are_same_size();
        }
        virtual ~decoded() override {}

        cv::Mat& get_image(int index) { return _images[index]; }
        cv::Size2i get_image_size() const {return _images[0].size(); }
        int get_image_channels() const { return _images[0].channels(); }
        size_t get_image_count() const { return _images.size(); }
        size_t get_size() const {
            return get_image_size().area() * get_image_channels() * get_image_count();
        }

    protected:
        bool all_images_are_same_size() {
            for( int i=1; i<_images.size(); i++ ) {
                if(_images[0].size()!=_images[i].size()) return false;
            }
            return true;
        }
        std::vector<cv::Mat> _images;
    };



    class image::extractor : public interface::extractor<image::decoded> {
    public:
        extractor(const image::config&);
        ~extractor() {}
        virtual std::shared_ptr<image::decoded> extract(const char*, int) override;

        const int get_channel_count() {return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
    private:
        int _pixel_type;
        int _color_mode;
    };


    class image::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(const image::config&);
        ~transformer() {}
        virtual std::shared_ptr<image::decoded> transform(
                                                std::shared_ptr<image::params>,
                                                std::shared_ptr<image::decoded>) override;

        cv::Mat transform_single_image(std::shared_ptr<image::params>, cv::Mat&);
    private:
        image::photometric photo;
    };


    class image::loader : public interface::loader<image::decoded> {
    public:
        loader(const image::config& cfg);
        ~loader() {}
        virtual void load(const std::vector<void*>&, std::shared_ptr<image::decoded>) override;

    private:
        void split(cv::Mat&, char*);

        bool        channel_major;
        bool        fixed_aspect_ratio;
        shape_type  stype;
        uint32_t    channels;
    };
}