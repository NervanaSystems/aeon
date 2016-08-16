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

        bool flip                = false;
        bool debug_deterministic = false;
    private:
        params() {}
    };

    class image_var::config : public interface::config {
    public:
        size_t                          min_size;
        size_t                          max_size;
        bool                            flip_enable = false;
        bool                            channel_major = true;
        size_t                          channels = 3;
        int32_t                         seed = 0; // Default is to seed deterministically
        std::string                     type_string{"uint8_t"};

        std::bernoulli_distribution     flip_distribution{0};

        config(nlohmann::json js)
        {
            if(js.is_null()) {
                throw std::runtime_error("missing image_var config in json config");
            }

            for(auto& info : config_list) {
                info->parse(js);
            }
            verify_config("image_var", config_list, js);

            // Now fill in derived
            shape_t shape;
            if (channel_major) {
                shape = {channels, max_size, max_size};
            } else{
                shape = {max_size, max_size, channels};
            }
            add_shape_type(shape, type_string);

            if(flip_enable) {
                flip_distribution = std::bernoulli_distribution{0.5};
            }

            validate();
        }

        virtual int num_crops() const { return 1; }

    private:
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_SCALAR(min_size, mode::REQUIRED),
            ADD_SCALAR(max_size, mode::REQUIRED),
            ADD_SCALAR(type_string, mode::OPTIONAL),
            ADD_SCALAR(flip_enable, mode::OPTIONAL),
            ADD_SCALAR(channel_major, mode::OPTIONAL),
            ADD_SCALAR(channels, mode::OPTIONAL),
            ADD_SCALAR(seed, mode::OPTIONAL)
        };

        config() {}
        void validate() {
            if(max_size < min_size) {
                throw std::invalid_argument("max_size must be greater than or equal to min_size");
            }
        }
    };

    class image_var::param_factory : public interface::param_factory<image_var::decoded, image_var::params> {
    public:
        param_factory(image_var::config& cfg) :
            _cfg{cfg},
            generator{0}
        {
            // A positive provided seed means to run deterministic with that seed
            if (_cfg.seed >= 0) {
                generator.seed((uint32_t) _cfg.seed);
            } else {
                generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
            }
        }

        virtual ~param_factory() {}

        std::shared_ptr<image_var::params> make_params(std::shared_ptr<const decoded>);
    private:

        image_var::config&         _cfg;
        std::default_random_engine generator;
    };

// ===============================================================================================
// Decoded
// ===============================================================================================

    class image_var::decoded : public interface::decoded_media {
    public:
        decoded() {}
        decoded(cv::Mat img) : _image{img} {}
        virtual ~decoded() override {}

        cv::Mat& get_image() { return _image; }
        cv::Size2i get_image_size() const {return _image.size(); }
        int get_image_channels() const { return _image.channels(); }

    protected:
        cv::Mat _image;
    };

    class image_var::extractor : public interface::extractor<image_var::decoded> {
    public:
        extractor(const image_var::config&);
        virtual ~extractor() {}
        virtual std::shared_ptr<image_var::decoded> extract(const char*, int) override;

        const int get_channel_count() {return _color_mode == CV_LOAD_IMAGE_COLOR ? 3 : 1;}
        void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);
    private:
        int _pixel_type;
        int _color_mode;
    };

    class image_var::transformer : public interface::transformer<image_var::decoded, image_var::params> {
    public:
        transformer(const image_var::config&);
        ~transformer() {}
        virtual std::shared_ptr<image_var::decoded> transform(
                                                std::shared_ptr<image_var::params>,
                                                std::shared_ptr<image_var::decoded>) override;

    private:
        int min_size;
        int max_size;
    };

    class image_var::loader : public interface::loader<image_var::decoded> {
    public:
        loader(const image_var::config&);
        virtual ~loader() {}
        virtual void load(const std::vector<void*>&, std::shared_ptr<image_var::decoded>) override;

//        void fill_info(count_size_type* cst) override
//        {
//            cst->count   = _load_count;
//            cst->size    = _load_size;
//            cst->type[0] = 'u';
//        }

    private:
        size_t _load_size;
        void split(cv::Mat&, char*);
        bool _channel_major;
        const shape_type& stype;
    };
}
