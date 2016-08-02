#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "interface.hpp"
#include "etl_image.hpp"

namespace nervana {

    namespace multicrop {
        class config;
        class transformer;
    }


    class multicrop::config : public interface::config {
    public:

        // Required config variables
        std::vector<float>  multicrop_scales;

        // Optional config variables
        int                 crops_per_scale = 5;
        bool                include_flips = true;

        // Stuff from image::config
        uint32_t                              height;
        uint32_t                              width;
        int32_t                               seed = 0; // Default is to seed deterministically
        std::string                           type_string{"uint8_t"};
        bool                                  do_area_scale = false;
        bool                                  channel_major = true;
        uint32_t                              channels = 3;
        std::uniform_real_distribution<float> scale{1.0f, 1.0f};
        std::uniform_int_distribution<int>    angle{0, 0};
        std::normal_distribution<float>       lighting{0.0f, 0.0f};
        std::uniform_real_distribution<float> aspect_ratio{1.0f, 1.0f};
        std::uniform_real_distribution<float> photometric{0.0f, 0.0f};
        std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};
        std::bernoulli_distribution           flip_distribution{0};
        bool                                  flip;

        // Derived config variables
        std::vector<cv::Point2f> offsets;
        cv::Size2i               output_size;

        config(nlohmann::json js)
        {
            if(js.is_null()) {
                throw std::runtime_error("missing multicrop config in json config");
            }

            for(auto& info : config_list) {
                info->parse(js);
            }
            verify_config(config_list, js);

            // Fill in derived variables
            otype = nervana::output_type(type_string);
            offsets.push_back(cv::Point2f(0.5, 0.5)); // Center
            if(flip) {
                flip_distribution = std::bernoulli_distribution{0.5};
            }
            if (crops_per_scale == 5) {
                offsets.push_back(cv::Point2f(0.0, 0.0)); // NW
                offsets.push_back(cv::Point2f(0.0, 1.0)); // SW
                offsets.push_back(cv::Point2f(1.0, 0.0)); // NE
                offsets.push_back(cv::Point2f(1.0, 1.0)); // SE
            }
            output_size = cv::Size2i(height, width);

            // shape is going to be different because of multiple images
            uint32_t num_views = crops_per_scale * multicrop_scales.size() * (include_flips ? 2 : 1);
            shape.insert(shape.begin(), num_views);

            validate();
        }

    private:
        config() {}
        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            // from image class
            ADD_SCALAR(height, mode::REQUIRED),
            ADD_SCALAR(width, mode::REQUIRED),
            ADD_SCALAR(seed, mode::OPTIONAL),
            ADD_DISTRIBUTION(scale, mode::OPTIONAL),
            ADD_DISTRIBUTION(angle, mode::OPTIONAL),
            ADD_DISTRIBUTION(lighting, mode::OPTIONAL),
            ADD_DISTRIBUTION(aspect_ratio, mode::OPTIONAL),
            ADD_DISTRIBUTION(photometric, mode::OPTIONAL),
            ADD_DISTRIBUTION(crop_offset, mode::OPTIONAL),
            ADD_SCALAR(flip, mode::OPTIONAL),
            ADD_SCALAR(type_string, mode::OPTIONAL),
            ADD_SCALAR(do_area_scale, mode::OPTIONAL),
            ADD_SCALAR(channel_major, mode::OPTIONAL),
            ADD_SCALAR(channels, mode::OPTIONAL),

            // local params
            ADD_SCALAR(multicrop_scales, mode::REQUIRED),
            ADD_SCALAR(crops_per_scale, mode::OPTIONAL),
            ADD_SCALAR(include_flips, mode::OPTIONAL)

        };
        void validate()
        {
            if(crops_per_scale != 5 && crops_per_scale != 1) {
                throw std::invalid_argument("crops_per_scale must be 1 or 5");
            }

            for (const float &s: multicrop_scales) {
                if(!( (0.0 < s) && (s < 1.0))) {
                    throw std::invalid_argument("multicrop_scales values must be between 0.0 and 1.0");
                }
            }
            base_validate();
        }
    };


    class multicrop::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(const multicrop::config& cfg) : _cfg(cfg) {}
        ~transformer() {}
        virtual std::shared_ptr<image::decoded> transform(
                                                std::shared_ptr<image::params>,
                                                std::shared_ptr<image::decoded>) override;
    private:
        const multicrop::config& _cfg;

        void add_resized_crops(const cv::Mat&, std::shared_ptr<image::decoded>&, std::vector<cv::Rect>&);
    };
}