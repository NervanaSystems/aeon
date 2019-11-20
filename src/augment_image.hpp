/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <iostream>
#include <memory>
#include <limits>
#include <string>

#ifdef WITH_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#include "json.hpp"
#include "interface.hpp"

namespace nervana
{
    namespace augment
    {
        namespace image
        {
            class params;
            class param_factory;
        }
    }
}

class nervana::augment::image::params
{
    friend class nervana::augment::image::param_factory;

public:
    friend std::ostream& operator<<(std::ostream& out, const params& obj)
    {
        out << "expand_ratio         " << obj.expand_ratio << "\n";
#ifdef WITH_OPENCV
        out << "expand_offset        " << obj.expand_offset << "\n";
        out << "expand_size          " << obj.expand_size << "\n";
        out << "cropbox                " << obj.cropbox << "\n";
        out << "padding_crop_offset    " << obj.padding_crop_offset << "\n";
        out << "output_size            " << obj.output_size << "\n";
#endif
        out << "resize_short_size      " << obj.resize_short_size << "\n";
        out << "angle                  " << obj.angle << "\n";
        out << "flip                   " << obj.flip << "\n";
        out << "padding                " << obj.padding << "\n";
        out << "lighting               " << join(obj.lighting, ", ") << "\n";
        out << "color_noise_std        " << obj.color_noise_std << "\n";
        out << "contrast               " << obj.contrast << "\n";
        out << "brightness             " << obj.brightness << "\n";
        out << "saturation             " << obj.saturation << "\n";
        out << "hue                    " << obj.hue << "\n";
        out << "debug_deterministic    " << obj.debug_deterministic << "\n";
        return out;
    }

    float              expand_ratio = 1.0;
    #ifdef WITH_OPENCV
    cv::Size2i         expand_offset;
    cv::Size2i         expand_size;
    cv::Rect           cropbox;
    cv::Size2i         output_size;
    cv::Size2i         padding_crop_offset;
    #endif
    int                resize_short_size = 0;
    std::string        interpolation_method = "LINEAR";
    int                angle = 0;
    bool               flip  = false;
    int                padding;
    std::vector<float> lighting; // pixelwise random values
    float              color_noise_std        = 0;
    float              contrast               = 1.0;
    float              brightness             = 1.0;
    float              saturation             = 1.0;
    int                hue                    = 0;
    bool               debug_deterministic    = false;

private:
    params() {}
};

class nervana::augment::image::param_factory : public json_configurable
{
    public:
        param_factory(nlohmann::json config);
        std::shared_ptr<params> make_params(size_t input_width,
                size_t input_height,
                size_t output_width,
                size_t output_height) const;

        bool                do_area_scale                 = false;
        bool                crop_enable                   = true;
        bool                fixed_aspect_ratio            = false;
        std::vector<double> mean                          = {};
        std::vector<double> stddev                        = {};
        int                 resize_short_size             = 0;
        std::string         interpolation_method          = "LINEAR";
        float               expand_probability            = 0.;
        float               fixed_scaling_factor          = -1;

        /** Scale the crop box (width, height) */
        mutable std::uniform_real_distribution<float> scale{1.0f, 1.0f};

        /** Rotate the image (rho, phi) */
        mutable std::uniform_int_distribution<int> angle{0, 0};

        /** Adjust lighting */
        mutable std::normal_distribution<float> lighting{0.0f, 0.0f};

        /** Adjust aspect ratio */
        mutable std::uniform_real_distribution<float> horizontal_distortion{1.0f, 1.0f};

        /** Adjust contrast */
        mutable std::uniform_real_distribution<float> contrast{1.0f, 1.0f};

        /** Adjust brightness */
        mutable std::uniform_real_distribution<float> brightness{1.0f, 1.0f};

        /** Adjust saturation */
        mutable std::uniform_real_distribution<float> saturation{1.0f, 1.0f};

        /** Expand image */
        mutable std::uniform_real_distribution<float> expand_ratio{1.0f, 1.0f};
        mutable std::uniform_real_distribution<float> expand_distribution{0.0f, 1.0f};

        /** Rotate hue in degrees. Valid values are [-180; 180] */
        mutable std::uniform_int_distribution<int> hue{0, 0};

        /** Offset from center for the crop */
        mutable std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};

        /** Flip the image left to right */
        mutable std::bernoulli_distribution flip_distribution{0};

        /** Image padding pixel number with random crop to original image size */
        int padding{0};

    private:
        bool      flip_enable = false;
        bool      center      = true;

        /** Offset for padding cropbox */
        mutable std::uniform_int_distribution<int> padding_crop_offset_distribution{0, 0};

        std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
            ADD_DISTRIBUTION(scale,
                    mode::OPTIONAL,
                    [](const std::uniform_real_distribution<float>& v) {
                    return v.a() >= 0 && v.a() <= 1 && v.b() >= 0 && v.b() <= 1 &&
                    v.a() <= v.b();
                    }),
            ADD_DISTRIBUTION(angle, mode::OPTIONAL, [](decltype(angle) v) { return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(lighting, mode::OPTIONAL),
            ADD_DISTRIBUTION(horizontal_distortion,
                    mode::OPTIONAL,
                    [](decltype(horizontal_distortion) v) { return v.a() <= v.b(); }),
            ADD_SCALAR(flip_enable, mode::OPTIONAL),
            ADD_SCALAR(center, mode::OPTIONAL),
            ADD_SCALAR(resize_short_size, mode::OPTIONAL),
            ADD_SCALAR(interpolation_method, mode::OPTIONAL),
            ADD_SCALAR(do_area_scale, mode::OPTIONAL),
            ADD_SCALAR(crop_enable, mode::OPTIONAL),
            ADD_SCALAR(expand_probability, mode::OPTIONAL),
            ADD_SCALAR(fixed_aspect_ratio, mode::OPTIONAL),
            ADD_SCALAR(mean, mode::OPTIONAL),
            ADD_SCALAR(stddev, mode::OPTIONAL),
            ADD_SCALAR(fixed_scaling_factor, mode::OPTIONAL),
            ADD_SCALAR(padding, mode::OPTIONAL),
            ADD_DISTRIBUTION(
                    contrast, mode::OPTIONAL, [](decltype(contrast) v) { return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(
                    brightness, mode::OPTIONAL, [](decltype(brightness) v) { return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(
                    saturation, mode::OPTIONAL, [](decltype(saturation) v) { return v.a() <= v.b(); }),
            ADD_DISTRIBUTION(expand_ratio,
                    mode::OPTIONAL,
                    [](decltype(expand_ratio) v) { return v.a() >= 1 && v.a() <= v.b(); }),
            ADD_DISTRIBUTION(hue, mode::OPTIONAL, [](decltype(hue) v) { return v.a() <= v.b(); })
        };
};
