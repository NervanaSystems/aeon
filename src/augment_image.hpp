/*
 Copyright 2017 Nervana Systems Inc.
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

#include <iostream>
#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

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
        out << "cropbox             " << obj.cropbox << "\n";
        out << "output_size         " << obj.output_size << "\n";
        out << "angle               " << obj.angle << "\n";
        out << "flip                " << obj.flip << "\n";
        out << "lighting            " << join(obj.lighting, ", ") << "\n";
        out << "color_noise_std     " << obj.color_noise_std << "\n";
        out << "contrast            " << obj.contrast << "\n";
        out << "brightness          " << obj.brightness << "\n";
        out << "saturation          " << obj.saturation << "\n";
        out << "hue                 " << obj.hue << "\n";
        out << "debug_deterministic " << obj.debug_deterministic << "\n";
        return out;
    }

    cv::Rect           cropbox;
    cv::Size2i         output_size;
    int                angle = 0;
    bool               flip  = false;
    std::vector<float> lighting; // pixelwise random values
    float              color_noise_std     = 0;
    float              contrast            = 1.0;
    float              brightness          = 1.0;
    float              saturation          = 1.0;
    int                hue                 = 0;
    bool               debug_deterministic = false;

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
                                        size_t output_height);

    bool  do_area_scale        = false;
    bool  crop_enable          = true;
    bool  fixed_aspect_ratio   = false;
    float fixed_scaling_factor = -1;

    /** Scale the crop box (width, height) */
    std::uniform_real_distribution<float> scale{1.0f, 1.0f};

    /** Rotate the image (rho, phi) */
    std::uniform_int_distribution<int> angle{0, 0};

    /** Adjust lighting */
    std::normal_distribution<float> lighting{0.0f, 0.0f};

    /** Adjust aspect ratio */
    std::uniform_real_distribution<float> horizontal_distortion{1.0f, 1.0f};

    /** Adjust contrast */
    std::uniform_real_distribution<float> contrast{1.0f, 1.0f};

    /** Adjust brightness */
    std::uniform_real_distribution<float> brightness{1.0f, 1.0f};

    /** Adjust saturation */
    std::uniform_real_distribution<float> saturation{1.0f, 1.0f};

    /** Rotate hue in degrees. Valid values are [0-360] */
    std::uniform_int_distribution<int> hue{0, 0};

    /** Offset from center for the crop */
    std::uniform_real_distribution<float> crop_offset{0.5f, 0.5f};

    /** Flip the image left to right */
    std::bernoulli_distribution flip_distribution{0};

private:
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
        ADD_SCALAR(do_area_scale, mode::OPTIONAL),
        ADD_SCALAR(crop_enable, mode::OPTIONAL),
        ADD_SCALAR(fixed_aspect_ratio, mode::OPTIONAL),
        ADD_SCALAR(fixed_scaling_factor, mode::OPTIONAL),
        ADD_DISTRIBUTION(
            contrast, mode::OPTIONAL, [](decltype(contrast) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(
            brightness, mode::OPTIONAL, [](decltype(brightness) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(
            saturation, mode::OPTIONAL, [](decltype(saturation) v) { return v.a() <= v.b(); }),
        ADD_DISTRIBUTION(hue, mode::OPTIONAL, [](decltype(hue) v) { return v.a() <= v.b(); })};

    bool flip_enable = false;
    bool center      = true;

    std::default_random_engine m_dre;
};
