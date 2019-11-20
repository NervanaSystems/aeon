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

#include <algorithm>
#include "augment_image.hpp"
#include "image.hpp"

using namespace std;
using namespace nervana;

using nlohmann::json;

augment::image::param_factory::param_factory(nlohmann::json js)
{
    if (js.is_null() == false)
    {
        string type;
        auto   val = js.find("type");
        if (val == js.end())
        {
            throw invalid_argument("augmentation missing 'type'");
        }
        else
        {
            type = val->get<string>();
            js.erase(val);
        }

        if (type == "image")
        {
            for (auto& info : config_list)
            {
                info->parse(js);
            }

            if (flip_enable)
            {
                flip_distribution = bernoulli_distribution{0.5};
            }

            if (!center)
            {
                crop_offset = uniform_real_distribution<float>{0.0f, 1.0f};
            }

            if (padding > 0)
            {
                padding_crop_offset_distribution =
                    uniform_int_distribution<int>(0, padding * 2);
            }
        }
    }
}
shared_ptr<augment::image::params> augment::image::param_factory::make_params(
    size_t input_width, size_t input_height, size_t output_width, size_t output_height) const
{
    // Must use this method for creating a shared_ptr rather than make_shared
    // since the params default ctor is private and factory is friend
    // make_shared is not friend :(
    auto settings = shared_ptr<augment::image::params>(new augment::image::params());

    auto& random = get_thread_local_random_engine();

#ifdef WITH_OPENCV
    settings->output_size = cv::Size2i(output_width, output_height);
#endif

    settings->angle                  = angle(random);
    settings->flip                   = flip_distribution(random);
    settings->hue                    = hue(random);
    settings->contrast               = contrast(random);
    settings->brightness             = brightness(random);
    settings->saturation             = saturation(random);
    settings->padding                = padding;
    settings->resize_short_size      = resize_short_size;
    settings->interpolation_method   = interpolation_method;

#ifdef WITH_OPENCV
    cv::Size2f input_size = cv::Size(input_width, input_height);
#endif

    if (!crop_enable)
    {
        int c_off_x                   = padding_crop_offset_distribution(random);
        int c_off_y                   = padding_crop_offset_distribution(random);
#ifdef WITH_OPENCV
        settings->padding_crop_offset = cv::Size2i(c_off_x, c_off_y);
        settings->cropbox             = cv::Rect(cv::Point2f(0, 0), input_size);
#endif

        float image_scale;
        if (fixed_scaling_factor > 0)
        {
            image_scale = fixed_scaling_factor;
        }
        else
        {
#ifdef WITH_OPENCV
            image_scale = nervana::image::calculate_scale(input_size, output_width, output_height);
#endif
        }
#ifdef WITH_OPENCV
        input_size = input_size * image_scale;
#endif

#ifdef WITH_OPENCV
        settings->output_size.width  = unbiased_round(input_size.width);
        settings->output_size.height = unbiased_round(input_size.height);
#endif
    }
    else
    {
        if (do_area_scale)
        {
#ifdef WITH_OPENCV
            float      _horizontal_distortion = horizontal_distortion(random);
            _horizontal_distortion = sqrt(_horizontal_distortion);
            cv::Size2f out_shape(_horizontal_distortion, 1 / _horizontal_distortion);

            float bound = min((float)input_width / input_height /(out_shape.width * out_shape.width),
                             (float)input_height / input_width /( out_shape.height * out_shape.height));

            float scale_max = std::min(scale.max(), bound);
            float scale_min = std::min(scale.min(), bound);

            std::uniform_real_distribution<float> scale2{scale_min, scale_max};

            float target_area = sqrt(input_height * input_width * scale2(random));
            out_shape.width  *= target_area;
            out_shape.height *= target_area;

            float c_off_x = crop_offset(random);
            float c_off_y = crop_offset(random);

            cv::Size2f cropbox_size = out_shape;
            cv::Point2i cropbox_origin = nervana::image::cropbox_shift(input_size, cropbox_size, c_off_x, c_off_y);
            settings->cropbox = cv::Rect(cropbox_origin, cropbox_size);
#endif
        }
        else
        {
            if (padding > 0)
            {
                throw invalid_argument(
                    "crop_enable should not be true: when padding is defined, crop is executed by "
                    "default with cropbox size equal to intput image size");
            }
            float      image_scale            = scale(random);
            float      _horizontal_distortion = horizontal_distortion(random);
#ifdef WITH_OPENCV
            cv::Size2f out_shape(output_width * _horizontal_distortion, output_height);

            // TODO(sfraczek): add test for this resize short
            if (resize_short_size > 0)
            {
                input_size = nervana::image::get_resized_short_size(input_width,
                                                                    input_height,
                                                                    resize_short_size);
            }

            cv::Size2f cropbox_size = nervana::image::cropbox_max_proportional(input_size, out_shape);
            if (do_area_scale)
            {
                cropbox_size =
                    nervana::image::cropbox_area_scale(input_size, cropbox_size, image_scale);
            }
            else
            {
                cropbox_size = nervana::image::cropbox_linear_scale(cropbox_size, image_scale);
            }

            float c_off_x = crop_offset(random);
            float c_off_y = crop_offset(random);

            cv::Point2i cropbox_origin =
                nervana::image::cropbox_shift(input_size, cropbox_size, c_off_x, c_off_y);
            settings->cropbox = cv::Rect(cropbox_origin, cropbox_size);
#endif
        }
    }

    if (lighting.stddev() != 0)
    {
        for (int i = 0; i < 3; i++)
        {
            settings->lighting.push_back(lighting(random));
        }
        settings->color_noise_std = lighting.stddev();
    }

    return settings;
}
