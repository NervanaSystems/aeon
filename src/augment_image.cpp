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

#include "augment_image.hpp"
#include "image.hpp"

using namespace std;
using namespace nervana;

augment::image::param_factory::param_factory(nlohmann::json js)
{
    if (js.is_null() == false)
    {
        string type;
        auto   val = js.find("type");
        if (val == js.end())
        {
            throw std::invalid_argument("augmentation missing 'type'");
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
            // verify_config("augment image", config_list, js);

            // Now fill in derived
            if (flip_enable)
            {
                flip_distribution = bernoulli_distribution{0.5};
            }

            if (!center)
            {
                crop_offset = uniform_real_distribution<float>{0.0f, 1.0f};
            }
        }
    }
}

shared_ptr<augment::image::params> augment::image::param_factory::make_params(size_t input_width,
                                                                              size_t input_height,
                                                                              size_t output_width,
                                                                              size_t output_height)
{
    // Must use this method for creating a shared_ptr rather than make_shared
    // since the params default ctor is private and factory is friend
    // make_shared is not friend :(
    auto settings = shared_ptr<augment::image::params>(new augment::image::params());

    settings->output_size = cv::Size2i(output_width, output_height);

    settings->angle      = angle(m_dre);
    settings->flip       = flip_distribution(m_dre);
    settings->hue        = hue(m_dre);
    settings->contrast   = contrast(m_dre);
    settings->brightness = brightness(m_dre);
    settings->saturation = saturation(m_dre);

    cv::Size2f input_size = cv::Size(input_width, input_height);
    if (!crop_enable)
    {
        settings->cropbox = cv::Rect(cv::Point2f(0, 0), input_size);
        float image_scale;
        if (fixed_scaling_factor > 0)
        {
            image_scale = fixed_scaling_factor;
        }
        else
        {
            image_scale = nervana::image::calculate_scale(input_size, output_width, output_height);
        }
        input_size                   = input_size * image_scale;
        settings->output_size.width  = nervana::unbiased_round(input_size.width);
        settings->output_size.height = nervana::unbiased_round(input_size.height);
    }
    else
    {
        float      image_scale            = scale(m_dre);
        float      _horizontal_distortion = horizontal_distortion(m_dre);
        cv::Size2f out_shape(output_width * _horizontal_distortion, output_height);

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

        float c_off_x = crop_offset(m_dre);
        float c_off_y = crop_offset(m_dre);

        cv::Point2f cropbox_origin =
            nervana::image::cropbox_shift(input_size, cropbox_size, c_off_x, c_off_y);
        settings->cropbox = cv::Rect(cropbox_origin, cropbox_size);
    }

    if (lighting.stddev() != 0)
    {
        for (int i = 0; i < 3; i++)
        {
            settings->lighting.push_back(lighting(m_dre));
        }
        settings->color_noise_std = lighting.stddev();
    }

    return settings;
}
