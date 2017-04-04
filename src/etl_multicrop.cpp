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

#include <memory>
#include <vector>

#include "etl_multicrop.hpp"

using namespace std;
using namespace nervana;

using cv::Point2f;

multicrop::config::config(nlohmann::json js)
    : crop_config(js["crop_config"])
{
    INFO << "\n" << js.dump(4);
    if (js.is_null())
    {
        throw std::runtime_error("missing multicrop config in json config");
    }

    INFO;
    for (auto& info : config_list)
    {
        info->parse(js);
    }

    INFO;
    verify_config("multicrop", config_list, js);
    INFO;

    if (crop_config.flip_enable)
    {
        orientations.push_back(true);
    }

    INFO;
    // shape is going to be different from crop_config because of multiple images
    shape_t multicrop_shape = crop_config.get_shape_type().get_shape();
    auto    axes_names      = crop_config.get_shape_type().get_names();
    axes_names.insert(axes_names.begin(), "views");

    uint32_t num_views = crop_count * crop_scales.size() * (crop_config.flip_enable ? 2 : 1);
    multicrop_shape.insert(multicrop_shape.begin(), num_views);
    add_shape_type(multicrop_shape, axes_names, crop_config.output_type);

    INFO;
    validate();
    INFO;
}

void multicrop::config::validate()
{
    if (crop_count != 5 && crop_count != 1)
    {
        throw std::invalid_argument("num_crops must be 1 or 5");
    }

    for (const float& s : crop_scales)
    {
        if (!((0.0 < s) && (s < 1.0)))
        {
            throw std::invalid_argument("crop_scales values must be between 0.0 and 1.0");
        }
    }
}

multicrop::transformer::transformer(const multicrop::config& cfg)
    : _crop_transformer{cfg.crop_config}
    , _crop_scales(cfg.crop_scales)
    , _orientations(cfg.orientations)
{
    if (cfg.crop_count == 5)
    {
        _offsets.emplace_back(0.0, 0.0); // NW
        _offsets.emplace_back(0.0, 1.0); // SW
        _offsets.emplace_back(1.0, 0.0); // NE
        _offsets.emplace_back(1.0, 1.0); // SE
    }
}

shared_ptr<image::decoded>
    multicrop::transformer::transform(shared_ptr<augment::image::params> crop_settings,
                                      shared_ptr<image::decoded>         input)
{
    cv::Size2i in_size      = input->get_image_size();
    auto       cropbox_size = image::cropbox_max_proportional(in_size, crop_settings->output_size);

    vector<cv::Rect> cropboxes;
    // Get the positional crop boxes
    for (const float& s : _crop_scales)
    {
        cv::Size2i boxdim = cropbox_size * s;
        cv::Size2i border = in_size - boxdim;
        for (const Point2f& offset : _offsets)
        {
            cv::Point2i corner(border.width * offset.x, border.height * offset.y);
            cropboxes.push_back(cv::Rect(corner, boxdim));
        }
    }

    auto out_imgs = make_shared<image::decoded>();

    for (auto cropbox : cropboxes)
    {
        crop_settings->cropbox = cropbox;
        for (auto orientation : _orientations)
        {
            crop_settings->flip = orientation;
            bool add_ok         = out_imgs->add(
                _crop_transformer.transform_single_image(crop_settings, input->get_image(0)));
            if (!add_ok)
            {
                return nullptr;
            }
        }
    }
    return out_imgs;
}
