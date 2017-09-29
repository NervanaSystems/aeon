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
#include "etl_localization_ssd.hpp"
#include "box.hpp"

using namespace std;
using namespace nervana;

using std::vector;
using std::invalid_argument;
using nlohmann::json;

using bbox = boundingbox::box;
using nbox = normalized_box::box;

localization::ssd::config::config(nlohmann::json js)
{
    if (js.is_null())
    {
        throw std::runtime_error("missing ssd config in json config");
    }

    for (auto& info : config_list)
    {
        info->parse(js);
    }
    verify_config("localization_ssd", config_list, js);

    add_shape_type({2, 1}, "int32_t");
    add_shape_type({max_gt_boxes, 4}, "float");
    add_shape_type({1, 1}, "int32_t");
    add_shape_type({max_gt_boxes, 1}, "int32_t");

    // 'difficult' tag for gt_boxes
    add_shape_type({max_gt_boxes, 1}, "int32_t");

    class_name_map.clear();
    for (int i = 0; i < class_names.size(); i++)
    {
        class_name_map.insert({class_names[i], i});
    }

    validate();
}

void ::localization::ssd::config::validate()
{
    if (width <= 0)
    {
        throw std::invalid_argument("invalid width");
    }
    if (height <= 0)
    {
        throw std::invalid_argument("invalid height");
    }
    if (max_gt_boxes <= 0)
    {
        throw std::invalid_argument("invalid max_gt_boxes");
    }
    if (class_names.empty())
    {
        throw std::invalid_argument("class_names cannot be empty");
    }
}

localization::ssd::extractor::extractor(const localization::ssd::config& _cfg)
    : bbox_extractor{_cfg.class_name_map}
    , cfg(_cfg)
{
}

std::shared_ptr<localization::ssd::decoded> localization::ssd::extractor::extract(const void* data,
                                                                                  size_t size) const
{
    auto rc = std::make_shared<ssd::decoded>();
    auto bb = std::static_pointer_cast<boundingbox::decoded>(rc);
    bbox_extractor.extract(data, size, bb);
    if (!bb)
    {
        return nullptr;
    }

    rc->input_image_size  = cv::Size2i(bb->width(), bb->height());
    rc->output_image_size = cv::Size2i(cfg.width, cfg.height);

    return rc;
}

shared_ptr<localization::ssd::decoded>
    localization::ssd::transformer::transform(shared_ptr<augment::image::params>     settings,
                                              shared_ptr<localization::ssd::decoded> mp) const
{
    mp->gt_boxes = boundingbox::transformer::transform_box(mp->boxes(), settings);
    return mp;
}

localization::ssd::loader::loader(const localization::ssd::config& cfg)
{
    max_gt_boxes = cfg.max_gt_boxes;
}

void localization::ssd::loader::load(const vector<void*>&                        buf_list,
                                     std::shared_ptr<localization::ssd::decoded> mp) const
{
    int32_t* im_shape     = (int32_t*)buf_list[0];
    float*   gt_boxes     = (float*)buf_list[1];
    int32_t* num_gt_boxes = (int32_t*)buf_list[2];
    int32_t* gt_classes   = (int32_t*)buf_list[3];
    int32_t* gt_difficult = (int32_t*)buf_list[4];

    im_shape[0] = mp->output_image_size.width;
    im_shape[1] = mp->output_image_size.height;

    *num_gt_boxes = min(max_gt_boxes, mp->gt_boxes.size());
    for (int i = 0; i < *num_gt_boxes; i++)
    {
        const bbox& gt  = mp->gt_boxes[i];
        const nbox& ngt = gt.normalize(mp->output_image_size.width, mp->output_image_size.height);
        *gt_boxes++     = ngt.xmin();
        *gt_boxes++     = ngt.ymin();
        *gt_boxes++     = ngt.xmax();
        *gt_boxes++     = ngt.ymax();
        *gt_classes++   = gt.label();
        *gt_difficult++ = gt.difficult();
    }
    for (int i = *num_gt_boxes; i < max_gt_boxes; i++)
    {
        *gt_boxes++     = 0;
        *gt_boxes++     = 0;
        *gt_boxes++     = 0;
        *gt_boxes++     = 0;
        *gt_classes++   = 0;
        *gt_difficult++ = 0;
    }
}
