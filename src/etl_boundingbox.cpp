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

#include <sstream>
#include "etl_boundingbox.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;
using namespace nlohmann; // json stuff

using bbox = boundingbox::box;

boundingbox::config::config(nlohmann::json js)
{
    if (js.is_null())
    {
        throw std::runtime_error("missing bbox config in json config");
    }

    for (auto& info : config_list)
    {
        info->parse(js);
    }
    verify_config("bbox", config_list, js);

    // Derived values
    add_shape_type({max_bbox_count, 4 * sizeof(float)}, output_type);
    label_map.clear();
    for (int i = 0; i < class_names.size(); i++)
    {
        label_map.insert({class_names[i], i});
    }

    validate();
}

void boundingbox::config::validate()
{
}

boundingbox::decoded::decoded()
{
}

boundingbox::extractor::extractor(const std::unordered_map<std::string, int>& map)
    : label_map{map}
{
}

void boundingbox::extractor::extract(const void*                            data,
                                     size_t                                 size,
                                     std::shared_ptr<boundingbox::decoded>& rc) const
{
    string buffer((const char*)data, size);
    json   j = json::parse(buffer);
    if (j["object"].is_null())
    {
        throw invalid_argument("'object' missing from metadata");
    }
    if (j["size"].is_null())
    {
        throw invalid_argument("'size' missing from metadata");
    }
    auto object_list = j["object"];
    auto image_size  = j["size"];
    if (image_size["width"].is_null())
    {
        throw invalid_argument("'width' missing from metadata");
    }
    if (image_size["height"].is_null())
    {
        throw invalid_argument("'height' missing from metadata");
    }
    if (image_size["depth"].is_null())
    {
        throw invalid_argument("'depth' missing from metadata");
    }
    rc->m_height = image_size["height"];
    rc->m_width  = image_size["width"];
    rc->m_depth  = image_size["depth"];
    for (auto object : object_list)
    {
        auto bndbox = object["bndbox"];
        if (bndbox["xmax"].is_null())
        {
            throw invalid_argument("'xmax' missing from metadata");
        }
        if (bndbox["xmin"].is_null())
        {
            throw invalid_argument("'xmin' missing from metadata");
        }
        if (bndbox["ymax"].is_null())
        {
            throw invalid_argument("'ymax' missing from metadata");
        }
        if (bndbox["ymin"].is_null())
        {
            throw invalid_argument("'ymin' missing from metadata");
        }
        if (object["name"].is_null())
        {
            throw invalid_argument("'name' missing from metadata");
        }

        bool difficult = object["difficult"].is_null() ? false : object["difficult"].get<bool>();
        bool truncated = object["truncated"].is_null() ? false : object["truncated"].get<bool>();
        int  label     = this->get_label(object);

        bbox b(bndbox["xmin"],
               bndbox["ymin"],
               bndbox["xmax"],
               bndbox["ymax"],
               label,
               difficult,
               truncated);
        rc->m_boxes.push_back(b);
    }
}

shared_ptr<boundingbox::decoded> boundingbox::extractor::extract(const void* data,
                                                                 size_t      size) const
{
    shared_ptr<decoded> rc = make_shared<decoded>();
    extract(data, size, rc);
    return rc;
}

boundingbox::transformer::transformer(const boundingbox::config&)
{
}

bool boundingbox::transformer::meet_emit_constraint(const cv::Rect& cropbox,
                                                    const bbox&     input_bbox,
                                                    const emit_type emit_constraint_type,
                                                    const float     emit_min_overlap)
{
    bbox crop_bbox(
        cropbox.x, cropbox.y, cropbox.x + cropbox.width - 1, cropbox.y + cropbox.height - 1, false);
    if (emit_constraint_type == emit_type::center)
    {
        float x_center = input_bbox.xcenter();
        float y_center = input_bbox.ycenter();
        if (x_center >= crop_bbox.xmin() && x_center <= crop_bbox.xmax() &&
            y_center >= crop_bbox.ymin() && y_center <= crop_bbox.ymax())
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else if (emit_constraint_type == emit_type::min_overlap)
    {
        return input_bbox.coverage(crop_bbox) >= emit_min_overlap;
    }
    return true;
}

vector<bbox> boundingbox::transformer::transform_box(const std::vector<bbox>&           boxes,
                                                     shared_ptr<augment::image::params> pptr)
{
    cv::Rect crop    = pptr->cropbox;
    float    x_scale = (float)(pptr->output_size.width) / (float)(crop.width);
    float    y_scale = (float)(pptr->output_size.height) / (float)(crop.height);

    /*
    * expand
    * crop
    * flip
    * scale
    */

    vector<bbox> rc;
    for (bbox b : boxes)
    {
        if (pptr->expand_ratio > 1.)
        {
            b = b.expand(pptr->expand_offset, pptr->expand_size, pptr->expand_ratio);
        }

        if (pptr->emit_constraint_type != emit_type::undefined &&
            !meet_emit_constraint(crop, b, pptr->emit_constraint_type, pptr->emit_min_overlap))
            continue;

        if (b.xmax() < crop.x)
        { // outside left
        }
        else if (b.xmin() >= crop.x + crop.width)
        { // outside right
        }
        else if (b.ymax() < crop.y)
        { // outside above
        }
        else if (b.ymin() >= crop.y + crop.height)
        { // outside below
        }
        else
        {
            if (b.xmin() < crop.x)
            {
                b.set_xmin(0);
            }
            else
            {
                b.set_xmin(b.xmin() - crop.x);
            }
            if (b.ymin() < crop.y)
            {
                b.set_ymin(0);
            }
            else
            {
                b.set_ymin(b.ymin() - crop.y);
            }
            if (b.xmax() >= crop.x + crop.width)
            {
                b.set_xmax(crop.width - 1);
            }
            else
            {
                b.set_xmax(b.xmax() - crop.x);
            }
            if (b.ymax() >= crop.y + crop.height)
            {
                b.set_ymax(crop.height - 1);
            }
            else
            {
                b.set_ymax(b.ymax() - crop.y);
            }

            if (pptr->flip)
            {
                auto xmax = b.xmax();
                b.set_xmax(crop.width - b.xmin() - 1);
                b.set_xmin(crop.width - xmax - 1);
            }

            // now rescale box
            b = b.rescale(x_scale, y_scale);

            rc.push_back(b);
        }
    }
    return rc;
}

int nervana::boundingbox::extractor::get_label(const json& object) const
{
    string obj_name = object["name"];
    auto   found    = label_map.find(obj_name);
    if (found == label_map.end())
    {
        // did not find the label in the ctor supplied label list
        stringstream ss;
        ss << "label '" << obj_name << "' not found in metadata label list";
        throw invalid_argument(ss.str());
    }
    return found->second;
}

shared_ptr<boundingbox::decoded>
    boundingbox::transformer::transform(shared_ptr<augment::image::params> pptr,
                                        shared_ptr<boundingbox::decoded>   boxes) const
{
    if (pptr->angle != 0)
    {
        return shared_ptr<boundingbox::decoded>();
    }
    shared_ptr<boundingbox::decoded> rc = make_shared<boundingbox::decoded>();
    rc->m_boxes                         = transform_box(boxes->boxes(), pptr);

    return rc;
}

boundingbox::loader::loader(const boundingbox::config& cfg)
    : max_bbox{cfg.max_bbox_count}
{
}

void boundingbox::loader::load(const vector<void*>&             outlist,
                               shared_ptr<boundingbox::decoded> boxes) const
{
    float* data         = (float*)outlist[0];
    size_t output_count = min(max_bbox, boxes->boxes().size());
    int    i            = 0;
    for (; i < output_count; i++)
    {
        data[0] = boxes->boxes()[i].xmin();
        data[1] = boxes->boxes()[i].ymin();
        data[2] = boxes->boxes()[i].xmax();
        data[3] = boxes->boxes()[i].ymax();
        data += 4;
    }
    for (; i < max_bbox; i++)
    {
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 0;
        data += 4;
    }
}
