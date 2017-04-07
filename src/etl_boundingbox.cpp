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

ostream& operator<<(ostream& out, const boundingbox::box& b)
{
    out << (box)b << " label=" << b.label;
    return out;
}

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
                                     std::shared_ptr<boundingbox::decoded>& rc)
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
        box  b;
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
        b.xmax = bndbox["xmax"];
        b.xmin = bndbox["xmin"];
        b.ymax = bndbox["ymax"];
        b.ymin = bndbox["ymin"];
        if (!object["difficult"].is_null())
        {
            b.difficult = object["difficult"];
        }
        if (!object["truncated"].is_null())
        {
            b.truncated = object["truncated"];
        }
        string name  = object["name"];
        auto   found = label_map.find(name);
        if (found == label_map.end())
        {
            // did not find the label in the ctor supplied label list
            stringstream ss;
            ss << "label '" << name << "' not found in metadata label list";
            throw invalid_argument(ss.str());
        }
        else
        {
            b.label = found->second;
        }
        rc->m_boxes.push_back(b);
    }
}

shared_ptr<boundingbox::decoded> boundingbox::extractor::extract(const void* data, size_t size)
{
    shared_ptr<decoded> rc = make_shared<decoded>();
    extract(data, size, rc);
    return rc;
}

boundingbox::transformer::transformer(const boundingbox::config&)
{
}

vector<boundingbox::box>
    boundingbox::transformer::transform_box(const std::vector<boundingbox::box>& boxes,
                                            const cv::Rect&                      crop,
                                            bool                                 flip,
                                            float                                x_scale,
                                            float                                y_scale)
{
    // 1) rotate
    // 2) crop
    // 3) scale
    // 4) flip

    vector<boundingbox::box> rc;
    for (boundingbox::box b : boxes)
    {
        if (b.xmax <= crop.x)
        { // outside left
        }
        else if (b.xmin >= crop.x + crop.width)
        { // outside right
        }
        else if (b.ymax <= crop.y)
        { // outside above
        }
        else if (b.ymin >= crop.y + crop.height)
        { // outside below
        }
        else
        {
            if (b.xmin < crop.x)
            {
                b.xmin = 0;
            }
            else
            {
                b.xmin -= crop.x;
            }
            if (b.ymin < crop.y)
            {
                b.ymin = 0;
            }
            else
            {
                b.ymin -= crop.y;
            }
            if (b.xmax > crop.x + crop.width)
            {
                b.xmax = crop.width;
            }
            else
            {
                b.xmax -= crop.x;
            }
            if (b.ymax > crop.y + crop.height)
            {
                b.ymax = crop.height;
            }
            else
            {
                b.ymax -= crop.y;
            }

            if (flip)
            {
                auto xmax = b.xmax;
                b.xmax    = crop.width - b.xmin - 1;
                b.xmin    = crop.width - xmax - 1;
            }

            // now rescale box
            b.xmin *= x_scale;
            b.xmax *= x_scale;
            b.ymin *= y_scale;
            b.ymax *= y_scale;

            rc.push_back(b);
        }
    }
    return rc;
}

shared_ptr<boundingbox::decoded>
    boundingbox::transformer::transform(shared_ptr<augment::image::params> pptr,
                                        shared_ptr<boundingbox::decoded>   boxes)
{
    if (pptr->angle != 0)
    {
        return shared_ptr<boundingbox::decoded>();
    }
    shared_ptr<boundingbox::decoded> rc   = make_shared<boundingbox::decoded>();
    cv::Rect                         crop = pptr->cropbox;
    float x_scale                         = (float)(pptr->output_size.width) / (float)(crop.width);
    float y_scale = (float)(pptr->output_size.height) / (float)(crop.height);

    rc->m_boxes = transform_box(boxes->boxes(), crop, pptr->flip, x_scale, y_scale);

    return rc;
}

boundingbox::loader::loader(const boundingbox::config& cfg)
    : max_bbox{cfg.max_bbox_count}
{
}

void boundingbox::loader::load(const vector<void*>& outlist, shared_ptr<boundingbox::decoded> boxes)
{
    float* data         = (float*)outlist[0];
    size_t output_count = min(max_bbox, boxes->boxes().size());
    int    i            = 0;
    for (; i < output_count; i++)
    {
        data[0] = boxes->boxes()[i].xmin;
        data[1] = boxes->boxes()[i].ymin;
        data[2] = boxes->boxes()[i].xmax;
        data[3] = boxes->boxes()[i].ymax;
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
