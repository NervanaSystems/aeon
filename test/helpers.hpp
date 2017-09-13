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

#include <vector>
#include <string>

#include "buffer_batch.hpp"
#include "etl_image.hpp"
#include "util.hpp"

std::vector<std::string> buffer_to_vector_of_strings(nervana::encoded_record_list& b);
bool sorted(std::vector<std::string> words);
void dump_vector_of_strings(std::vector<std::string>& words);

void assert_vector_unique(std::vector<std::string>& words);

class element_info
{
public:
    element_info(const std::string& s)
    {
        auto tmp         = nervana::split(s, ':');
        m_record_number  = stod(tmp[0]);
        m_element_number = stod(tmp[1]);
    }

    size_t record_number() const { return m_record_number; }
    size_t element_number() const { return m_element_number; }
private:
    size_t m_record_number;
    size_t m_element_number;
};

class image_params_builder
{
public:
    image_params_builder(std::shared_ptr<nervana::augment::image::params> _obj) { obj = _obj; }
    image_params_builder&
        expand(float expand_ratio, const cv::Size2i& expand_offset, const cv::Size2i& expand_size)
    {
        obj->expand_ratio  = expand_ratio;
        obj->expand_offset = expand_offset;
        obj->expand_size   = expand_size;
        return *this;
    }
    image_params_builder& cropbox(int x, int y, int w, int h)
    {
        obj->cropbox = cv::Rect(x, y, w, h);
        return *this;
    }
    image_params_builder& cropbox(const cv::Rect& cropbox)
    {
        obj->cropbox = cropbox;
        return *this;
    }
    image_params_builder& output_size(int w, int h)
    {
        obj->output_size = cv::Size2i(w, h);
        return *this;
    }
    image_params_builder& angle(int val)
    {
        obj->angle = val;
        return *this;
    }
    image_params_builder& padding(int padding, int crop_offset_x, int crop_offset_y)
    {
        obj->padding             = padding;
        obj->padding_crop_offset = cv::Size2i(crop_offset_x, crop_offset_y);
        return *this;
    }
    image_params_builder& flip(bool val)
    {
        obj->flip = val;
        return *this;
    }
    image_params_builder& lighting(float f1, float f2, float f3)
    {
        obj->lighting = {f1, f2, f3};
        return *this;
    }
    image_params_builder& color_noise_std(float f)
    {
        obj->color_noise_std = f;
        return *this;
    }
    image_params_builder& contrast(float f)
    {
        obj->contrast = f;
        return *this;
    }
    image_params_builder& brightness(float f)
    {
        obj->brightness = f;
        return *this;
    }
    image_params_builder& saturation(float f)
    {
        obj->saturation = f;
        return *this;
    }

    operator std::shared_ptr<nervana::augment::image::params>() const { return obj; }
private:
    std::shared_ptr<nervana::augment::image::params> obj;
};

nlohmann::json create_box(const cv::Rect& rect, const std::string& label);
nlohmann::json create_box(const nervana::boundingbox::box& box, const std::string& label);
nlohmann::json create_metadata(const std::vector<nlohmann::json>& boxes, int width, int height);
