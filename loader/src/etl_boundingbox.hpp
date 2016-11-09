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

#include <string>
#include <unordered_map>

#include "interface.hpp"
#include "etl_image.hpp"
#include "json.hpp"
#include "box.hpp"

namespace nervana
{
    namespace boundingbox
    {
        class decoded;
        class config;
        class extractor;
        class transformer;
        class loader;
        class box;
    }
}

class nervana::boundingbox::box : public nervana::box
{
public:
    bool difficult = false;
    bool truncated = false;
    int label;

    box operator*(float v) const {
        box rc = *this;
        rc.xmin *= v;
        rc.ymin *= v;
        rc.xmax *= v;
        rc.ymax *= v;
        return rc;
    }
};

std::ostream& operator<<(std::ostream&,const nervana::boundingbox::box&);

class nervana::boundingbox::config : public nervana::interface::config
{
public:
    size_t                      height;
    size_t                      width;
    size_t                      max_bbox_count;
    std::vector<std::string>    class_names;
    std::string                 output_type = "float";

    std::unordered_map<std::string,int> label_map;

    config(nlohmann::json js);

private:
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(height, mode::REQUIRED),
        ADD_SCALAR(width, mode::REQUIRED),
        ADD_SCALAR(max_bbox_count, mode::REQUIRED),
        ADD_SCALAR(class_names, mode::REQUIRED),
        ADD_SCALAR(output_type, mode::OPTIONAL, [](const std::string& v){ return output_type::is_valid_type(v); })
    };

    config() {}
    void validate();
};

class nervana::boundingbox::decoded : public interface::decoded_media
{
    friend class transformer;
    friend class extractor;
public:
    decoded();
    bool extract(const char* data, int size, const std::unordered_map<std::string,int>& label_map);
    virtual ~decoded() {}

    const std::vector<boundingbox::box>& boxes() const { return _boxes; }
    int width() const { return _width; }
    int height() const { return _height; }
    int depth() const { return _depth; }

private:
    std::vector<boundingbox::box> _boxes;
    int _width;
    int _height;
    int _depth;
};


class nervana::boundingbox::extractor : public nervana::interface::extractor<nervana::boundingbox::decoded>
{
public:
    extractor(const std::unordered_map<std::string,int>&);
    virtual ~extractor(){}
    virtual std::shared_ptr<boundingbox::decoded> extract(const char*, int) override;
    void extract(const char*, int, std::shared_ptr<boundingbox::decoded>&);

private:
    extractor() = delete;
    std::unordered_map<std::string,int> label_map;
};

class nervana::boundingbox::transformer : public nervana::interface::transformer<nervana::boundingbox::decoded, nervana::image::params>
{
public:
    transformer(const boundingbox::config&);
    virtual ~transformer(){}
    virtual std::shared_ptr<boundingbox::decoded> transform(
                                            std::shared_ptr<image::params>,
                                            std::shared_ptr<boundingbox::decoded>) override;

    static std::vector<boundingbox::box> transform_box( const std::vector<boundingbox::box>& b, const cv::Rect& crop, bool flip, float x_scale, float y_scale);
private:
};

class nervana::boundingbox::loader : public nervana::interface::loader<nervana::boundingbox::decoded>
{
public:
    loader(const boundingbox::config&);
    virtual ~loader(){}
    virtual void load(const std::vector<void*>&, std::shared_ptr<boundingbox::decoded>) override;

private:
    const size_t max_bbox;
};
