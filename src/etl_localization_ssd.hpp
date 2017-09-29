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
#include <tuple>
#include <random>

#include "interface.hpp"
#include "etl_boundingbox.hpp"
#include "etl_image.hpp"
#include "util.hpp"
#include "box.hpp"

namespace nervana
{
    namespace localization
    {
        namespace ssd
        {
            class decoded;
            class params;
            class config;

            class extractor;
            class transformer;
            class loader;
        }
    }
}

class nervana::localization::ssd::config : public nervana::interface::config
{
public:
    size_t                   height;
    size_t                   width;
    size_t                   max_gt_boxes = 64;
    std::vector<std::string> class_names;
    std::string              name;

    // Derived values
    size_t output_buffer_size;
    std::unordered_map<std::string, int> class_name_map;

    explicit config(nlohmann::json js);

private:
    config() {}
    void validate();

    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(height, mode::REQUIRED),
        ADD_SCALAR(width, mode::REQUIRED),
        ADD_SCALAR(class_names, mode::REQUIRED),
        ADD_SCALAR(name, mode::OPTIONAL),
        ADD_SCALAR(max_gt_boxes, mode::OPTIONAL)};
};

class nervana::localization::ssd::decoded : public boundingbox::decoded
{
    friend class extractor;

public:
    decoded() {}
    virtual ~decoded() override {}
    cv::Size                      output_image_size;
    cv::Size                      input_image_size;
    std::vector<boundingbox::box> gt_boxes;
};

class nervana::localization::ssd::extractor : public nervana::interface::extractor<ssd::decoded>
{
public:
    extractor(const ssd::config&);
    virtual ~extractor() {}
    virtual std::shared_ptr<ssd::decoded> extract(const void* data, size_t size) const override;

private:
    extractor() = delete;
    boundingbox::extractor bbox_extractor;
    config                 cfg;
};

class nervana::localization::ssd::transformer
    : public interface::transformer<ssd::decoded, augment::image::params>
{
public:
    transformer() = default;

    virtual ~transformer() {}
    std::shared_ptr<ssd::decoded> transform(std::shared_ptr<augment::image::params> txs,
                                            std::shared_ptr<ssd::decoded> mp) const override;

private:
};

class nervana::localization::ssd::loader : public interface::loader<ssd::decoded>
{
public:
    loader(const ssd::config&);

    virtual ~loader() {}
    void load(const std::vector<void*>& buf_list, std::shared_ptr<ssd::decoded> mp) const override;

private:
    loader() = delete;
    size_t max_gt_boxes;
};
