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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "interface.hpp"
#include "etl_image.hpp"

namespace nervana
{
    namespace multicrop
    {
        class config;
        class transformer;
    }
}

class nervana::multicrop::config : public interface::config
{
public:
    // Required config variables
    std::vector<float>     crop_scales;
    nervana::image::config crop_config;
    std::string            name;

    // Optional config variables
    int crop_count = 5;

    // Derived variables
    std::vector<bool> orientations{false};

    config(nlohmann::json js);

private:
    config() {}
    std::vector<std::shared_ptr<interface::config_info_interface>> config_list = {
        ADD_SCALAR(name, mode::OPTIONAL),
        // from image class
        ADD_IGNORE(crop_config),
        // local params
        ADD_SCALAR(crop_scales, mode::REQUIRED),
        ADD_SCALAR(crop_count, mode::OPTIONAL)};

    void validate();
};

class nervana::multicrop::transformer
    : public interface::transformer<image::decoded, augment::image::params>
{
public:
    transformer(const multicrop::config& cfg);
    ~transformer() {}
    virtual std::shared_ptr<image::decoded> transform(std::shared_ptr<augment::image::params>,
                                                      std::shared_ptr<image::decoded>) override;

private:
    image::transformer _crop_transformer;
    std::vector<float> _crop_scales;
    std::vector<bool>  _orientations;

    // By default we include only center crop
    std::vector<cv::Point2f> _offsets{cv::Point2f(0.5, 0.5)};
};
