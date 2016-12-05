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

#include "provider_interface.hpp"
#include "etl_boundingbox.hpp"
#include "etl_image.hpp"

namespace nervana
{
    class image_boundingbox;
}

class nervana::image_boundingbox : public provider_interface
{
public:
    image_boundingbox(nlohmann::json js);
    virtual ~image_boundingbox() {}
    void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf);
    size_t get_input_count() const override;

private:
    image_boundingbox() = delete;
    image::config       image_config;
    boundingbox::config bbox_config;

    image::extractor     image_extractor;
    image::transformer   image_transformer;
    image::loader        image_loader;
    image::param_factory image_factory;

    boundingbox::extractor   bbox_extractor;
    boundingbox::transformer bbox_transformer;
    boundingbox::loader      bbox_loader;

    std::default_random_engine _r_eng;
};
