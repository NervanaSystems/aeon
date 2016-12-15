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
#include "etl_image.hpp"
#include "etl_localization.hpp"

namespace nervana
{
    class image_localization;
}

class nervana::image_localization : public provider_interface
{
public:
    image_localization(nlohmann::json js);
    void provide(int idx, nervana::encoded_record_list& in_buf, nervana::fixed_buffer_map& out_buf);

private:
    image::config             image_config;
    localization::config      localization_config;
    image::extractor          image_extractor;
    image::transformer        image_transformer;
    image::loader             image_loader;
    image::param_factory      image_factory;
    localization::extractor   localization_extractor;
    localization::transformer localization_transformer;
    localization::loader      localization_loader;
};
