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

#include "provider_image_pixelmask.hpp"

using namespace nervana;
using namespace std;

image_pixelmask::image_pixelmask(nlohmann::json js)
    : image_config(js["image"])
    , target_config(js["pixelmask"])
    , image_extractor(image_config)
    , image_transformer(image_config)
    , image_loader(image_config)
    , image_factory(image_config)
    , target_extractor(target_config)
    , target_transformer(target_config)
    , target_loader(target_config)
{
    m_output_shapes.insert({"image",image_config.get_shape_type()});
    m_output_shapes.insert({"pixelmask",target_config.get_shape_type()});
}

void image_pixelmask::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf)
{
    std::vector<char>& datum_in   = in_buf[0]->get_item(idx);
    std::vector<char>& target_in  = in_buf[1]->get_item(idx);
    char*              datum_out  = out_buf["image"]->get_item(idx);
    char*              target_out = out_buf["pixelmask"]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    auto image_dec         = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params      = image_factory.make_params(image_dec);
    auto image_transformed = image_transformer.transform(image_params, image_dec);
    image_loader.load({datum_out}, image_transformed);

    // Process target data
    auto target_dec         = target_extractor.extract(target_in.data(), target_in.size());
    auto target_transformed = target_transformer.transform(image_params, target_dec);
    target_loader.load({target_out}, target_transformed);
}

size_t image_pixelmask::get_input_count() const
{
    return 2;
}
