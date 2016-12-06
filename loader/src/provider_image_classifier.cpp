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

#include "provider_image_classifier.hpp"

using namespace nervana;
using namespace std;

image_classifier::image_classifier(nlohmann::json js)
    : image_config(js["image"])
    ,
    // must use a default value {} otherwise, segfault ...
    label_config(js["label"])
    , image_extractor(image_config)
    , image_transformer(image_config)
    , image_loader(image_config)
    , image_factory(image_config)
    , label_extractor(label_config)
    , label_loader(label_config)
{
    m_output_shapes.insert({"image",image_config.get_shape_type()});
    m_output_shapes.insert({"label",label_config.get_shape_type()});
}

void image_classifier::provide(int idx, variable_buffer_array& in_buf, fixed_buffer_map& out_buf)
{
    std::vector<char>& datum_in   = in_buf[0].get_item(idx);
    std::vector<char>& target_in  = in_buf[1].get_item(idx);
    char*              datum_out  = out_buf["image"]->get_item(idx);
    char*              target_out = out_buf["label"]->get_item(idx);

    if (datum_in.size() == 0)
    {
        std::stringstream ss;
        ss << "received encoded image with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process image data
    auto image_dec    = image_extractor.extract(datum_in.data(), datum_in.size());
    auto image_params = image_factory.make_params(image_dec);
    image_loader.load({datum_out}, image_transformer.transform(image_params, image_dec));

    // Process target data
    auto label_dec = label_extractor.extract(target_in.data(), target_in.size());
    label_loader.load({target_out}, label_dec);
}

size_t image_classifier::get_input_count() const
{
    return 2;
}
