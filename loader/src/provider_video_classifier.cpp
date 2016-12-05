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

#include "provider_video_classifier.hpp"

using namespace nervana;
using namespace std;

video_classifier::video_classifier(nlohmann::json js)
    : video_config(js["video"])
    , video_extractor(video_config)
    , video_transformer(video_config)
    , video_loader(video_config)
    , frame_factory(video_config.frame)
    , label_config(js["label"])
    , label_extractor(label_config)
    , label_loader(label_config)
{
    m_output_shapes.insert({"video", video_config.get_shape_type()});
    m_output_shapes.insert({"label", label_config.get_shape_type()});
}

void video_classifier::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf)
{
    std::vector<char>& datum_in   = in_buf[0]->get_item(idx);
    std::vector<char>& target_in  = in_buf[1]->get_item(idx);
    char*              datum_out  = out_buf["video"]->get_item(idx);
    char*              target_out = out_buf["label"]->get_item(idx);

    if (datum_in.size() == 0)
    {
        throw std::runtime_error("received encoded video with size 0, at idx " + to_string(idx));
    }

    // Process video data
    auto video_dec    = video_extractor.extract(datum_in.data(), datum_in.size());
    auto frame_params = frame_factory.make_params(video_dec);
    video_loader.load({datum_out}, video_transformer.transform(frame_params, video_dec));

    // Process target data
    auto label_dec = label_extractor.extract(target_in.data(), target_in.size());
    label_loader.load({target_out}, label_dec);
}

size_t video_classifier::get_input_count() const
{
    return 2;
}
