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

#include "provider_audio_only.hpp"

using namespace nervana;
using namespace std;

audio_only::audio_only(nlohmann::json js)
    : audio_config(js["audio"])
    , audio_extractor()
    , audio_transformer(audio_config)
    , audio_loader(audio_config)
    , audio_factory(audio_config)
{
    m_output_shapes.insert({"audio",audio_config.get_shape_type()});
}

void audio_only::provide(int idx, variable_buffer_array& in_buf, fixed_buffer_map& out_buf)
{
    vector<char>& datum_in  = in_buf[0].get_item(idx);
    char*         datum_out = out_buf["audio"]->get_item(idx);

    // Process audio data
    auto audio_dec    = audio_extractor.extract(datum_in.data(), datum_in.size());
    auto audio_params = audio_factory.make_params(audio_dec);
    audio_loader.load({datum_out}, audio_transformer.transform(audio_params, audio_dec));
}

size_t audio_only::get_input_count() const
{
    return 1;
}
