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

#include "provider_audio_transcriber.hpp"

using namespace nervana;
using namespace std;

audio_transcriber::audio_transcriber(nlohmann::json js)
    : provider_interface(js, 2)
    , audio_config(js["audio"])
    , trans_config(js["transcription"])
    , audio_extractor()
    , audio_transformer(audio_config)
    , audio_loader(audio_config)
    , audio_factory(audio_config)
    , trans_extractor(trans_config)
    , trans_loader(trans_config)
{
    m_output_shapes.insert({"audio", audio_config.get_shape_type()});
    m_output_shapes.insert({"transcription", trans_config.get_shape_type()});

    shape_type trans_length({1, 1}, output_type("uint32_t"));
    m_output_shapes.insert({"trans_length", trans_length});

    shape_type valid_pct({1, 1}, output_type("uint32_t"));
    m_output_shapes.insert({"valid_pct", valid_pct});
}

void audio_transcriber::provide(int idx, encoded_record_list& in_buf, fixed_buffer_map& out_buf)
{
    vector<char>& datum_in  = in_buf.record(idx).element(0);
    vector<char>& target_in = in_buf.record(idx).element(1);

    char* datum_out  = out_buf["audio"]->get_item(idx);
    char* target_out = out_buf["transcription"]->get_item(idx);
    char* length_out = out_buf["trans_length"]->get_item(idx);
    char* valid_out  = out_buf["valid_pct"]->get_item(idx);

    // Process audio data
    auto audio_dec    = audio_extractor.extract(datum_in.data(), datum_in.size());
    auto audio_params = audio_factory.make_params(audio_dec);
    audio_loader.load({datum_out}, audio_transformer.transform(audio_params, audio_dec));

    // Process target data
    auto trans_dec = trans_extractor.extract(target_in.data(), target_in.size());
    trans_loader.load({target_out}, trans_dec);

    // Save out the length
    uint32_t trans_length = trans_dec->get_length();
    pack(length_out, trans_length);

    // Get the length of each audio record as a percentage of
    // maximum utterance length
    uint32_t valid_pct = 100 * audio_dec->valid_frames / audio_config.time_steps;

    pack(valid_out, valid_pct);
}

void audio_transcriber::post_process(fixed_buffer_map& out_buf)
{
    auto transcription = out_buf["transcription"];
    auto trans_length  = out_buf["trans_length"];
    if (trans_config.pack_for_ctc)
    {
        auto     num_items  = transcription->get_item_count();
        char*    dptr       = transcription->data();
        uint32_t packed_len = 0;

        for (int i = 0; i < num_items; i++)
        {
            uint32_t len = unpack<uint32_t>(trans_length->get_item(i));
            memmove(dptr + packed_len, transcription->get_item(i), len);
            packed_len += len;
        }
        memset(dptr + packed_len, 0, transcription->size() - packed_len);
    }
}
