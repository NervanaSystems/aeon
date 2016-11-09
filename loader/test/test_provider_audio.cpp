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

#include "gtest/gtest.h"

#include <typeinfo>

#include "provider_factory.hpp"
#include "provider_audio_classifier.hpp"
#include "provider_audio_only.hpp"
#include "provider_audio_transcriber.hpp"
#include "etl_audio.hpp"
#include "etl_label.hpp"
#include "etl_char_map.hpp"
#include "wav_data.hpp"
#include "json.hpp"
#include "util.hpp"
#include "buffer_in.hpp"

using namespace std;
using namespace nervana;


TEST(provider,audio_classify)
{
    nlohmann::json js = {{"type","audio,label"},
                         {"audio", {
                            {"max_duration","2000 milliseconds"},
                            {"frame_length","1024 samples"},
                            {"frame_stride","256 samples"},
                            {"sample_freq_hz",44100},
                            {"feature_type","specgram"}}},
                         {"label", {{"binary",true}}}};
    auto media = nervana::provider_factory::create(js);
    const vector<nervana::shape_type>& oshapes = media->get_oshapes();

    size_t dsize = oshapes[0].get_byte_size();
    size_t tsize = 4;

    size_t batch_size = 128;

    wav_data wav(sinewave_generator(400, 200), 3, 16000, false);
    vector<char> buf(wav_data::HEADER_SIZE + wav.nbytes());
    wav.write_to_buffer(&buf[0], buf.size());

    buffer_in_array bp(2);
    buffer_in& data_p = *bp[0];
    buffer_in& target_p = *bp[1];

    buffer_out_array outBuf({dsize, tsize}, batch_size);

    for (int i=0; i<batch_size; i++) {
        data_p.add_item(buf);
        vector<char> packed_int(4);
        pack<int>(&packed_int[0], 42 + i);
        target_p.add_item( packed_int );
    }

    EXPECT_EQ(data_p.get_item_count(),batch_size);

    for (int i=0; i<batch_size; i++ ) {
        media->provide(i, bp, outBuf);
    }

    for (int i=0; i<batch_size; i++ ) {
        int target_value = unpack<int>(outBuf[1]->get_item(i));
        EXPECT_EQ(42+i, target_value);
    }
}


TEST(provider,audio_transcript)
{
    nlohmann::json js = {{"type","audio,transcription"},
                         {"audio", {
                            {"max_duration","2000 milliseconds"},
                            {"frame_length","1024 samples"},
                            {"frame_stride","256 samples"},
                            {"sample_freq_hz",44100},
                            {"feature_type","specgram"}}},
                         {"transcription", {
                            {"alphabet","ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()"},
                            {"pack_for_ctc", true},
                            {"max_length",50}}}};

    // Create the config
    auto media = dynamic_pointer_cast<audio_transcriber>(nervana::provider_factory::create(js));

    // Get the character map
    auto cmap = media->get_cmap();

    size_t batch_size = 128;

    // Generate a simple sine wav
    float sine_freq = 400;
    int16_t sine_ampl = 500;
    sinewave_generator sg{sine_freq, sine_ampl};
    int wav_len_sec = 4, sample_freq = 44100;
    bool stereo = false;

    wav_data wav(sg, wav_len_sec, sample_freq, stereo);
    vector<char> buf(wav_data::HEADER_SIZE + wav.nbytes());
    wav.write_to_buffer(&buf[0], buf.size());

    // Generate alternating fake transcripts
    vector<string> tr {"The quick brown fox jumped over the lazy dog",
                       "A much more interesting sentence."};
    vector<char> tr0_char(tr[0].begin(), tr[0].end());
    vector<char> tr1_char(tr[1].begin(), tr[1].end());

    // Create the input buffer
    buffer_in_array bp(2);
    buffer_in& data_p = *bp[0];
    buffer_in& target_p = *bp[1];

    // Fill the input buffer
    for (int i=0; i<batch_size; i++) {
        data_p.add_item(buf);
        target_p.add_item( ( (i % 2) == 0 ? tr0_char : tr1_char) );
    }

    EXPECT_EQ(data_p.get_item_count(), batch_size);
    EXPECT_EQ(target_p.get_item_count(), batch_size);

    // Generate output buffers using shapes from the provider
    buffer_out_array outBuf({media->get_oshapes()[0].get_byte_size(),
                             media->get_oshapes()[1].get_byte_size(),
                             media->get_oshapes()[2].get_byte_size(),
                             media->get_oshapes()[3].get_byte_size()},
                            batch_size);

    // Call the provider
    for (int i=0; i<batch_size; i++)
    {
        media->provide(i, bp, outBuf);
    }

    // Check target sequences against their source string
    for (int i=0; i<batch_size; i++)
    {
        char* target_out = outBuf[1]->get_item(i);
        auto orig_string = tr[i % 2];
        for (auto c : orig_string)
        {
            ASSERT_EQ(unpack<uint8_t>(target_out++),
                      cmap[std::toupper(c)]);
        }
    }

    // Check the transcript lengths match source string length
    for (int i=0; i<batch_size; i++)
    {
        ASSERT_EQ(unpack<uint32_t>(outBuf[2]->get_item(i)), tr[i % 2].length());
    }

    for (int i=0; i<batch_size; i++)
    {
        ASSERT_EQ(unpack<uint32_t>(outBuf[3]->get_item(i)), 100);
    }

    // Do the packing
    media->post_process(outBuf);
    string combined_string = tr[0] + tr[1];
    uint32_t packed_length = combined_string.size() * batch_size / 2;

    // Check that target sequence contains abutted vals corresponding to original strings
    char* target_ptr = outBuf[1]->data();
    for (int i=0; i<packed_length; i++)
    {
        char c = combined_string[i % combined_string.size()];
        ASSERT_EQ(unpack<uint8_t>(target_ptr++),
                  cmap[std::toupper(c)]);
    }
    for (int i=packed_length; i<outBuf[1]->size(); i++)
    {
        ASSERT_EQ(0, *(target_ptr++));
    }
}
