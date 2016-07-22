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
#include "etl_audio.hpp"
#include "etl_char_map.hpp"
#include "wav_data.hpp"
#include "json.hpp"
#include "util.hpp"
#include "buffer_in.hpp"

using namespace std;
using namespace nervana;

TEST(provider,audio_transcript) {
    nlohmann::json js = {{"media","audio_transcript"},
                         {"data_config",{
                            {"type", "audio"},
                            {"config", {
                              {"max_duration","2000 milliseconds"},
                              {"frame_length","1024 samples"},
                              {"frame_stride","256 samples"},
                              {"sample_freq_hz",44100},
                              {"feature_type","specgram"}
                            }}}},
                         {"target_config",{
                            {"type", "transcript"},
                            {"config", {
                              {"alphabet","ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()"},
                              {"max_length",50}
                            }}}}};

    // Create the config
    shared_ptr<transcribed_audio> media = static_pointer_cast<transcribed_audio>(nervana::train_provider_factory::create(js));
    // auto cmap = media->trans_config.get_cmap();

    // Get buffer output shapes for data and target. Here there are only 1 input and 1 output.
    const vector<nervana::shape_type>& oshapes = media->get_oshapes();
    size_t dsize = oshapes[0].get_byte_size();
    size_t tsize = oshapes[1].get_byte_size();
    // cout << "dsize is " << dsize << " and tsize is " << tsize << endl;

    size_t batch_size = 128;

    // Generate a simple sine wav
    float sine_freq = 400;
    int16_t sine_ampl = 500;
    sinewave_generator sg{sine_freq, sine_ampl};
    int wav_len_sec = 4, sample_freq = 44100;
    bool stereo = false;

    wav_data wav(sg, wav_len_sec, sample_freq, stereo);
    uint32_t wavbufsize = wav_data::HEADER_SIZE + wav.nbytes();

    vector<char> buf(wavbufsize);
    wav.write_to_buffer(&buf[0], wavbufsize);

    // Generate a fake transcript
    string t1 = "The quick brown fox jumped over the lazy dog";
    vector<char> t1_char(t1.begin(), t1.end());

    string t2 = "A much more interesting sentence.";
    vector<char> t2_char(t2.begin(), t2.end());

    // Create the input buffer
    buffer_in_array bp({0,0});
    buffer_in& data_p = *bp[0];
    buffer_in& target_p = *bp[1];

    // Fill the input buffer
    for (int i=0; i<batch_size; i++) {
        data_p.addItem(buf);
        target_p.addItem(t1_char);
    }
    EXPECT_EQ(data_p.getItemCount(), batch_size);
    EXPECT_EQ(target_p.getItemCount(), batch_size);

    // Generate an output buffer
    buffer_out_array outBuf({dsize, tsize, 4}, batch_size);

    // Call the provider
    for (int i=0; i<batch_size; i++ ) {
       media->provide(i, bp, outBuf);
    }

    // Check the output data
    // char* data_out = outBuf[1]->getItem(0);


    // Check first target starts with "t"(19) and ends with "g"(6)
    char* target_out = outBuf[1]->getItem(0);
    // for (int i=0; i<t1.length(), i++) {
    //     ASSERT_EQ(unpack_le<uint8_t>(target_out, i), cmap[t1[i]]);
    // }
    ASSERT_EQ(unpack_le<uint8_t>(target_out, 0), 19);
    ASSERT_EQ(unpack_le<uint8_t>(target_out, t1.length() - 1), 6);

    // Check last target starts with "t"(19) and ends with "g"(6)
    target_out = outBuf[1]->getItem(batch_size - 1);
    ASSERT_EQ(unpack_le<uint8_t>(target_out, 0), 19);
    ASSERT_EQ(unpack_le<uint8_t>(target_out, t1.length() - 1), 6);

    // Check the first transcript length (should be 44)
    char* len_out = outBuf[2]->getItem(0);
    ASSERT_EQ(unpack_le<uint32_t>(len_out, 0), t1.length());

    // Check the last transcript length is also 44
    len_out = outBuf[2]->getItem(batch_size-1);
    ASSERT_EQ(unpack_le<uint32_t>(len_out, 0), t1.length());
    // cout << "transcript_length is " << transcript_length << endl;
    // uint32_t transcript_length2;
    // memcpy(&transcript_length2, outBuf[2]->getItem(0), 1);
    // cout << "transcript_length2 is " << transcript_length2 << endl;
    // This is currently failing...
    // ASSERT_EQ(max_length, t1.length())


}
