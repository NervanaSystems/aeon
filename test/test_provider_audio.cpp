/*
 Copyright 2016 Intel(R) Nervana(TM)
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
#include "etl_label.hpp"
#include "etl_char_map.hpp"
#include "wav_data.hpp"
#include "json.hpp"
#include "util.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

TEST(provider, audio_classify)
{
    nlohmann::json js_label = {{"type", "label"}, {"binary", true}};
    nlohmann::json js_audio = {{"type", "audio"},
                               {"max_duration", "2000 milliseconds"},
                               {"frame_length", "1024 samples"},
                               {"frame_stride", "256 samples"},
                               {"sample_freq_hz", 44100},
                               {"feature_type", "specgram"}};
    nlohmann::json js = {{"etl", {js_audio, js_label}}};

    auto media   = nervana::provider_factory::create(js);
    auto oshapes = media->get_output_shapes();

    auto buf_names = media->get_buffer_names();
    ASSERT_EQ(2, buf_names.size());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "audio"), buf_names.end());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "label"), buf_names.end());

    size_t batch_size = 128;

    wav_data     wav(sinewave_generator(400, 200), 3, 16000, false);
    vector<char> buf(wav_data::HEADER_SIZE + wav.nbytes());
    wav.write_to_buffer(&buf[0], buf.size());

    fixed_buffer_map    out_buf(oshapes, batch_size);
    encoded_record_list bp;

    for (int i = 0; i < batch_size; i++)
    {
        encoded_record record;
        record.add_element(buf);
        vector<char> packed_int(4);
        pack<int>(&packed_int[0], 42 + i);
        record.add_element(packed_int);
        bp.add_record(record);
    }

    EXPECT_EQ(bp.size(), batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        media->provide(i, bp, out_buf);
    }

    for (int i = 0; i < batch_size; i++)
    {
        int target_value = unpack<int>(out_buf["label"]->get_item(i));
        EXPECT_EQ(42 + i, target_value);
    }
}

TEST(provider, transcript_length_check)
{
    uint32_t       max_length    = 15;
    nlohmann::json js_transcript = {{{"type", "char_map"},
                                     {"alphabet", "abcdefgß "},
                                     {"max_length", max_length},
                                     {"emit_length", true}}};

    nlohmann::json js = {{"etl", js_transcript}};

    auto media     = nervana::provider_factory::create(js);
    auto oshapes   = media->get_output_shapes();
    auto buf_names = media->get_buffer_names();

    // Ensure that we have two output buffers (an extra one for the transcript length since emit_length == true)
    ASSERT_EQ(2, buf_names.size());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "char_map"), buf_names.end());
    ASSERT_NE(find(buf_names.begin(), buf_names.end(), "char_map_length"), buf_names.end());

    size_t batch_size = 4;

    fixed_buffer_map                   out_buf(oshapes, batch_size);
    encoded_record_list                bp;
    std::vector<string>                transcripts{"abcß", "ßßad", "abcabc", "ddefggf"};
    std::vector<uint32_t>              expected_lengths{4, 4, 6, 7};
    std::vector<std::vector<uint32_t>> expected_encodings{
        {0, 1, 2, 7}, {7, 7, 0, 3}, {0, 1, 2, 0, 1, 2}, {3, 3, 4, 5, 6, 6, 5}};
    for (auto&& s : transcripts)
    {
        encoded_record record;
        record.add_element(s.data(), s.length());
        bp.add_record(record);
    }

    EXPECT_EQ(bp.size(), batch_size);

    for (int i = 0; i < batch_size; i++)
    {
        media->provide(i, bp, out_buf);
    }

    // Check that the lengths are emitted as expected
    for (int i = 0; i < batch_size; i++)
    {
        uint32_t target_length = unpack<uint32_t>(out_buf["char_map_length"]->get_item(i));
        EXPECT_EQ(target_length, expected_lengths[i]);
    }

    for (int i = 0; i < batch_size; i++)
    {
        for (uint32_t j = 0; j < max_length; ++j)
        {
            uint32_t loaded_transcript_j =
                unpack<uint32_t>(out_buf["char_map"]->get_item(i), j * sizeof(uint32_t));
            if (j < expected_encodings[i].size())
            {
                EXPECT_EQ(loaded_transcript_j, expected_encodings[i][j]);
            }
            else
            {
                EXPECT_EQ(loaded_transcript_j, 0);
            }
        }
    }
}

TEST(provider, audio_transcript)
{
    nlohmann::json js_audio = {{"type", "audio"},
                               {"max_duration", "2000 milliseconds"},
                               {"frame_length", "1024 samples"},
                               {"frame_stride", "256 samples"},
                               {"sample_freq_hz", 44100},
                               {"feature_type", "specgram"},
                               {"emit_length", true}};
    std::string alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()");

    nlohmann::json js_transcript = {
        {"type", "char_map"}, {"alphabet", alphabet}, {"max_length", 50}, {"emit_length", true}};
    nlohmann::json js = {{"etl", {js_audio, js_transcript}}};

    // Create the config
    auto media = nervana::provider_factory::create(js);

    // Create the character map that should be in the provider
    std::unordered_map<wchar_t, uint32_t> cmap;
    uint32_t     idx        = 0;
    std::wstring w_alphabet = to_wstring(alphabet);
    for (auto& c : w_alphabet)
    {
        cmap.insert({std::towupper(c), idx++});
    }

    // Generate a simple sine wav
    wav_data     wav(sinewave_generator(400, 500), 1, 44100, false);
    vector<char> buf(wav_data::HEADER_SIZE + wav.nbytes());
    wav.write_to_buffer(&buf[0], buf.size());

    // Generate alternating fake transcripts
    vector<string> tr{"The quick brown fox jumped over the lazy dog",
                      "A much more interesting sentence."};
    vector<char> tr0_char(tr[0].begin(), tr[0].end());
    vector<char> tr1_char(tr[1].begin(), tr[1].end());

    // Create the input buffer
    size_t              batch_size = 128;
    encoded_record_list bp;
    for (int i = 0; i < batch_size; i++)
    {
        encoded_record record;
        record.add_element(buf);
        record.add_element(((i % 2) == 0 ? tr0_char : tr1_char));
        bp.add_record(record);
    }
    EXPECT_EQ(bp.size(), batch_size);

    // Generate output buffers using shapes from the provider
    auto             oshapes = media->get_output_shapes();
    fixed_buffer_map out_buf(oshapes, batch_size);

    // Call the provider
    for (int i = 0; i < batch_size; i++)
    {
        media->provide(i, bp, out_buf);
    }

    // Check target sequences against their source string
    for (int i = 0; i < batch_size; i++)
    {
        auto orig_string = tr[i % 2];
        int  j           = 0;
        for (auto c : orig_string)
        {
            uint32_t loaded_transcript_j = unpack<uint32_t>(out_buf["char_map"]->get_item(i), j);
            ASSERT_EQ(loaded_transcript_j, cmap[std::towupper(c)]);
            j += sizeof(uint32_t);
        }
    }

    // Check the audio and transcript lengths match source string length
    for (int i = 0; i < batch_size; i++)
    {
        ASSERT_EQ(unpack<uint32_t>(out_buf["audio_length"]->get_item(i)), wav.nsamples());
        ASSERT_EQ(unpack<uint32_t>(out_buf["char_map_length"]->get_item(i)), tr[i % 2].length());
    }
}

TEST(provider, audio_transcript_extends_max_duration)
{
    nlohmann::json js_audio = {{"type", "audio"},
                               {"max_duration", "2000 milliseconds"},
                               {"frame_length", "1024 samples"},
                               {"frame_stride", "256 samples"},
                               {"sample_freq_hz", 44100},
                               {"feature_type", "specgram"},
                               {"emit_length", true}};
    std::string alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()");

    nlohmann::json js_transcript = {
        {"type", "char_map"}, {"alphabet", alphabet}, {"max_length", 50}, {"emit_length", true}};
    nlohmann::json js = {{"etl", {js_audio, js_transcript}}};

    // Create the config
    auto media = nervana::provider_factory::create(js);

    // Create the character map that should be in the provider
    std::unordered_map<wchar_t, uint32_t> cmap;
    uint32_t     idx        = 0;
    std::wstring w_alphabet = to_wstring(alphabet);
    for (auto& c : w_alphabet)
    {
        cmap.insert({std::towupper(c), idx++});
    }

    // Generate a simple sine wav
    wav_data     wav(sinewave_generator(400, 500), 4, 44100, false);
    vector<char> buf(wav_data::HEADER_SIZE + wav.nbytes());
    wav.write_to_buffer(&buf[0], buf.size());

    // Generate alternating fake transcripts
    vector<string> tr{"The quick brown fox jumped over the lazy dog",
                      "A much more interesting sentence."};
    vector<char> tr0_char(tr[0].begin(), tr[0].end());
    vector<char> tr1_char(tr[1].begin(), tr[1].end());

    // Create the input buffer
    size_t              batch_size = 128;
    encoded_record_list bp;
    for (int i = 0; i < batch_size; i++)
    {
        encoded_record record;
        record.add_element(buf);
        record.add_element(((i % 2) == 0 ? tr0_char : tr1_char));
        bp.add_record(record);
    }
    EXPECT_EQ(bp.size(), batch_size);

    // Generate output buffers using shapes from the provider
    auto             oshapes = media->get_output_shapes();
    fixed_buffer_map out_buf(oshapes, batch_size);

    // Call the provider
    for (int i = 0; i < batch_size; i++)
    {
        media->provide(i, bp, out_buf);
    }

    // Check target sequences against their source string
    for (int i = 0; i < batch_size; i++)
    {
        auto orig_string = tr[i % 2];
        int  j           = 0;
        for (auto c : orig_string)
        {
            uint32_t loaded_transcript_j = unpack<uint32_t>(out_buf["char_map"]->get_item(i), j);
            ASSERT_EQ(loaded_transcript_j, cmap[std::towupper(c)]);
            j += sizeof(uint32_t);
        }
    }

    // Check the audio and transcript lengths match source string length
    for (int i = 0; i < batch_size; i++)
    {
        ASSERT_EQ(unpack<uint32_t>(out_buf["audio_length"]->get_item(i)),
                  2 * 44100 /*max length * sample_freq*/);
        ASSERT_EQ(unpack<uint32_t>(out_buf["char_map_length"]->get_item(i)), tr[i % 2].length());
    }
}
