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

#include <vector>
#include <string>
#include <sstream>
#include <random>

#include "gtest/gtest.h"
#include "gen_image.hpp"
#include "cpio.hpp"

#include "etl_char_map.hpp"

using namespace std;
using namespace nervana;

TEST(char_map, bad)
{
    nlohmann::json js = {{"alphabet", "abcccc "}, {"max_length", 15}};
    EXPECT_THROW(char_map::config cfg{js}, std::runtime_error);
}

TEST(char_map, emit_length_check)
{
    uint32_t       max_length = 15;
    nlohmann::json js         = {
        {"alphabet", "abcdefg "}, {"max_length", max_length}, {"emit_length", true}};
    char_map::config    cfg{js};
    char_map::extractor extractor(cfg);
    char_map::loader    loader(cfg);

    {
        string      transcript = "cab a dabba";
        vector<int> expected   = {2, 0, 1, 7, 0, 7, 3, 0, 1, 1, 0};

        vector<uint32_t> _outbuf0(max_length);
        vector<uint32_t> _outbuf1(1);

        uint32_t* outbuf0 = _outbuf0.data();
        uint32_t* outbuf1 = _outbuf1.data();

        auto decoded = extractor.extract(&transcript[0], transcript.size());
        loader.load({outbuf0, outbuf1}, decoded);
        for (int i = 0; i < max_length; i++)
        {
            if (i < expected.size())
            {
                EXPECT_EQ(outbuf0[i], expected[i]) << "at index " << i;
            }
            else
            {
                EXPECT_EQ(outbuf0[i], 0);
            }
        }
        EXPECT_EQ(outbuf1[0], transcript.size());
    }
}

TEST(char_map, test)
{
    {
        string  alphabet        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()ąćŁŻź𐍈✓ß";
        string  transcript      = "The quick brown fox jumps over the lazy dog. Żołądź 𐍈✓";
        size_t  transcript_size = wstring_length(transcript);
        uint8_t max_length      = transcript_size + 5;

        nlohmann::json js = {
            {"alphabet", alphabet}, {"max_length", max_length}, {"unknown_value", 0}};
        char_map::config    cfg{js};
        char_map::extractor extractor(cfg);
        char_map::loader    loader(cfg);

        // Ensure cmap is set up properly
        std::unordered_map<wchar_t, uint32_t> data = cfg.get_cmap();
        EXPECT_EQ(2, data['C']);
        EXPECT_EQ(26, data[' ']);
        EXPECT_EQ(36, data[L'𐍈']);

        // Make sure mapping is correct
        {
            vector<int> expected = {19, 7,  4,  26, 16, 20, 8,  2,  10, 26, 1,  17, 14, 22,
                                    13, 26, 5,  14, 23, 26, 9,  20, 12, 15, 18, 26, 14, 21,
                                    4,  17, 26, 19, 7,  4,  26, 11, 0,  25, 24, 26, 3,  14,
                                    6,  27, 26, 34, 14, 33, 31, 3,  35, 26, 36, 37};
            auto decoded = extractor.extract(&transcript[0], transcript_size);
            // decoded exists
            ASSERT_NE(nullptr, decoded);
            // has the right length
            ASSERT_EQ(expected.size(), transcript_size);
            EXPECT_EQ(decoded->get_length(), transcript_size);
            // and the right values
            for (int i = 0; i < decoded->get_length(); i++)
            {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }

        // handle mapping of unknown characters
        {
            // Skip unknown characters
            string unknown   = "The0:3 ☭q!uick brown";
            string discarded = "The quick brown";
            auto   unk_dec   = extractor.extract(&unknown[0], unknown.size());
            auto   exp_dec   = extractor.extract(&discarded[0], discarded.size());

            for (int i = 0; i < discarded.size(); i++)
            {
                EXPECT_EQ(exp_dec->get_data()[i], unk_dec->get_data()[i]);
            }

            // Unknown characters should be given value of UINT8_MAX
            nlohmann::json js1 = {
                {"alphabet", alphabet}, {"max_length", max_length}, {"unknown_value", 255}};
            char_map::config    unk_cfg{js1};
            char_map::extractor unk_extractor(unk_cfg);
            vector<int>         expected = {19, 7, 4, 255, 255, 255, 26, 255, 16, 255,
                                    20, 8, 2, 10,  26,  1,   17, 14,  22, 13};
            unk_dec = unk_extractor.extract(&unknown[0], unknown.size());
            for (int i = 0; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], unk_dec->get_data()[i]) << "at index " << i;
            }
        }

        // Now check max length truncation
        vector<wchar_t> _outbuf(max_length);
        wchar_t*        outbuf = _outbuf.data();
        {
            string long_str =
                "This is a really long transcript that should overflow the ßuffer at the letter ß "
                "in overflow";
            size_t long_str_size = wstring_length(long_str);
            auto   decoded       = extractor.extract(&long_str[0], long_str_size);
            loader.load({outbuf}, decoded);

            ASSERT_EQ(outbuf[max_length - 1], 38);
        }
    }
}
