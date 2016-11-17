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

TEST(char_map, test)
{
    {
        string  alphabet   = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()";
        string  transcript = "The quick brown fox jumps over the lazy dog";
        uint8_t max_length = transcript.size() + 5;

        nlohmann::json      js = {{"alphabet", alphabet}, {"max_length", max_length}, {"unknown_value", 0}};
        char_map::config    cfg{js};
        char_map::extractor extractor(cfg);
        char_map::loader    loader(cfg);

        // Ensure cmap is set up properly
        std::unordered_map<char, uint8_t> data = cfg.get_cmap();
        EXPECT_EQ(2, data['C']);
        EXPECT_EQ(26, data[' ']);

        // Make sure mapping is correct and extra characters are mapped to 0
        {
            vector<int> expected = {19, 7,  4,  26, 16, 20, 8,  2,  10, 26, 1,  17, 14, 22, 13, 26, 5, 14, 23, 26, 9, 20, 12, 15,
                                    18, 26, 14, 21, 4,  17, 26, 19, 7,  4,  26, 11, 0,  25, 24, 26, 3, 14, 6,  0,  0, 0,  0,  0};
            auto decoded = extractor.extract(&transcript[0], transcript.size());
            // decoded exists
            ASSERT_NE(nullptr, decoded);
            // has the right length
            EXPECT_EQ(expected.size(), max_length);
            // and the right values
            for (int i = 0; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }

        // handle mapping of unknown characters
        {
            // Skip unknown characters
            string unknown   = "The0:3 ?q!uick brown";
            string discarded = "The quick brown";
            auto   unk_dec   = extractor.extract(&unknown[0], unknown.size());
            auto   exp_dec   = extractor.extract(&discarded[0], discarded.size());

            for (int i = 0; i < discarded.size(); i++)
            {
                EXPECT_EQ(exp_dec->get_data()[i], unk_dec->get_data()[i]);
            }

            // Unknown characters should be given value of UINT8_MAX
            nlohmann::json      js = {{"alphabet", alphabet}, {"max_length", max_length}, {"unknown_value", 255}};
            char_map::config    unk_cfg{js};
            char_map::extractor unk_extractor(unk_cfg);
            vector<int>         expected = {19, 7, 4, 255, 255, 255, 26, 255, 16, 255, 20, 8, 2, 10, 26, 1, 17, 14, 22, 13};
            unk_dec                      = unk_extractor.extract(&unknown[0], unknown.size());
            for (int i = 0; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], unk_dec->get_data()[i]) << "at index " << i;
            }
        }

        // Now check max length truncation
        char outbuf[max_length];
        {
            string long_str = "This is a really long transcript that should overflow the buffer at the letter e in overflow";
            auto   decoded  = extractor.extract(&long_str[0], long_str.size());
            loader.load({outbuf}, decoded);

            ASSERT_EQ(outbuf[max_length - 1], 4);
        }

        // Check zero padding
        {
            string short_str = "now";
            auto   decoded   = extractor.extract(&short_str[0], short_str.size());
            loader.load({outbuf}, decoded);
            for (int i = 3; i < max_length; i++)
            {
                ASSERT_EQ(outbuf[i], 0);
            }
        }
    }
}
