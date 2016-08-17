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

TEST(char_map, bad) {
    nlohmann::json js = {{"alphabet", "abcccc "},
                         {"max_length", 15}};
    EXPECT_THROW(char_map::config cfg{js}, std::runtime_error);

}

TEST(char_map, test) {
    {
        nlohmann::json js = {{"alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ .,()"},
                             {"max_length", 20}};
        char_map::config cfg{js};
        char_map::extractor extractor(cfg);
        auto data = cfg.get_cmap();
        EXPECT_EQ(2, data['C']);

        // handle mapping of unknown character
        {
            string t1 = "The quick brown -fox jump over the lazy dog";
            auto extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(UINT8_MAX, extracted->get_data()[16]);
        }


        {
            string t1 = "The quick brOwn";
            vector<int> expected = {19, 7, 4, 26, 16, 20, 8, 2, 10, 26, 1, 17, 14, 22, 13,
                                    0, 0, 0, 0, 0};
            auto decoded = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, decoded);
            ASSERT_EQ(expected.size(),decoded->get_data().size());
            for( int i=0; i<expected.size(); i++ ) {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }

        char_map::loader loader(cfg);
        int max_length = js["max_length"];
        char outbuf[max_length];
        // Now check max length truncation
        {
            string t1 = "now is the winter of our discontent";
            auto decoded = extractor.extract(&t1[0], t1.size());
            loader.load({outbuf}, decoded);

            ASSERT_EQ(outbuf[max_length - 1], 5);
        }

        // Check zero padding
        {
            string t1 = "now";
            auto decoded = extractor.extract(&t1[0], t1.size());
            loader.load({outbuf}, decoded);
            ASSERT_EQ(outbuf[max_length - 1], 0);
        }

    }
}
