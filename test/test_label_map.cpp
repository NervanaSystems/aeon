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

#include "interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_boundingbox.hpp"
#include "etl_label_map.hpp"
#include "json.hpp"

using namespace std;
using namespace nervana;

TEST(label_map, test)
{
    {
        nlohmann::json js = {{"class_names",
                              {"a",
                               "and",
                               "the",
                               "quick",
                               "fox",
                               "cow",
                               "dog",
                               "blue",
                               "black",
                               "brown",
                               "happy",
                               "lazy",
                               "skip",
                               "jumped",
                               "run",
                               "under",
                               "over",
                               "around"}}};
        label_map::config             cfg{js};
        label_map::extractor          extractor{cfg};
        label_map::transformer        transformer;
        label_map::loader             loader{cfg};
        shared_ptr<label_map::params> params = make_shared<label_map::params>();

        auto data = extractor.get_data();
        EXPECT_EQ(2, data["the"]);

        {
            // the word 'jump' is not in the vocab
            string t1        = "the quick brown fox jump over the lazy dog";
            auto   extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(nullptr, extracted);
        }
        {
            string      t1       = "the quick brown fox jumped over the lazy dog";
            vector<int> expected = {2, 3, 9, 4, 13, 16, 2, 11, 6};
            auto        decoded  = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, decoded);
            ASSERT_EQ(expected.size(), decoded->get_data().size());
            for (int i = 0; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }

            // transform should do nothing
            decoded = transformer.transform(params, decoded);
            for (int i = 0; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }

            // loader
            const int    pad_size    = 100;
            const int    data_size   = cfg.max_label_count() * sizeof(uint32_t);
            const int    buffer_size = data_size + pad_size;
            vector<char> buffer(buffer_size);
            fill_n(buffer.begin(), buffer.size(), 0xFF);
            loader.load({buffer.data()}, decoded);

            char* data_p = buffer.data();
            int   i      = 0;
            for (; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], unpack<int32_t>(&data_p[i * sizeof(int32_t)]));
            }
            for (; i < cfg.max_label_count(); i++)
            {
                EXPECT_EQ(0, unpack<int32_t>(&data_p[i * sizeof(int32_t)]));
            }
            // check for overrun
            for (i *= 4; i < buffer_size; i++)
            {
                EXPECT_EQ(buffer.data()[i], (char)0xFF);
            }
        }
    }
    {
        nlohmann::json js = {{"class_names",
                              {"a",
                               "and",
                               "the",
                               "quick",
                               "fox",
                               "cow",
                               "dog",
                               "blue",
                               "black",
                               "brown",
                               "happy",
                               "lazy",
                               "skip",
                               "jumped",
                               "run",
                               "under",
                               "over",
                               "around"}}};
        label_map::config    cfg{js};
        label_map::extractor extractor(cfg);
        auto                 data = extractor.get_data();
        EXPECT_EQ(2, data["the"]);

        {
            // the word 'jump' is not in the vocab
            string t1        = "the quick brown fox jump over the lazy dog";
            auto   extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(nullptr, extracted);
        }
        {
            string      t1        = "the quick brown fox jumped over the lazy dog";
            vector<int> expected  = {2, 3, 9, 4, 13, 16, 2, 11, 6};
            auto        extracted = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, extracted);
            shared_ptr<label_map::decoded> decoded =
                static_pointer_cast<label_map::decoded>(extracted);
            ASSERT_EQ(expected.size(), decoded->get_data().size());
            for (int i = 0; i < expected.size(); i++)
            {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }
    }
}
