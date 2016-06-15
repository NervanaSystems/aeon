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
#include "argtype.hpp"
#include "imagegen.hpp"
#include "batchfile.hpp"

#include "params.hpp"
#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "etl_label.hpp"
#include "etl_bbox.hpp"
#include "etl_lmap.hpp"
#include "provider.hpp"
#include "json.hpp"

extern image_gen _datagen;

using namespace std;
using namespace nervana;

TEST(etl, lmap) {
    {
        vector<string> vocab = {"a","and","the","quick","fox","cow","dog","blue",
            "black","brown","happy","lazy","skip","jumped","run","under","over","around"};
        lmap::extractor extractor(vocab);
        auto data = extractor.get_data();
        EXPECT_EQ(2,data["the"]);

        {
            // the word 'jump' is not in the vocab
            string t1 = "the quick brown fox jump over the lazy dog";
            auto extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(nullptr, extracted);
        }
        {
            string t1 = "the quick brown fox jumped over the lazy dog";
            vector<int> expected = {2, 3, 9, 4, 13, 16, 2, 11, 6};
            auto extracted = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, extracted);
            shared_ptr<lmap::decoded> decoded = static_pointer_cast<lmap::decoded>(extracted);
            ASSERT_EQ(expected.size(),decoded->get_data().size());
            for( int i=0; i<expected.size(); i++ ) {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }
    }
    {
        stringstream vocab("a and the quick fox cow dog blue black brown happy lazy skip jumped run under over around");
        lmap::extractor extractor(vocab);
        auto data = extractor.get_data();
        EXPECT_EQ(2,data["the"]);

        {
            // the word 'jump' is not in the vocab
            string t1 = "the quick brown fox jump over the lazy dog";
            auto extracted = extractor.extract(&t1[0], t1.size());
            EXPECT_EQ(nullptr, extracted);
        }
        {
            string t1 = "the quick brown fox jumped over the lazy dog";
            vector<int> expected = {2, 3, 9, 4, 13, 16, 2, 11, 6};
            auto extracted = extractor.extract(&t1[0], t1.size());
            ASSERT_NE(nullptr, extracted);
            shared_ptr<lmap::decoded> decoded = static_pointer_cast<lmap::decoded>(extracted);
            ASSERT_EQ(expected.size(),decoded->get_data().size());
            for( int i=0; i<expected.size(); i++ ) {
                EXPECT_EQ(expected[i], decoded->get_data()[i]) << "at index " << i;
            }
        }
    }
}
