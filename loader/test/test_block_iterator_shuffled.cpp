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

#include "helpers.hpp"
#include "block_iterator_shuffled.hpp"
#include "block_iterator_sequential.hpp"
#include "block_loader_util.hpp"

using namespace std;
using namespace nervana;

TEST(block_iterator_shuffled, sequential_block)
{
    block_loader_alphabet bl(8);
    buffer_in_array       bp(2);

    // ensure that loading successive blocks from block_loader_sequential
    // result in sorted strings
    bl.load_block(bp, 0);
    bl.load_block(bp, 1);
    bl.load_block(bp, 2);

    vector<string> words = buffer_to_vector_of_strings(*bp[0]);

    ASSERT_EQ(sorted(words), true);
}

TEST(block_iterator_shuffled, shuffled_block)
{
    auto                    mbl = make_shared<block_loader_alphabet>(5);
    block_iterator_shuffled bis(mbl);
    buffer_in_array         bp(2);

    uint32_t num_records = mbl->object_count();

    // ensure that loading successive blocks from shuffling iterator
    // result in unsorted strings
    for (int i = 0; i < mbl->block_count(); ++i)
    {
        bis.read(bp);
    }

    vector<string> words_a = buffer_to_vector_of_strings(*bp[0]);
    vector<string> words_b = buffer_to_vector_of_strings(*bp[1]);

    ASSERT_EQ(words_a.size(), num_records);
    ASSERT_EQ(words_b.size(), num_records);

    // ensure that there is correspondence between the elements of each record
    for (uint32_t i = 0; i < num_records; i++)
    {
        ASSERT_EQ(words_a[i], words_b[i]);
    }

    ASSERT_EQ(sorted(words_a), false);
    // now sort the words and make sure they are all unique.  We should
    // have loaded an entire 'epoch' and have no duplicates
    assert_vector_unique(words_a);
}
