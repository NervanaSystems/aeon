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
#include "shuffled_batch_iterator.hpp"
#include "mock_batch_loader.hpp"

using namespace std;

TEST(shuffled_batch_iterator, sequential_batch_loader) {
    MockBatchLoader bl;

    // buffer_in* dataBuffer = new buffer_in(0);
    // buffer_in* targetBuffer = new buffer_in(0);
    // buffer_in_array bp{dataBuffer, targetBuffer};
    buffer_in_array bp(vector<size_t>{0, 0});

    // ensure that loading successive blocks from SequentialBatchLoader
    // result in sorted strings
    bl.loadBlock(bp, 0, 3);
    bl.loadBlock(bp, 1, 3);
    bl.loadBlock(bp, 2, 3);

    vector<string> words = buffer_to_vector_of_strings(*bp[0]);

    ASSERT_EQ(sorted(words), true);
}

TEST(shuffled_batch_iterator, shuffled_block) {
    ShuffledBatchIterator sbi(
        make_shared<MockBatchLoader>(), 3, 0
    );
    // buffer_in* dataBuffer = new buffer_in(0);
    // buffer_in* targetBuffer = new buffer_in(0);
    // buffer_in_array bp{dataBuffer, targetBuffer};
    buffer_in_array bp(vector<size_t>{0, 0});


    // ensure that loading successive blocks from SequentialBatchLoader
    // result in sorted strings
    for(int i = 0; i < 26; ++i) {
        sbi.read(bp);
    }

    vector<string> words = buffer_to_vector_of_strings(*bp[0]);

    ASSERT_EQ(sorted(words), false);

    // now sort the words and make sure they are all unique.  We should
    // have loaded an entire 'epoch' and have no duplicates

    assert_vector_unique(words);
}
