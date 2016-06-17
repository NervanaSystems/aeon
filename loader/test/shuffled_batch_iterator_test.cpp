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

using namespace std;

class SequentialBatchLoader : public BatchLoader {
public:
    void loadBlock(BufferPair &dest, uint block_num, uint block_size) {
        // load BufferPair with strings.
        // block_num 0: 'aa', 'ab', 'ac'
        // block_num 1: 'ba', 'bb', 'bc'
        // ...
        assert(block_size == 3);
        for(uint i = 0; i < block_size; ++i) {
            stringstream ss;
            ss << (char)('a' + block_num);
            ss << (char)('a' + i);
            string s = ss.str();

            dest.first->read(s.c_str(), s.length());
            dest.second->read(s.c_str(), s.length());
        }
    };

    uint objectCount() {
        return 26 * 3;
    }
};

TEST(shuffled_batch_iterator, sequential_batch_loader) {
    SequentialBatchLoader bl;

    Buffer* dataBuffer = new Buffer(0);
    Buffer* targetBuffer = new Buffer(0);
    BufferPair bp = make_pair(dataBuffer, targetBuffer);

    // ensure that loading successive blocks from SequentialBatchLoader
    // result in sorted strings
    bl.loadBlock(bp, 0, 3);
    bl.loadBlock(bp, 1, 3);
    bl.loadBlock(bp, 2, 3);

    vector<string> words = buffer_to_vector_of_strings(*bp.first);

    ASSERT_EQ(sorted(words), true);
}

TEST(shuffled_batch_iterator, shuffled_block) {
    ShuffledBatchIterator sbi(
        make_shared<SequentialBatchLoader>(),
        3,
        0
    );

    Buffer* dataBuffer = new Buffer(0);
    Buffer* targetBuffer = new Buffer(0);
    BufferPair bp = make_pair(dataBuffer, targetBuffer);

    // ensure that loading successive blocks from SequentialBatchLoader
    // result in sorted strings
    sbi.read(bp);
    sbi.read(bp);
    sbi.read(bp);

    vector<string> words = buffer_to_vector_of_strings(*bp.first);

    ASSERT_EQ(sorted(words), false);
}
