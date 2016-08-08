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
#include "batch_iterator_minibatch.hpp"
#include "batch_iterator_sequential.hpp"
#include "batch_iterator_shuffled.hpp"

#include "mock_batch_loader.hpp"


using namespace std;

TEST(minibatch_iterator, simple) {
    // give a BatchIteratorMinibatch a block-level BatchIterator and make sure
    // that reading through the BatchIteratorMinibatch results in reading
    // all of the records in the block-level BatchIterator.

    // a little odd here that our block size is 3 and our minibatchsize is 13 ...
    auto mbl = make_shared<MockBatchLoader>(3);
    auto bis = make_shared<BatchIteratorSequential>(mbl);
    BatchIteratorMinibatch mi(bis, 13);

    // we know there are 3 * 26 objects in the dataset so we should get
    // 6 minibatches fo 13 out.  They should all be unique and still in
    // sorted order.

    buffer_in_array bp(vector<size_t>{0, 0});

    // read 6 minibatches
    for(int i = 0; i < 6; ++i) {
        mi.read(bp);
    }

    vector<string> words = buffer_to_vector_of_strings(*bp[0]);

    // for (auto w : words) cout << w << endl;

    ASSERT_EQ(words.size(), mbl->objectCount());
    ASSERT_EQ(sorted(words), true);

    assert_vector_unique(words);
}

TEST(minibatch_iterator, shuffled) {
    // give a BatchIteratorMinibatch a block-level BatchIterator and make sure
    // that reading through the BatchIteratorMinibatch results in reading
    // all of the records in the block-level BatchIterator.

    // a little odd here that our block size is 3 and our minibatchsize is 13 ...
    auto mbl = make_shared<MockBatchLoader>(3);
    auto bis = make_shared<BatchIteratorShuffled>(mbl, 0);
    BatchIteratorMinibatch mi(bis, 13);

    // we know there are 3 * 26 objects in the dataset so we should get
    // 6 minibatches fo 13 out.  They should all be unique and still in
    // sorted order.

    buffer_in_array bp(vector<size_t>{0, 0});

    // read 6 minibatches
    for(int i = 0; i < 6; ++i) {
        mi.read(bp);
    }

    vector<string> words_a = buffer_to_vector_of_strings(*bp[0]);
    vector<string> words_b = buffer_to_vector_of_strings(*bp[1]);

    for (uint i=0; i<words_a.size(); ++i)
        ASSERT_EQ(words_a[i], words_b[i]);

    ASSERT_EQ(words_a.size(), mbl->objectCount());
    ASSERT_EQ(sorted(words_a), false);

    assert_vector_unique(words_a);

}
