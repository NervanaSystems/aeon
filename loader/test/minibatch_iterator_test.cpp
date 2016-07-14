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
#include "minibatch_iterator.hpp"
#include "mock_batch_loader.hpp"
#include "sequential_batch_iterator.hpp"

using namespace std;

TEST(minibatch_iterator, simple) {
    // give a MinibatchIterator a MacroBatchIterator and make sure
    // that reading through the MinibatchIterator results in reading
    // all of the records in the MacroBatchIterator.

    // a little odd here that our block size is 3 and our minibatchsize is 13 ...
    MinibatchIterator mi(
        make_shared<SequentialBatchIterator>(make_shared<MockBatchLoader>(), 3),
        13
    );

    // we know there are 3 * 26 objects in the dataset so we should get
    // 6 minibatches fo 13 out.  They should all be unique and still in
    // sorted order.

    // buffer_in* dataBuffer = new buffer_in(0);
    // buffer_in* targetBuffer = new buffer_in(0);
    // buffer_in_array bp{dataBuffer, targetBuffer};
    buffer_in_array bp(vector<uint32_t>{0, 0});


    // read 6 minibatches
    for(int i = 0; i < 6; ++i) {
        mi.read(bp);
    }

    vector<string> words = buffer_to_vector_of_strings(*bp[0]);

    ASSERT_EQ(sorted(words), true);

    assert_vector_unique(words);
}
