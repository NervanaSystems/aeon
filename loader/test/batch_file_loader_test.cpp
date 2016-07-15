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
#include "batch_file_loader.hpp"
#include "manifest_maker.hpp"

using namespace std;

TEST(blocked_file_loader, constructor) {
    string tmpname = tmp_manifest_file(0, {0, 0});
    BatchFileLoader bfl(make_shared<Manifest>(tmpname, true), 100);
}

TEST(blocked_file_loader, loadBlock) {
    // load one block of size 2
    uint block_size = 2;
    uint object_size = 16;
    uint target_size = 16;

    BatchFileLoader bfl(make_shared<Manifest>(tmp_manifest_file(4, {object_size, target_size}), true), 100);

    buffer_in_array bp(vector<size_t>{0, 0});

    bfl.loadBlock(bp, 0, block_size);

    // the object_data and target_data should be full of repeating
    // uints.  the uints in target_data will be 1 bigger than the uints
    // in object_data.  Make sure that this is the case here.
    for(int block=0; block<block_size; block++) {
        uint* object_data = (uint*)bp[0]->getItem(block).data();
        uint* target_data = (uint*)bp[1]->getItem(block).data();
        for(int offset=0; offset<object_size / sizeof(uint); offset++) {
            ASSERT_EQ(object_data[offset] + 1, target_data[offset]);
        }
    }
}

TEST(blocked_file_loader, subsetPercent) {
    // a 10 object manifest iterated through blocks sized 4 with
    // percentSubset 50 should result in an output block size of 2, 2
    // and then 1.
    uint block_size = 4;
    uint object_size = 16;
    uint target_size = 16;

    BatchFileLoader bfl(make_shared<Manifest>(tmp_manifest_file(10, {object_size, target_size}), true), 50);

    buffer_in_array bp(vector<size_t>{0, 0});


    bfl.loadBlock(bp, 0, block_size);
    ASSERT_EQ(bp[0]->getItemCount(), block_size / 2);
    bp[0]->reset();

    bfl.loadBlock(bp, 1, block_size);
    ASSERT_EQ(bp[0]->getItemCount(), block_size / 2);
    bp[0]->reset();

    bfl.loadBlock(bp, 2, block_size);
    ASSERT_EQ(bp[0]->getItemCount(), 1);
    bp[0]->reset();
}

TEST(blocked_file_loader, exception) {
    BatchFileLoader bfl(
        make_shared<Manifest>(
            tmp_manifest_file_with_invalid_filename(), false
        ), 100
    );

    buffer_in_array bp(vector<size_t>{0, 0});

    // loadBlock doesn't actually raise the exception
    bfl.loadBlock(bp, 0, 1);

    // Could not find file exception raised when we try to access the item
    try {
        bp[0]->getItem(0);
        FAIL();
    } catch (std::exception& e) {
        ASSERT_EQ(string("Could not find "), string(e.what()).substr(0, 15));
    }
}
