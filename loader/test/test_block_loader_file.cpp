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
#include "block_loader_file.hpp"
#include "csv_manifest_maker.hpp"

using namespace std;
using namespace nervana;

TEST(blocked_file_loader, constructor)
{
    manifest_maker mm;
    string tmpname = mm.tmp_manifest_file(0, {0, 0});
    block_loader_file blf(make_shared<nervana::manifest_csv>(tmpname, true), 1.0, 4);
}

TEST(blocked_file_loader, loadBlock)
{
    manifest_maker mm;
    // load one block of size 2
    uint32_t block_size = 2;
    uint32_t object_size = 16;
    uint32_t target_size = 16;
    float subset_fraction = 1.0;

    block_loader_file blf(
        make_shared<nervana::manifest_csv>(mm.tmp_manifest_file(4, {object_size, target_size}), true),
        subset_fraction,
        block_size
    );

    buffer_in_array bp(2);

    blf.loadBlock(bp, 0);

    // the object_data and target_data should be full of repeating
    // uints.  the uints in target_data will be 1 bigger than the uints
    // in object_data.  Make sure that this is the case here.
    for(int block=0; block<block_size; block++) {
        uint* object_data = (uint*)bp[0]->get_item(block).data();
        uint* target_data = (uint*)bp[1]->get_item(block).data();
        for(int offset=0; offset<object_size / sizeof(uint); offset++) {
            ASSERT_EQ(object_data[offset] + 1, target_data[offset]);
        }
    }
}

TEST(blocked_file_loader, subset_fraction)
{
    // a 10 object manifest iterated through blocks sized 4 with
    // percentSubset 50 should result in an output block size of 2, 2
    // and then 1.
    manifest_maker mm;
    uint32_t block_size = 4;
    uint32_t object_size = 16;
    uint32_t target_size = 16;
    float subset_fraction = 0.01;

    block_loader_file blf(
        make_shared<nervana::manifest_csv>(mm.tmp_manifest_file(10000, {object_size, target_size}), true),
        subset_fraction,
        block_size
    );

    buffer_in_array bp(2);


    blf.loadBlock(bp, 0);
    ASSERT_EQ(bp[0]->get_item_count(), block_size);
    bp[0]->reset();

    blf.loadBlock(bp, 1);
    ASSERT_EQ(bp[0]->get_item_count(), block_size);
    bp[0]->reset();

    blf.loadBlock(bp, 2);
    ASSERT_EQ(bp[0]->get_item_count(), block_size);
    bp[0]->reset();
}

TEST(blocked_file_loader, exception)
{
    manifest_maker mm;
    float subset_fraction = 1.0;

    block_loader_file blf(
        make_shared<nervana::manifest_csv>(mm.tmp_manifest_file_with_invalid_filename(), false),
        subset_fraction,
        1
    );

    buffer_in_array bp(2);

    // loadBlock doesn't actually raise the exception
    blf.loadBlock(bp, 0);

    // Could not find file exception raised when we try to access the item
    try {
        bp[0]->get_item(0);
        FAIL();
    } catch (std::exception& e) {
        ASSERT_EQ(string("Could not find "), string(e.what()).substr(0, 15));
    }
}

//TEST(blocked_file_loader, subset_object_count)
//{
//    manifest_maker mm;
//    float subset_fraction = 0.5;
//    block_loader_file blf(
//        make_shared<nervana::manifest_csv>(mm.tmp_manifest_file(13, {16, 16}), false),
//        subset_fraction,
//        5
//    );

//    ASSERT_EQ(blf.objectCount(), 2 + 2 + 1);
//}
