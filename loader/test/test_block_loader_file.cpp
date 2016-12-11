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
#include "block_loader_file_async.hpp"
#include "csv_manifest_maker.hpp"
#include "file_util.hpp"

using namespace std;
using namespace nervana;

TEST(block_loader_file, constructor)
{
    manifest_maker    mm;
    string            tmpname = mm.tmp_manifest_file(0, {0, 0});
    block_loader_file blf(make_shared<nervana::manifest_csv>(tmpname, true), 1.0, 4);
}

TEST(block_loader_file, load_block)
{
    manifest_maker mm;
    // load one block of size 2
    uint32_t block_size      = 2;
    uint32_t object_size     = 16;
    uint32_t target_size     = 16;
    float    subset_fraction = 1.0;

    auto manifest_file = mm.tmp_manifest_file(4, {object_size, target_size});
    auto manifest = make_shared<manifest_csv>(manifest_file, true);
    block_loader_file_async blf(manifest, subset_fraction, block_size);

    buffer_in_array bp(2);

    blf.load_block(bp, 0);

    // the object_data and target_data should be full of repeating
    // uints.  the uints in target_data will be 1 bigger than the uints
    // in object_data.  Make sure that this is the case here.
    for (int block = 0; block < block_size; block++)
    {
        uint* object_data = (uint*)bp[0]->get_item(block).data();
        uint* target_data = (uint*)bp[1]->get_item(block).data();
        for (int offset = 0; offset < object_size / sizeof(uint); offset++)
        {
            EXPECT_EQ(object_data[offset] + 1, target_data[offset]);
        }
    }
}

TEST(block_loader_file, subset_fraction)
{
    // a 10 object manifest iterated through blocks sized 4 with
    // percentSubset 50 should result in an output block size of 2, 2
    // and then 1.
    manifest_maker mm;
    uint32_t       block_size      = 4;
    uint32_t       object_size     = 4;
    uint32_t       target_size     = 4;
    float          subset_fraction = 0.01;
    size_t         total_records   = 1000;

    block_loader_file blf(make_shared<nervana::manifest_csv>(mm.tmp_manifest_file(total_records, {object_size, target_size}), true),
                          subset_fraction, block_size);

    EXPECT_EQ(blf.record_count(), size_t(total_records * subset_fraction));

    buffer_in_array bp(2);

    blf.load_block(bp, 0);
    ASSERT_EQ(bp[0]->record_count(), 4);
    bp[0]->reset();

    blf.load_block(bp, 1);
    ASSERT_EQ(bp[0]->record_count(), 4);
    bp[0]->reset();

    blf.load_block(bp, 2);
    ASSERT_EQ(bp[0]->record_count(), 2);
    bp[0]->reset();
}

TEST(block_loader_file, exception)
{
    manifest_maker mm;
    float          subset_fraction = 1.0;

    block_loader_file blf(make_shared<nervana::manifest_csv>(mm.tmp_manifest_file_with_invalid_filename(), false), subset_fraction,
                          1);

    buffer_in_array bp(2);

    // loadBlock doesn't actually raise the exception
    blf.load_block(bp, 0);

    // Could not find file exception raised when we try to access the item
    try
    {
        bp[0]->get_item(0);
        FAIL();
    }
    catch (std::exception& e)
    {
        ASSERT_EQ(string("Could not find "), string(e.what()).substr(0, 15));
    }
}

// TEST(block_loader_file, subset_record_count)
//{
//    manifest_maker mm;
//    float subset_fraction = 0.5;
//    block_loader_file blf(
//        make_shared<nervana::manifest_csv>(mm.tmp_manifest_file(13, {16, 16}), false),
//        subset_fraction,
//        5
//    );

//    ASSERT_EQ(blf.record_count(), 2 + 2 + 1);
//}

TEST(block_loader_file, performance)
{
    manifest_maker mm;
    uint32_t       block_size      = 50;
    uint32_t       object_size     = 4;
    uint32_t       target_size     = 4;
    float          subset_fraction = 1.0;
    string         cache_id        = block_loader_random::randomString();
    string         version         = "version123";

    auto blf = make_shared<block_loader_file>(
        make_shared<nervana::manifest_csv>(mm.tmp_manifest_file(1000, {object_size, target_size}), true), subset_fraction,
        block_size);

    string                        cache_dir = file_util::make_temp_directory();
    chrono::high_resolution_clock timer;
    auto                          cache = make_shared<block_loader_cpio_cache>(cache_dir, cache_id, version, blf);
    block_iterator_shuffled       iter(cache);

    auto startTime = timer.now();
    for (int i = 0; i < 30; i++)
    {
        buffer_in_array dest(2);
        iter.read(dest);
    }
    auto endTime = timer.now();
    cout << "time " << (chrono::duration_cast<chrono::milliseconds>(endTime - startTime)).count() << " ms" << endl;
    file_util::remove_directory(cache_dir);
}
