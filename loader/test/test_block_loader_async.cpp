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
#include "async_manager.hpp"
#include "manifest_csv.hpp"
#include "block_loader_file_async.hpp"
#include "batch_iterator_async.hpp"
#include "csv_manifest_maker.hpp"
#include "file_util.hpp"
#include "log.hpp"


using namespace std;
using namespace nervana;

TEST(block_loader_async, load_block)
{
    manifest_maker mm;

    // load one block of size 2
    uint32_t num_items       = 20;
    uint32_t block_size      = 2;
    uint32_t object_size     = 16;
    uint32_t target_size     = 16;
    // float    subset_fraction = 1.0;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    manifest_csv mfs(mm.tmp_manifest_file(num_items, {object_size, target_size}),
                            false);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of buffer_variable_size_elements
    block_loader_file_async blf(&mfs, block_size);

    auto num_blocks = num_items / block_size;

    for (int block = 0; block < num_blocks; ++block)
    {
        auto blk = *blf.next();

        for (int item = 0; item < block_size; ++item)
        {
            uint* object_data = (uint*)blk[0].get_item(item).data();
            uint* target_data = (uint*)blk[1].get_item(item).data();
            for (int offset = 0; offset < object_size / sizeof(uint); offset++)
            {
                EXPECT_EQ(object_data[offset] + 1, target_data[offset]);
            }
        }
    }
}


TEST(block_loader_async, iterate_batch)
{
    manifest_maker mm;

    // load one block of size 2
    uint32_t num_items       = 100;
    uint32_t block_size      = 10;
    uint32_t batch_size      = 4;
    uint32_t object_size     = 16;
    uint32_t target_size     = 16;
    // float    subset_fraction = 1.0;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    manifest_csv mfs(mm.tmp_manifest_file(num_items, {object_size, target_size}), false);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of buffer_variable_size_elements
    block_loader_file_async blf(&mfs, block_size);
    batch_iterator_async biter(&blf, batch_size);

    auto num_batches = num_items / batch_size;

    for (int batch = 0; batch < num_batches; ++batch)
    {
        auto b = *biter.next();

        for (int item = 0; item < batch_size; ++item)
        {
            uint* object_data = (uint*)b[0].get_item(item).data();
            uint* target_data = (uint*)b[1].get_item(item).data();
            for (int offset = 0; offset < object_size / sizeof(uint); offset++)
            {
                EXPECT_EQ(object_data[offset] + 1, target_data[offset]);
            }
        }
    }
}

