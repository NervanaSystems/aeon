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

TEST(block_loader_async, file_block)
{
    manifest_maker mm;

    // load one block of size 2
    size_t record_count    = 20;
    size_t block_size      = 2;
    size_t object_size     = 16;
    size_t target_size     = 16;
    // float    subset_fraction = 1.0;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    auto manifest_file = mm.tmp_manifest_file(record_count, {object_size, target_size});
    manifest_csv manifest(manifest_file, false);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of buffer_variable_size_elements
    block_loader_file_async loader(&manifest, block_size);

    auto block_count = record_count / block_size;

    for (int block = 0; block < block_count; ++block)
    {
        auto data = *loader.next();

        for (int item = 0; item < block_size; ++item)
        {
            uint* object_data = (uint*)data[0].get_item(item).data();
            uint* target_data = (uint*)data[1].get_item(item).data();
            for (int offset = 0; offset < object_size / sizeof(uint); offset++)
            {
                EXPECT_EQ(object_data[offset] + 1, target_data[offset]);
            }
        }
    }
}

TEST(block_loader_async, file_block_odd)
{
    manifest_maker mm;

    // load one block of size 2
    size_t record_count    = 3;
    size_t block_size      = 2;
    size_t object_size     = 16;
    size_t target_size     = 16;
    // float    subset_fraction = 1.0;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    auto manifest_file = mm.tmp_manifest_file(record_count, {object_size, target_size});
    manifest_csv manifest(manifest_file, false);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of buffer_variable_size_elements
    block_loader_file_async loader(&manifest, block_size);

    auto block_count = ceil((float)record_count / (float)block_size);
    ASSERT_EQ(2, block_count);

    {
        variable_buffer_array& data = *loader.next();
        ASSERT_EQ(2, data.size());
        buffer_variable_size_elements& data0 = data[0];
        buffer_variable_size_elements& data1 = data[1];
        ASSERT_EQ(2, data0.size());
        ASSERT_EQ(2, data1.size());
    }

    {
        variable_buffer_array& data = *loader.next();
        ASSERT_EQ(2, data.size());
        buffer_variable_size_elements& data0 = data[0];
        buffer_variable_size_elements& data1 = data[1];
        ASSERT_EQ(1, data0.size());
        ASSERT_EQ(1, data1.size());
    }
}

TEST(block_loader_async, iterate_batch)
{
    manifest_maker mm;

    // load one block of size 2
    size_t record_count    = 100;
    size_t block_size      = 10;
    size_t batch_size      = 4;
    size_t object_size     = 16;
    size_t target_size     = 16;
    // float    subset_fraction = 1.0;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    auto manifest_file = mm.tmp_manifest_file(record_count, {object_size, target_size});
    manifest_csv manifest(manifest_file, false);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of buffer_variable_size_elements
    block_loader_file_async blf(&manifest, block_size);
    block_manager_async block_manager(&blf, block_size, "", false);
    batch_iterator_async biter(&block_manager, batch_size);

    auto batch_count = record_count / batch_size;

    for (int batch = 0; batch < batch_count; ++batch)
    {
        auto b1 = biter.next();
        ASSERT_NE(nullptr, b1);
        auto b = *b1;
        for (int item = 0; item < batch_size; ++item)
        {
            uint* object_data = (uint*)b[0].get_item(item).data();
            uint* target_data = (uint*)b[1].get_item(item).data();
            for (int offset = 0; offset < object_size / sizeof(uint); offset++)
            {
                EXPECT_EQ(object_data[offset] + 1, target_data[offset]);
                // INFO << "batch_iter " << object_data[offset] << " " << offset << ", " << item <<  ", " << batch << ", " << batch_size;
                EXPECT_EQ(object_data[offset], 2 * (batch * batch_size + item));
            }
        }
    }
}

