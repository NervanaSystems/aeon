/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gtest/gtest.h"
#include "async_manager.hpp"
#include "manifest_file.hpp"
#include "block_loader_file.hpp"
#include "batch_iterator.hpp"
#include "block_manager.hpp"
#include "manifest_builder.hpp"
#include "file_util.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

TEST(block_loader_file, file_block)
{
    manifest_builder mb;

    // load one block of size 2
    size_t record_count = 20;
    size_t block_size   = 2;
    size_t object_size  = 16;
    size_t target_size  = 16;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    stringstream& manifest_stream =
        mb.sizes({object_size, target_size}).record_count(record_count).create();
    auto manifest = make_shared<manifest_file>(manifest_stream, false, "", 1.0, block_size);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of encoded_record_list
    block_loader_file loader(manifest, block_size);

    auto block_count = loader.block_count();
    ASSERT_EQ(record_count / block_size, block_count);

    size_t record_number = 0;
    for (size_t block = 0; block < block_count; ++block)
    {
        auto data = *loader.next();

        for (size_t item = 0; item < block_size; ++item)
        {
            const encoded_record& record = data.record(item);
            for (size_t element_number = 0; element_number < record.size(); element_number++)
            {
                stringstream ss;
                ss << record_number << ":" << element_number;
                string expected = ss.str();
                string element  = vector2string(record.element(element_number));
                EXPECT_STREQ(expected.c_str(), element.c_str());
            }
            record_number++;
        }
    }
}

TEST(block_loader_file, file_block_odd)
{
    manifest_builder mb;

    // load one block of size 2
    size_t record_count = 3;
    size_t block_size   = 2;
    size_t object_size  = 16;
    size_t target_size  = 16;

    // each call to next() will yield pointer to vector<string> (filename list per record)
    stringstream& manifest_stream =
        mb.sizes({object_size, target_size}).record_count(record_count).create();
    auto manifest = make_shared<manifest_file>(manifest_stream, false, "", 1.0, block_size);

    // each call to next() will yield pointer to variable buffer_array
    //   which is vector of encoded_record_list
    block_loader_file loader(manifest, block_size);

    auto block_count = ceil((float)record_count / (float)block_size);
    ASSERT_EQ(2, block_count);

    {
        encoded_record_list& data = *loader.next();
        ASSERT_EQ(2, data.size());
        ASSERT_EQ(2, data.elements_per_record());
    }

    {
        encoded_record_list& data = *loader.next();
        ASSERT_EQ(1, data.size());
        ASSERT_EQ(2, data.elements_per_record());
    }
}
