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

#include <vector>

#include "gtest/gtest.h"
#include "file_util.hpp"
#include "log.hpp"
#include "csv_manifest_maker.hpp"
#include "manifest_csv.hpp"
#include "block_loader_file_async.hpp"
#include "block_loader_source_async.hpp"

#define private public

#include "block_manager_async.hpp"

using namespace std;
using namespace nervana;

TEST(block_manager, cache_complete)
{
    string cache_root = file_util::make_temp_directory();

    EXPECT_FALSE(block_manager_async::check_if_complete(cache_root));
    block_manager_async::mark_cache_complete(cache_root);
    EXPECT_TRUE(block_manager_async::check_if_complete(cache_root));

    file_util::remove_directory(cache_root);
}

TEST(block_manager, cache_ownership)
{
    string cache_root = file_util::make_temp_directory();

    int lock;
    EXPECT_TRUE(block_manager_async::take_ownership(cache_root, lock));

    int lock2;
    EXPECT_FALSE(block_manager_async::take_ownership(cache_root, lock2));

    block_manager_async::release_ownership(cache_root, lock);

    EXPECT_TRUE(block_manager_async::take_ownership(cache_root, lock));
    block_manager_async::release_ownership(cache_root, lock);

    file_util::remove_directory(cache_root);
}

TEST(block_manager, cache_busy)
{
    string cache_root = file_util::make_temp_directory();

    manifest_maker mm;

    size_t record_count    = 10;
    size_t block_size      = 4;
    size_t object_size     = 16;
    size_t target_size     = 16;

    auto manifest_file = mm.tmp_manifest_file(record_count, {object_size, target_size});
    manifest_csv manifest(manifest_file, false);

    block_loader_file_async file_reader(&manifest, block_size);
    string cache_name = block_manager_async::create_cache_name(file_reader.get_uid());
    auto cache_dir = file_util::path_join(cache_root, cache_name);

    int lock;
    file_util::make_directory(cache_dir);
    EXPECT_TRUE(block_manager_async::take_ownership(cache_dir, lock));

    EXPECT_THROW(block_manager_async(&file_reader, block_size, cache_root, false), runtime_error);

    block_manager_async::release_ownership(cache_root, lock);

    file_util::remove_directory(cache_root);
}

TEST(block_manager, build_cache)
{
    string cache_root = file_util::make_temp_directory();

    manifest_maker mm;

    size_t record_count    = 12;
    size_t block_size      = 4;
    size_t object_size     = 16;
    size_t target_size     = 16;
    size_t block_count     = record_count / block_size;
    ASSERT_EQ(0, record_count % block_size);
    ASSERT_EQ(0, object_size % sizeof(uint32_t));
    ASSERT_EQ(0, target_size % sizeof(uint32_t));

    auto manifest_file = mm.tmp_manifest_file(record_count, {object_size, target_size});
    manifest_csv manifest(manifest_file, false);

    block_loader_file_async file_reader(&manifest, block_size);
    string cache_name = block_manager_async::create_cache_name(file_reader.get_uid());
    auto cache_dir = file_util::path_join(cache_root, cache_name);

    block_manager_async manager(&file_reader, block_size, cache_root, false);

    size_t record_index = 0;
    for (size_t i=0; i<block_count*2; i++)
    {
        variable_buffer_array* buffer = manager.next();
        ASSERT_NE(nullptr, buffer);
        ASSERT_EQ(2, buffer->size());

        for (size_t record=0; record<block_size; record++)
        {
            auto data0 = (uint32_t*)buffer->at(0).get_item(record).data();
            auto data1 = (uint32_t*)buffer->at(1).get_item(record).data();
            for (size_t offset = 0; offset < object_size / sizeof(uint32_t); offset++)
            {
                EXPECT_EQ(data0[offset] + 1, data1[offset]);
                EXPECT_EQ(data0[offset], record_index * 2);
            }
            record_index = (record_index + 1) % record_count;
        }
    }

    // check that the cache files exist
    string cache_complete = block_manager_async::m_cache_complete_filename;
    string cache_complete_path = file_util::path_join(cache_dir, cache_complete);
    EXPECT_TRUE(file_util::exists(cache_complete_path));
    for (size_t block_number=0; block_number<block_count; block_number++)
    {
        string cache_block_name = manager.create_cache_block_name(block_number);
        string cache_block_path = file_util::path_join(cache_dir, cache_block_name);
        EXPECT_TRUE(file_util::exists(cache_block_path));
    }

    EXPECT_EQ(block_count, manager.m_cache_hit);
    EXPECT_EQ(block_count, manager.m_cache_miss);

    file_util::remove_directory(cache_root);
}

TEST(block_manager, reuse_cache)
{
    string cache_root = file_util::make_temp_directory();

    manifest_maker mm;

    size_t record_count    = 12;
    size_t block_size      = 4;
    size_t object_size     = 16;
    size_t target_size     = 16;
    size_t block_count     = record_count / block_size;
    ASSERT_EQ(0, record_count % block_size);
    ASSERT_EQ(0, object_size % sizeof(uint32_t));
    ASSERT_EQ(0, target_size % sizeof(uint32_t));

    auto manifest_file = mm.tmp_manifest_file(record_count, {object_size, target_size});
    manifest_csv manifest(manifest_file, false);

    // first build the cache
    {
        block_loader_file_async file_reader(&manifest, block_size);

        block_manager_async manager(&file_reader, block_size, cache_root, false);

        size_t record_index = 0;
        for (size_t i=0; i<block_count; i++)
        {
            variable_buffer_array* buffer = manager.next();
            ASSERT_NE(nullptr, buffer);
            ASSERT_EQ(2, buffer->size());

            for (size_t record=0; record<block_size; record++)
            {
                auto data0 = (uint32_t*)buffer->at(0).get_item(record).data();
                auto data1 = (uint32_t*)buffer->at(1).get_item(record).data();
                for (size_t offset = 0; offset < object_size / sizeof(uint32_t); offset++)
                {
                    EXPECT_EQ(data0[offset] + 1, data1[offset]);
                    EXPECT_EQ(data0[offset], record_index * 2);
                }
                record_index = (record_index + 1) % record_count;
            }
        }
        ASSERT_EQ(0, manager.m_cache_hit);
        ASSERT_EQ(block_count, manager.m_cache_miss);
    }

    // now read data with new reader, same manifest
    {
        block_loader_file_async file_reader(&manifest, block_size);

        block_manager_async manager(&file_reader, block_size, cache_root, false);

        size_t record_index = 0;
        for (size_t i=0; i<block_count; i++)
        {
            variable_buffer_array* buffer = manager.next();
            ASSERT_NE(nullptr, buffer);
            ASSERT_EQ(2, buffer->size());

            for (size_t record=0; record<block_size; record++)
            {
                auto data0 = (uint32_t*)buffer->at(0).get_item(record).data();
                auto data1 = (uint32_t*)buffer->at(1).get_item(record).data();
                for (size_t offset = 0; offset < object_size / sizeof(uint32_t); offset++)
                {
                    EXPECT_EQ(data0[offset] + 1, data1[offset]);
                    EXPECT_EQ(data0[offset], record_index * 2);
                }
                record_index = (record_index + 1) % record_count;
            }
        }
        EXPECT_EQ(block_count, manager.m_cache_hit);
        EXPECT_EQ(0, manager.m_cache_miss);
    }

    file_util::remove_directory(cache_root);
}
