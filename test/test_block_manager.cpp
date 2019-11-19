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

#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "file_util.hpp"
#include "manifest_builder.hpp"
#include "manifest_file.hpp"
#include "block_loader_file.hpp"
#include "block.hpp"

#define private public
#include "block_manager.hpp"

using namespace std;
using namespace nervana;

TEST(block_manager, block_list)
{
    {
        vector<block_info> seq = generate_block_list(1003, 335);
        ASSERT_EQ(3, seq.size());

        EXPECT_EQ(0, seq[0].start());
        EXPECT_EQ(335, seq[0].count());

        EXPECT_EQ(335, seq[1].start());
        EXPECT_EQ(335, seq[1].count());

        EXPECT_EQ(670, seq[2].start());
        EXPECT_EQ(333, seq[2].count());
    }
    {
        vector<block_info> seq = generate_block_list(20, 5000);
        ASSERT_EQ(1, seq.size());

        EXPECT_EQ(0, seq[0].start());
        EXPECT_EQ(20, seq[0].count());
    }
}
