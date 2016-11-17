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
#include <string>
#include <sstream>
#include <random>

#include "gtest/gtest.h"
#include "cpio.hpp"
#include "buffer_in.hpp"

#define private public

using namespace std;
using namespace nervana;

TEST(cpio, read_nds)
{
    cpio::file_reader reader;

    reader.open(CURDIR "/test_data/test.cpio");
    EXPECT_EQ(1, reader.itemCount());

    nervana::buffer_in buffer;
    EXPECT_EQ(0, buffer.get_item_count());
    reader.read(buffer);
    EXPECT_EQ(1, buffer.get_item_count());
}
