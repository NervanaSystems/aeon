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
#include "util.hpp"
#include "buffer_batch.hpp"

#define private public

using namespace std;
using namespace nervana;

TEST(cpio, read_nds)
{

    ifstream f(CURDIR "/test_data/test.cpio", istream::binary);
    ASSERT_TRUE(f);
    cpio::reader reader(f);
    EXPECT_EQ(1, reader.record_count());

    nervana::buffer_variable_size_elements buffer;
    EXPECT_EQ(0, buffer.get_item_count());
    reader.read(buffer);
    EXPECT_EQ(1, buffer.get_item_count());
}

TEST(cpio,write_string)
{
    int record_count = 10;
    stringstream ss;
    {
        vector<char> image_data(32);
        vector<char> label_data(4);
        cpio::writer writer(ss);
        for (int i=0; i<record_count; i++)
        {
            variable_buffer_array bin{2};
            bin[0].add_item(image_data);
            bin[1].add_item(label_data);
            writer.write_all_records(bin);
        }
    }
    cpio::reader reader(ss);
    EXPECT_EQ(record_count, reader.record_count());
}
