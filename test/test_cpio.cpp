/*
 Copyright 2016 Intel(R) Nervana(TM)
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

// TODO put a better test here later
TEST(cpio, read_canonical)
{
    ifstream f(string(CURDIR) + "/test_data/test.cpio", istream::binary);
    ASSERT_TRUE(f);
    cpio::reader reader(f);
    EXPECT_EQ(1, reader.record_count());

    encoded_record_list buffer;
    reader.read(buffer, 1);
    EXPECT_EQ(1, buffer.size());
}

TEST(cpio, write_string)
{
    int          record_count = 10;
    stringstream ss;
    {
        vector<char>        image_data(32);
        vector<char>        label_data(4);
        cpio::writer        writer(ss);
        encoded_record_list bin;
        for (int i = 0; i < record_count; i++)
        {
            encoded_record record;
            record.add_element(image_data);
            record.add_element(label_data);
            bin.add_record(record);
        }
        writer.write_all_records(bin);
    }
    cpio::reader reader(ss);
    EXPECT_EQ(record_count, reader.record_count());
}
