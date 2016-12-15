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

#include "buffer_batch.hpp"
#include "helpers.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

void read(encoded_record_list& b, const string& str)
{
    encoded_record record;
    record.add_element(string2vector(str));
    b.add_record(record);
}

TEST(buffer, shuffle)
{
    // create a buffer with lots of words in sorted order.  assert
    // that they are sorted, then shuffle, then assert that they are
    // not sorted

    encoded_record_list b;

    read(b, "abc");
    read(b, "asd");
    read(b, "hello");
    read(b, "qwe");
    read(b, "world");
    read(b, "xyz");
    read(b, "yuiop");
    read(b, "zxcvb");

    ASSERT_EQ(sorted(buffer_to_vector_of_strings(b)), true);

    b.shuffle(0);

    ASSERT_EQ(sorted(buffer_to_vector_of_strings(b)), false);
}

void setup_buffer_exception(encoded_record_list& b)
{
    // setup b with length 4, one value is an exception
    read(b, "a");

    try
    {
        throw std::runtime_error("expect me");
    }
    catch (std::exception& e)
    {
        encoded_record record;
        record.add_exception(std::current_exception());
        b.add_record(record);
    }

    read(b, "c");
    read(b, "d");
}

TEST(buffer, write_exception)
{
    encoded_record_list b;

    setup_buffer_exception(b);
}

TEST(buffer, read_exception)
{
    encoded_record_list b;

    setup_buffer_exception(b);

    // no exceptions if we hit index 0, 2 and 3
    ASSERT_EQ(b.record(0).element(0)[0], 'a');
    ASSERT_EQ(b.record(2).element(0)[0], 'c');
    ASSERT_EQ(b.record(3).element(0)[0], 'd');

    // assert that exception is raised
    try
    {
        b.record(1).element(0);
        FAIL();
    }
    catch (std::exception& e)
    {
        ASSERT_STREQ("expect me", e.what());
    }
}
