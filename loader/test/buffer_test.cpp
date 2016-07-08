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

#include "buffer_in.hpp"
#include "helpers.hpp"

using namespace std;

TEST(buffer, shuffle) {
    // create a buffer with lots of words in sorted order.  assert
    // that they are sorted, then shuffle, then assert that they are
    // not sorted

    buffer_in b(0);

    b.read("abc", 3);
    b.read("asd", 3);
    b.read("hello", 5);
    b.read("qwe", 3);
    b.read("world", 5);
    b.read("xyz", 3);
    b.read("yuiop", 5);
    b.read("zxcvb", 5);

    ASSERT_EQ(sorted(buffer_to_vector_of_strings(b)), true);

    b.shuffle(0);

    ASSERT_EQ(sorted(buffer_to_vector_of_strings(b)), false);
}
