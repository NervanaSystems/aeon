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
#include "util.hpp"
#include "wav_data.hpp"

using namespace std;
using namespace nervana;

TEST(util, unpack_le) {
    {
        char data[] = {1,0,0,0};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(0x00000001,actual);
    }
    {
        char data[] = {0,1,0,0};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(0x00000100,actual);
    }
    {
        char data[] = {0,0,0,1};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(0x01000000,actual);
    }
    {
        char data[] = {0,0,0,1};
        int actual = unpack_le<int>(data,0,3);
        EXPECT_EQ(0,actual);
    }
    {
        char data[] = {0,0,0,1};
        int actual = unpack_le<int>(data,1,3);
        EXPECT_EQ(0x00010000,actual);
    }
    {
        char data[] = {(char)0x80,0,0,0};
        int actual = unpack_le<int>(data);
        EXPECT_EQ(128,actual);
    }
}

TEST(util, unpack_be) {
    {
        char data[] = {0,0,0,1};
        int actual = unpack_be<int>(data);
        EXPECT_EQ(0x00000001,actual);
    }
    {
        char data[] = {0,0,1,0};
        int actual = unpack_be<int>(data);
        EXPECT_EQ(0x00000100,actual);
    }
    {
        char data[] = {1,0,0,0};
        int actual = unpack_be<int>(data);
        EXPECT_EQ(0x01000000,actual);
    }
    {
        char data[] = {1,0,0,0};
        int actual = unpack_be<int>(data,0,3);
        EXPECT_EQ(0x00010000,actual);
    }
    {
        char data[] = {1,0,0,0};
        int actual = unpack_be<int>(data,1,3);
        EXPECT_EQ(0,actual);
    }
}


TEST(util, pack_le) {
    {
        char actual[] = {0,0,0,0};
        char expected[] = {1,0,0,0};
        pack_le<int>(actual,1);
        EXPECT_EQ(*(unsigned int*)expected,*(unsigned int*)actual);
    }
    {
        char actual[] = {0,0,0,0};
        char expected[] = {0,1,0,0};
        pack_le<int>(actual,0x00000100);
        EXPECT_EQ(*(unsigned int*)expected,*(unsigned int*)actual);
    }
    {
        char actual[] = {0,0,0,0};
        char expected[] = {0,0,0,1};
        pack_le<int>(actual,0x01000000);
        EXPECT_EQ(*(unsigned int*)expected,*(unsigned int*)actual);
    }
//    {
//        char actual[] = {0,0,0,0};
//        char expected[] = {0,0,0,1};
//        pack_le<int>(actual,0,3);
//        EXPECT_EQ(expected,actual);
//    }
//    {
//        char actual[] = {0,0,0,0};
//        char expected[] = {0,0,0,1};
//        pack_le<int>(actual,0x00010000,1,3);
//        EXPECT_EQ(expected,actual);
//    }
}

TEST(avi,video) {

}
