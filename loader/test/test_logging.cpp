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

#include <iostream>
#include <cstdio>

#include "gtest/gtest.h"
#include "log.hpp"

using namespace std;
using namespace nervana;

TEST(logging,conststring)
{
    {
        const char* s = find_last("this/is/a/test",'/');
        EXPECT_STREQ("test",s);
    }
    {
        const char* s = find_last("test",'/');
        EXPECT_STREQ("test",s);
    }
}

TEST(logging,error)
{
    INFO << "This is info";
    WARN << "This is warn";
    ERR << "This is error";
}
