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

#include "gtest/gtest.h"
#include "util.hpp"
#include "interface.hpp"

using namespace std;
using namespace nervana;

enum class foo
{
    TEST1,
    TEST2,
    TEST3
};

static void from_string(foo& v, const std::string& s)
{
    string tmp = nervana::to_lower(s);
    if (tmp == "test1")
        v = foo::TEST1;
    else if (tmp == "test2")
        v = foo::TEST2;
    else if (tmp == "test3")
        v = foo::TEST3;
}

TEST(params, parse_enum)
{
    foo            bar;
    nlohmann::json j1 = {{"test", "TEST2"}};

    interface::config::parse_enum(bar, "test", j1);
    EXPECT_EQ(bar, foo::TEST2);
}
