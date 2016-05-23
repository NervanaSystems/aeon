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

#include "gtest/gtest.h"
#include "argtype.hpp"

using namespace std;

class ParamList1 : public ParameterCollection {
public:
    const ArgType_int arg1{*this, "arg1", "description of arg1", "a1", "arg-1", false, 3};
    const ArgType_int arg2{*this, "arg2", "description of arg2", "a2", "arg-2", false, 3, 0, 100};
    const ArgType_int arg3{*this, "arg3", "description of arg3", "a3", "arg-3", true, 3};
    const ArgType_int arg4{*this, "arg4", "description of arg4", "a4", "arg-4", true, 3, 0, 100};
    const ArgType_int arg5{*this, "arg5", "description of arg5", "a5", "arg-5", false, -50, -100, -10};
};

static ParamList1 _ParamList1;

TEST(loader,argtype) {
    vector<const ArgType*> args = _ParamList1.get_args();
    ASSERT_EQ(5, args.size());

    {
        string argString = "-a1 5 -a3 10 -a4 20";
        EXPECT_TRUE(_ParamList1.parse(argString));
    }

    // arg1
    EXPECT_EQ(5,args.size()) << "ParamList1";
    {
        const ArgType& arg = *args[0];
        EXPECT_EQ("3",arg.default_value());
        EXPECT_TRUE(arg.validate("10"));
        EXPECT_TRUE(arg.validate("1000"));
    }

    // arg2
    EXPECT_EQ(5,args.size()) << "ParamList1";
    {
        const ArgType& arg = *args[1];
        EXPECT_EQ("3",arg.default_value());
        EXPECT_TRUE(arg.validate("10"));
        EXPECT_TRUE(arg.validate("0"));
        EXPECT_TRUE(arg.validate("99"));
        EXPECT_FALSE(arg.validate("100"));
        EXPECT_FALSE(arg.validate("-1"));
    }

    // arg5
    EXPECT_EQ(5,args.size()) << "ParamList1";
    {
        const ArgType& arg = *args[4];
        EXPECT_EQ("-50",arg.default_value());
        EXPECT_TRUE(arg.validate("-11"));
        EXPECT_TRUE(arg.validate("-100"));
        EXPECT_TRUE(arg.validate("-50"));
        EXPECT_FALSE(arg.validate("-101"));
        EXPECT_FALSE(arg.validate("0"));
    }}
