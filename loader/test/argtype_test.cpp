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
using namespace nervana;

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,NAME,...) NAME
#define ADD_ARG(...) GET_MACRO(__VA_ARGS__,F7,F6,F5,F4)(__VA_ARGS__)
#define F7(t,desc,vs,vl,def,minimum,maximum) add(t,#t,desc,vs,vl,(decltype(t))def,(decltype(t))minimum,(decltype(t))maximum)
#define F4(t,desc,vs,vl) add(t,#t,desc,vs,vl)
#define F5(t,desc,vs,vl,def) add(t,#t,desc,vs,vl,(decltype(t))def)

// #define ADD_ARGR(t,desc,vs,vl) add(t,#t,desc,vs,vl)
// #define ADD_ARG(t,desc,vs,vl,def,minimum,maximum) add(t,#t,desc,vs,vl,(decltype(t))def,(decltype(t))minimum,(decltype(t))maximum)

class ParamList1 : public ParameterCollection {
public:
    int int1;
    int int2;
    int int3;
    int int4;
    int int5;
    float float1;
    float float2;
    float float3;
    float float4;
    float float5;
    bool bool1;
    std::string string1;

    ParamList1() {
        ADD_ARG(int1, "description of arg1", "a1", "arg-1", 3, 0, 50);
        ADD_ARG(int2, "description of arg2", "a2", "arg-2", 3, 0, 100);
        ADD_ARG(int3, "description of arg3", "a3", "arg-3");
        ADD_ARG(int4, "description of arg4", "a4", "arg-4");
        ADD_ARG(int5, "description of arg5", "a5", "arg-5", -50, -100, -10);

        ADD_ARG(float1, "description of arg1", "f1", "float-1", 3, 0, 50);
        ADD_ARG(float2, "description of arg2", "f2", "float-2", 3, 0, 100);
        ADD_ARG(float3, "description of arg3", "f3", "float-3", 3, 0, 20);
        ADD_ARG(float4, "description of arg4", "f4", "float-4", 3, 0, 100);
        ADD_ARG(float5, "description of arg5", "f5", "float-5", -50, -100, -10);

        ADD_ARG(bool1, "description of bool1", "b1", "bool-1", false);

        ADD_ARG(string1, "description of string1", "s1", "string-1","");
    }
};

static ParamList1 _ParamList1;

TEST(loader,argtype) {
    map<string,shared_ptr<interface_ArgType> > args = _ParamList1.get_args();
    ASSERT_EQ(12, args.size());

    {
        string argString = "-a1 5";
        parsed_args parsed;
        EXPECT_FALSE(_ParamList1.parse(argString,parsed)) << "**** failed to detect missing required arguments";
    }
    {
        string argString = "-a1 5 -a3 10 -a4 20";
        parsed_args parsed;
        EXPECT_TRUE(_ParamList1.parse(argString,parsed));
        EXPECT_EQ(5,parsed.value<int>("int1"));
        EXPECT_EQ(3,parsed.value<int>("int2"));
        EXPECT_EQ(10,parsed.value<int>("int3"));
        EXPECT_EQ(20,parsed.value<int>("int4"));
        EXPECT_EQ(-50,parsed.value<int>("int5"));
        EXPECT_EQ(3,parsed.value<float>("float1"));
        EXPECT_EQ(3,parsed.value<float>("float2"));
        EXPECT_EQ(3,parsed.value<float>("float3"));
        EXPECT_EQ(3,parsed.value<float>("float4"));
        EXPECT_EQ(-50,parsed.value<float>("float5"));
        EXPECT_EQ(false,parsed.value<bool>("bool1"));
        EXPECT_STREQ("blah",parsed.value<string>("string1").c_str());
    }
    {
        string argString = "-a1 5 -a3 10 -a4";
        parsed_args parsed;
        EXPECT_FALSE(_ParamList1.parse(argString,parsed)) << "**** missing value for -a4 arg";
    }
    {
        string argString = "-a1 5 -a3 10 -a4 true";
        parsed_args parsed;
        EXPECT_FALSE(_ParamList1.parse(argString,parsed)) << "**** non-int value for -a4";
    }
    {
        string argString = "-a1 5 -a3 10 -a4 20 --arg-4 30";
        parsed_args parsed;
        EXPECT_FALSE(_ParamList1.parse(argString,parsed)) << "**** argument int4 included more than once";
    }

    // Test arg validators
    // arg1
    {
        const interface_ArgType& arg = *args["int1"];
        EXPECT_EQ("3",arg.default_value());
        EXPECT_TRUE(arg.validate("10"));
        EXPECT_TRUE(arg.validate("1000"));
    }

    // arg2
    {
        const interface_ArgType& arg = *args["int2"];
        EXPECT_EQ("3",arg.default_value());
        EXPECT_TRUE(arg.validate("10"));
        EXPECT_TRUE(arg.validate("0"));
        EXPECT_TRUE(arg.validate("99"));
        EXPECT_FALSE(arg.validate("100"));
        EXPECT_FALSE(arg.validate("-1"));
    }

    // arg5
    {
        const interface_ArgType& arg = *args["int5"];
        EXPECT_EQ("-50",arg.default_value());
        EXPECT_TRUE(arg.validate("-11"));
        EXPECT_TRUE(arg.validate("-100"));
        EXPECT_TRUE(arg.validate("-50"));
        EXPECT_FALSE(arg.validate("-101"));
        EXPECT_FALSE(arg.validate("0"));
    }

    {
        const interface_ArgType& arg = *args["bool1"];
        EXPECT_EQ("false",arg.default_value());
        EXPECT_TRUE(arg.validate("true"));
        EXPECT_TRUE(arg.validate("false"));
        EXPECT_FALSE(arg.validate("0"));
        EXPECT_FALSE(arg.validate("1"));
    }
}
