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

class ParamList1 : public parameter_collection {
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
        ADD_OPTIONAL(int1, "description of arg1", "i1", "int-1", 3, 0, 50);
        ADD_OPTIONAL(int2, "description of arg2", "i2", "int-2", 3, 0, 100);
        ADD_REQUIRED(int3, "description of arg3", "i3", "int-3");
        ADD_REQUIRED(int4, "description of arg4", "i4", "int-4", 10, 40);
        ADD_OPTIONAL(int5, "description of arg5", "i5", "int-5", -50, -100, -10);

        ADD_OPTIONAL(float1, "description of arg1", "f1", "float-1", 3, 0, 50);
        ADD_OPTIONAL(float2, "description of arg2", "f2", "float-2", 3, 0, 100);
        ADD_OPTIONAL(float3, "description of arg3", "f3", "float-3", 3, 0, 20);
        ADD_OPTIONAL(float4, "description of arg4", "f4", "float-4", 3, 0, 100);
        ADD_OPTIONAL(float5, "description of arg5", "f5", "float-5", -50, -100, -10);

        ADD_OPTIONAL(bool1, "description of bool1", "b1", "bool-1", false);

        ADD_OPTIONAL(string1, "description of string1", "s1", "string-1","blah");
    }
};

static ParamList1 _ParamList1;

TEST(loader,argtype) {
    map<string,shared_ptr<interface_ArgType> > args = _ParamList1.get_args();
    ASSERT_EQ(12, args.size());

    string help = _ParamList1.help();
    cout << help << endl;

    {
        string argString = "-i1 5";
        EXPECT_FALSE(_ParamList1.parse(argString)) << "**** failed to detect missing required arguments in '" << argString << "'";
    }
    {
        string argString = "-i3 5 -i4 40";
        EXPECT_FALSE(_ParamList1.parse(argString)) << "**** argument out-of-range '" << argString << "'";
    }
    {
        string argString = "-i1 5 -i3 10 --int-4 30";
        EXPECT_TRUE(_ParamList1.parse(argString)) << argString;
        EXPECT_EQ(5,_ParamList1.int1);
        EXPECT_EQ(3,_ParamList1.int2);
        EXPECT_EQ(10,_ParamList1.int3);
        EXPECT_EQ(30,_ParamList1.int4);
        EXPECT_EQ(-50,_ParamList1.int5);
        EXPECT_EQ(3,_ParamList1.float1);
        EXPECT_EQ(3,_ParamList1.float2);
        EXPECT_EQ(3,_ParamList1.float3);
        EXPECT_EQ(3,_ParamList1.float4);
        EXPECT_EQ(-50,_ParamList1.float5);
        EXPECT_EQ(false,_ParamList1.bool1);
        EXPECT_STREQ("blah",_ParamList1.string1.c_str());
    }
    {
        string argString = "-i1 5 -i3 10 -i4";
        EXPECT_FALSE(_ParamList1.parse(argString)) << "**** missing value for -a4 arg in '" << argString << "'";
    }
    {
        string argString = "-i1 5 -i3 10 -i4 true";
        EXPECT_FALSE(_ParamList1.parse(argString)) << "**** non-int value for -a4 in '" << argString << "'";
    }
    {
        string argString = "-i1 5 -i3 10 -i4 20 --int-4 30";
        EXPECT_FALSE(_ParamList1.parse(argString)) << "**** argument int4 included more than once in '" << argString << "'";
    }

    // Test arg validators
    // arg1
    {
        const interface_ArgType& arg = *args["int1"];
        EXPECT_EQ("3",arg.default_value());
        EXPECT_TRUE(arg.validate("10"));
        EXPECT_FALSE(arg.validate("1000"));
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
