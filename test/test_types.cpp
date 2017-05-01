/*
 Copyright 2016-2017 Nervana Systems Inc.
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
#include "typemap.hpp"
#include "json.hpp"
#include <typeinfo>
#include <typeindex>

using namespace std;
using namespace nervana;

TEST(typemap, numpy)
{
    {
        output_type opt{"int8_t"};
        EXPECT_EQ(CV_8S, opt.get_cv_type());
    }
}

TEST(typemap, otype_serialize)
{
    {
        output_type    opt{"int8_t"};
        nlohmann::json js = opt;

        std::stringstream is;
        is << js;

        // Check that the serialized representation matches expectation
        std::string ref = R"({"cv_type":1,"name":"int8_t","np_type":1,"size":1})";
        EXPECT_EQ(ref, is.str());

        // Check that we can deserialize and achieve equality
        output_type opt2 = js.get<output_type>();
        EXPECT_EQ(opt, opt2);
    }
}

TEST(typemap, otype_equal)
{
    {
        output_type opt1{"int32_t"};
        output_type opt2{"int32_t"};
        EXPECT_EQ(opt1, opt2);
    }
}

TEST(typemap, shapetype_equal)
{
    {
        shape_type s1{{1, 2, 3}, {"int32_t"}};
        shape_type s2{{1, 2, 3}, {"int32_t"}};
        EXPECT_EQ(s1, s2);

        std::vector<std::string> axis_names{"dim1", "dim2", "dim3"};
        s1.set_names(axis_names);
        s2.set_names(axis_names);
        EXPECT_EQ(s1, s2);
    }
}

TEST(typemap, shapetype_not_equal)
{
    {
        shape_type s1{{1, 2, 3}, {"int32_t"}};
        shape_type s2{{1, 2, 4}, {"int32_t"}};
        shape_type s3{{1, 2, 3}, {"int8_t"}};
        shape_type s4{{1, 2, 3}, {"int32_t"}};

        std::vector<std::string> axis_names{"dim1", "dim2", "dim3"};
        s4.set_names(axis_names);

        EXPECT_NE(s1, s2);
        EXPECT_NE(s1, s3);
        EXPECT_NE(s1, s4);
    }
}

TEST(typemap, shapetype_serialize)
{
    {
        shape_type               s1{{1, 2, 3}, {"int32_t"}};
        std::vector<std::string> axis_names{"dim1", "dim2", "dim3"};
        s1.set_names(axis_names);

        nlohmann::json    js = s1;
        std::stringstream is;
        is << js;

        // Check that the serialized representation matches expectation
        std::string ref =
            R"({"byte_size":24,"names":["dim1","dim2","dim3"],"otype":{"cv_type":4,"name":"int32_t","np_type":5,"size":4},"shape":[1,2,3]})";
        EXPECT_EQ(ref, is.str());

        // Check that we can deserialize and achieve equality
        shape_type s2 = js.get<shape_type>();
        EXPECT_EQ(s1, s2);
    }
}
