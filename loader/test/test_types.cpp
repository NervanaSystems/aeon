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
#include "typemap.hpp"
#include <typeinfo>
#include <typeindex>

using namespace std;
using namespace nervana;

TEST(typemap, numpy)
{
    {
        // auto opt = all_outputs.find("int8_t");
        // ASSERT_NE(opt, all_outputs.end());
        output_type opt{"int8_t"};
        EXPECT_EQ(CV_8S, opt.cv_type);
    }

}
